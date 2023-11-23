import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from os.path import join

import hydra
import torch
from pytorch_ood.dataset.img import (
    LSUNCrop,
    LSUNResize,
    Textures,
    TinyImageNetCrop,
    TinyImageNetResize,
    TinyImages300k,
    Places365,
    iNaturalist,
    GaussianNoise,
    UniformNoise,
)
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import ToRGB
from pytorch_ood.utils import fix_random_seed
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
from torchvision.models.convnext import (
    convnext_small,
    ConvNeXt_Small_Weights,
)
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

from dataset import GTSRB
from utils import AccuracyMeter, ViTFeatureExtractor

log = logging.getLogger(__name__)

datasets = {
    d.__name__: d
    for d in (
        LSUNCrop,
        LSUNResize,
        Textures,
        TinyImageNetCrop,
        TinyImageNetResize,
        Places365,
        iNaturalist,
        GaussianNoise,
        UniformNoise,
    )
}


def seed_worker(worker_id):
    fix_random_seed(worker_id)


def get_model(backbone, num_classes):
    """Factory method for model creation based on backbone type."""
    models_map = {
        "resnet18": lambda: resnet18(
            num_classes=1000, weights=ResNet18_Weights.IMAGENET1K_V1
        ),
        "wrn40": lambda: WideResNet(num_classes=1000, pretrained="imagenet32"),
        "wrn40-scratch": lambda: WideResNet(num_classes=1000),
        "vit": lambda: ViTFeatureExtractor(num_classes=num_classes),
        "efficientnet_b0": lambda: efficientnet_b0(
            num_classes=1000, weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        ),
        "convnext_small": lambda: convnext_small(
            num_classes=1000, weights=ConvNeXt_Small_Weights.IMAGENET1K_V1
        ),
        "convnext_tiny": lambda: convnext_tiny(
            num_classes=1000, weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        ),
    }

    if backbone not in models_map:
        raise ValueError(f"Unknown backbone: {backbone}")

    model = models_map[backbone]()
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def extract_logits_or_features(model, loader, device, extract_features=False):
    """Unified method for extracting logits or features."""
    model.eval()
    ys, outputs = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            if extract_features and hasattr(model, "features"):
                out = model.features(x).view(x.shape[0], -1)
            elif extract_features:
                out = nn.Sequential(
                    model.conv1,
                    model.bn1,
                    model.relu,
                    model.maxpool,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4,
                    model.avgpool,
                    nn.Flatten(1),
                )(x)
            else:
                out = model(x)
        outputs.append(out.cpu())
        ys.append(y.cpu())

    return torch.cat(outputs, dim=0), torch.cat(ys, dim=0)


def prepare_dataloader(data, batch_size, is_train=True):
    """Prepare DataLoader for training or validation."""
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=10,
        worker_init_fn=seed_worker,
    )


def train_model(cfg, att_index, num_classes, trans):
    """Train a model for a specific attribute index."""
    data = GTSRB(root=cfg.paths.root, train=True, transforms=trans)
    train_data, val_data = random_split(
        data, [35000, 4209], generator=torch.Generator().manual_seed(123)
    )

    train_loader = prepare_dataloader(train_data, batch_size=32, is_train=True)
    val_loader = prepare_dataloader(val_data, batch_size=32, is_train=False)

    model = get_model(cfg.backbone, num_classes).to(cfg.device)
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    accuracy_meter = AccuracyMeter(window_size=50)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(cfg.device), labels[:, att_index].to(cfg.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            accuracy_meter.update(outputs.argmax(dim=1), labels)

        scheduler.step()

        validate_model(model, val_loader, cfg.device, att_index)

    return model


def validate_model(model, loader, device, att_index=None):
    """Validate the model on a dataset."""
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if att_index is not None:
                labels = labels[:, att_index]

            outputs = model(inputs)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    log.info(f"Validation Accuracy: {accuracy:.2%}")


def train_oe_model(cfg, trans):
    """Train the Outlier Exposure (OE) model."""
    in_data = GTSRB(
        root=cfg.paths.root,
        train=True,
        transforms=trans,
        target_transform=lambda x: torch.tensor([1]),
    )
    train_data_in, val_data_in = random_split(
        in_data, [35000, 4209], generator=torch.Generator().manual_seed(123)
    )

    ood_data = TinyImages300k(
        root=cfg.paths.root,
        transform=trans,
        target_transform=lambda x: torch.tensor([0]),
        download=True,
    )
    train_ood, val_ood, _ = random_split(
        ood_data, [50000, 10000, 240000], generator=torch.Generator().manual_seed(123)
    )

    train_data = ConcatDataset((train_ood, train_data_in))
    val_data = ConcatDataset((val_ood, val_data_in))

    train_loader = prepare_dataloader(train_data, batch_size=32, is_train=True)
    val_loader = prepare_dataloader(val_data, batch_size=32, is_train=False)

    model = get_model(cfg.backbone, num_classes=2).to(cfg.device)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)[:, 0]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        validate_model(model, val_loader, cfg.device)

    return model


def extract_and_save_model_data(models, loader, cfg, seed, data_type):
    """
    Extract logits and features from multiple models and save the data.
    """
    data = {}

    for model_name, model in models.items():
        logits, ys = extract_logits_or_features(
            model, loader, cfg.device, extract_features=False
        )
        features, _ = extract_logits_or_features(
            model, loader, cfg.device, extract_features=True
        )
        data[f"{model_name}-logits"] = logits
        data[f"{model_name}-features"] = features

    data["labels"] = ys
    torch.save(data, join(cfg.paths.predictions, f"data-{data_type}-{seed:05d}.pt"))


def create_dataset_loader(cfg, dataset_name, dataset_class, trans):
    """
    Create a DataLoader for a given dataset.
    """
    if dataset_name in ["GaussianNoise", "UniformNoise"]:
        dataset = dataset_class(
            length=1000, transform=trans, target_transform=lambda x: torch.tensor([-1])
        )
    else:
        dataset = dataset_class(
            root=cfg.paths.root,
            transform=trans,
            target_transform=lambda x: torch.tensor([-1]),
            download=True,
        )

    return prepare_dataloader(dataset, batch_size=128, is_train=False)


@hydra.main(config_path="config", config_name="train.yaml", version_base="1.2")
def main(cfg):
    trans = Compose([ToRGB(), ToTensor(), Resize(cfg.image_size, antialias=True)])

    os.makedirs(cfg.paths.predictions)
    os.makedirs(cfg.paths.models)

    for seed in range(cfg.n_seeds):
        fix_random_seed(seed)

        # Train models and store in a dictionary
        models = {
            "oe": train_oe_model(cfg, trans=trans),
            "shape": train_model(cfg, att_index=2, num_classes=5, trans=trans),
            "color": train_model(cfg, att_index=1, num_classes=4, trans=trans),
            "label": train_model(cfg, att_index=0, num_classes=43, trans=trans),
        }

        # Save trained models
        log.info("Saving models...")
        for model_name, model in models.items():
            torch.save(
                model, join(cfg.paths.models, f"model-{model_name}-{seed:05d}.pt")
            )

        # Prepare training and validation data loaders
        data = GTSRB(root=cfg.paths.root, train=True, transforms=trans)
        train_data, val_data = random_split(
            data, [35000, 4209], generator=torch.Generator().manual_seed(123)
        )
        train_loader = prepare_dataloader(train_data, batch_size=32, is_train=False)
        val_loader = prepare_dataloader(val_data, batch_size=32, is_train=False)

        # Extract training data
        log.info("Extracting training data...")
        extract_and_save_model_data(models, train_loader, cfg, seed, "train")

        # Extract validation data
        log.info("Extracting validation data...")
        extract_and_save_model_data(models, val_loader, cfg, seed, "val")

        # Prepare test data loader
        test_data = GTSRB(root=cfg.paths.root, train=False, transforms=trans)
        test_loader = prepare_dataloader(test_data, batch_size=32, is_train=False)

        # Extract test data
        log.info("Extracting test data...")
        extract_and_save_model_data(models, test_loader, cfg, seed, "test")

        # Extract data for other datasets
        log.info("Extracting data for additional datasets...")
        for dataset_name, dataset_class in datasets.items():
            log.info(f"Processing dataset: {dataset_name}")
            dataset_loader = create_dataset_loader(
                cfg, dataset_name, dataset_class, trans
            )
            extract_and_save_model_data(models, dataset_loader, cfg, seed, dataset_name)


if __name__ == "__main__":
    main()
