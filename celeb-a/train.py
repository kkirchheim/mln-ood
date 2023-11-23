import logging
from os.path import join

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from pytorch_ood.utils import ToRGB
from pytorch_ood.utils import fix_random_seed, extract_features, ToUnknown
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.models import (
    resnet18,
    wide_resnet50_2,
    Wide_ResNet50_2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    resnet50,
)
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from tqdm import tqdm

from dataset import CelebA
from shared import (
    extract_and_update_data,
    seed_worker,
    AccuracyMeter,
    ViTFeatureExtractor,
)

log = logging.getLogger(__name__)


def Model(backbone, num_classes=None, *args, **kwargs):
    #
    if backbone == "resnet18":
        model = resnet18(num_classes=1000, weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet50":
        model = resnet50(num_classes=1000, weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "wrn":
        model = wide_resnet50_2(
            num_classes=1000, weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "vit":
        model = ViTFeatureExtractor(num_classes=num_classes)
    else:
        raise ValueError()

    return model


@torch.no_grad()
def extract_logits(model, loader, device, att_name=None):
    model.eval()
    model.to(device)
    ys = []
    y_hats = []

    for x, y in tqdm(
        loader,
        desc="Extracting Logits" if not att_name else f"Extracting Logits {att_name}",
    ):
        logits = model(x.to(device))
        y_hats.append(model(x.to(device)).cpu())
        ys.append(y.to(device).cpu())

    y_hats = torch.cat(y_hats, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()
    return y_hats, ys


@torch.no_grad()
def extract_features(model, loader, device, att_name=None):
    model.eval()
    ys = []
    y_hats = []

    for x, y in tqdm(
        loader,
        desc=(
            "Extracting Features" if not att_name else f"Extracting Features {att_name}"
        ),
    ):
        x = x.to(device)

        if hasattr(model, "features"):
            y_hats.append(model.features(x).view(x.shape[0], -1).cpu())
        else:
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            y_hats.append(x.cpu())

        ys.append(y.cpu())

    y_hats = torch.cat(y_hats, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()
    return y_hats, ys


def train_oe_model(cfg, seed, trans):
    """
    train supervised model for OOD detection
    """

    data = CelebA(
        root=join(cfg.paths.root, "celeba"),
        transforms=trans,
        target_transform=lambda x: torch.tensor([1]),
    )

    train_data_in, val_data_in, _ = random_split(
        data, [150000, 2599, 50000], generator=torch.Generator().manual_seed(seed)
    )

    train_ood_data = TinyImages300k(
        root=cfg.paths.root,
        transform=trans,
        download=True,
        target_transform=lambda x: torch.tensor([0]),
    )

    train_data_ood, val_data_ood, _ = random_split(
        train_ood_data,
        [50000, 10000, 240000],
        generator=torch.Generator().manual_seed(123),
    )
    train_data = ConcatDataset((train_ood_data, train_data_in))
    val_data = ConcatDataset((val_data_ood, val_data_in))

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        worker_init_fn=seed_worker,
    )
    # val_loader = DataLoader(
    #     val_data,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=10,
    #     worker_init_fn=seed_worker,
    # )

    model = Model(backbone=cfg.backbone, num_classes=2).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for inputs, y in bar:
            labels = y[:, 0].long()
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            bar.set_postfix({"loss": running_loss, "lr": sched.get_last_lr()[0]})
            sched.step()

    return model


def train_model(cfg, att_index, num_classes, train_data, val_data):
    """
    train a model for the given attribute index
    """

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    model = Model(backbone=cfg.backbone, num_classes=num_classes).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))
    accuracy_meter = AccuracyMeter(window_size=50)

    for epoch in range(cfg.epochs):
        running_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for inputs, y in bar:
            labels = y[:, att_index].long()
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            # Compute predictions
            _, predicted = torch.max(outputs.data, dim=1)

            # Update accuracy meter with predictions and labels
            accuracy_meter.update(predicted, labels)
            window_accuracy = accuracy_meter.compute()

            # Update progress bar with loss and windowed accuracy
            bar.set_postfix(
                {
                    "loss": running_loss,
                    "accuracy": window_accuracy * 100,
                    "lr": sched.get_last_lr()[0],
                }
            )
            sched.step()

        correct = 0
        total = 0

        with torch.no_grad():
            model.eval()

            for inputs, y in val_loader:
                labels = y[:, att_index]
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        log.info(f"Accuracy of the network on the test images: {correct / total:.2%}")

    return model


def load_models(cfg, attributes, seed):
    """ """
    models = {}

    for att in attributes + ["oe"]:
        target_path = join(cfg.paths.models, f"model-{att}-{seed:05d}.pt")
        if os.path.exists(target_path):
            log.info(f"Loading model from {target_path}")
            model = torch.load(target_path, map_location="cpu", weights_only=False)
            models[att] = model.cpu()

    return models


@hydra.main(config_path="config", config_name="train.yaml", version_base="1.2")
def main(cfg):
    trans = Compose(
        [
            ToRGB(),
            ToTensor(),
            Resize((224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    os.makedirs(cfg.paths.predictions, exist_ok=True)
    os.makedirs(cfg.paths.models, exist_ok=True)

    for seed in range(cfg.n_seeds):
        fix_random_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        data = CelebA(root=join(cfg.paths.root, "celeba"), transforms=trans)
        train_data, val_data, test_data = random_split(
            data, [150000, 2599, 50000], generator=torch.Generator().manual_seed(seed)
        )

        # attributes = [a for a in data.data.columns if a != "image_id"]
        attributes = cfg.attributes
        models = load_models(cfg, [a.lower() for a in attributes], seed)

        for index, att in enumerate(attributes):
            if att.lower() in models:
                log.info(f"Model for {att} already exists, skipping...")
                continue

            log.info(f"Training model for {att}...")
            model = train_model(
                cfg,
                att_index=index,
                num_classes=2,
                train_data=train_data,
                val_data=val_data,
            )
            models[att] = model.cpu()
            torch.save(model, join(cfg.paths.models, f"model-{att}-{seed:05d}.pt"))

        if "oe" in models:
            log.info(f"Model for 'oe' already exists, skipping...")
        else:
            log.info(f"Training model for 'oe'...")
            model = train_oe_model(cfg, seed, trans)
            models["oe"] = model.cpu()
            torch.save(model, join(cfg.paths.models, f"model-oe-{seed:05d}.pt"))

        extract_all(cfg, models, seed, test_data, train_data, trans, val_data)


def extract_all(cfg, models, seed, test_data, train_data, trans, val_data):
    """ """
    output_path = join(cfg.paths.predictions, f"data-train-{seed:05d}.pt")
    if not os.path.exists(output_path):
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=10,
            worker_init_fn=seed_worker,
        )
        data = {}
        for att, model in models.items():
            logits, ys = extract_logits(model, train_loader, cfg.device)
            features, ys = extract_features(model, train_loader, cfg.device)
            data.update(
                {
                    f"{att}-logits": logits.half(),
                    f"{att}-features": features.half(),
                    f"{att}-labels": ys,
                }
            )
            model.cpu()
        torch.save(data, output_path)

    output_path = join(cfg.paths.predictions, f"data-val-{seed:05d}.pt")
    if not os.path.exists(output_path):
        val_loader = DataLoader(
            val_data,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=10,
            worker_init_fn=seed_worker,
        )
        data = {}
        for att, model in models.items():
            logits, ys = extract_logits(model, val_loader, cfg.device)
            features, ys = extract_features(model, val_loader, cfg.device)
            data.update(
                {
                    f"{att}-logits": logits.half(),
                    f"{att}-features": features.half(),
                    f"{att}-labels": ys,
                }
            )
            model.cpu()
        torch.save(data, output_path)

    output_path = join(cfg.paths.predictions, f"data-test-{seed:05d}.pt")
    if not os.path.exists(output_path):
        test_loader = DataLoader(
            test_data,
            batch_size=cfg.inference.batch_size,
            shuffle=False,
            num_workers=10,
            worker_init_fn=seed_worker,
        )
        data = {}
        for att, model in models.items():
            logits, ys = extract_logits(model, test_loader, cfg.device)
            features, ys = extract_features(model, test_loader, cfg.device)
            data.update(
                {
                    f"{att}-logits": logits.half(),
                    f"{att}-features": features.half(),
                    f"{att}-labels": ys,
                }
            )
            model.cpu()
        torch.save(data, output_path)

    datasets = {
        d.__name__: d
        for d in (
            LSUNCrop,
            LSUNResize,
            Textures,
            TinyImageNetCrop,
            TinyImageNetResize,
            iNaturalist,
            Places365,
            UniformNoise,
            GaussianNoise,
        )
    }

    for dataset_name, dataset_c in datasets.items():
        log.info(f"Extracting for {dataset_name}")

        output_path = join(cfg.paths.predictions, f"data-{dataset_name}-{seed:05d}.pt")

        if dataset_name in ["GaussianNoise", "UniformNoise"]:
            data_out = dataset_c(
                length=1000,
                transform=trans,
                target_transform=ToUnknown(),
            )
        else:
            data_out = dataset_c(
                root=cfg.paths.root,
                transform=trans,
                target_transform=ToUnknown(),
                download=True,
            )

        extract_and_update_data(cfg, models, output_path, data_out)


if __name__ == "__main__":
    main()
