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
    iNaturalist,
    Places365,
    UniformNoise,
    GaussianNoise,
    TinyImages300k,
)
from pytorch_ood.utils import ToRGB
from pytorch_ood.utils import fix_random_seed, ToUnknown
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from tqdm import tqdm

from shared import extract_and_update_data

from dataset import CUB200

log = logging.getLogger(__name__)


def seed_worker(worker_id):
    fix_random_seed(worker_id)


def Model(backbone, num_classes=None, *args, **kwargs):
    model = resnet50(num_classes=1000, weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_oe_model(cfg, seed, trans, train_id_data: Subset):
    """
    train supervised model for OOD detection
    """
    log.info(f"Training model for OE")
    train_ood_data = TinyImages300k(
        root=cfg.paths.root,
        transform=trans,
        download=True,
        target_transform=lambda x: torch.tensor([0]),
    )

    train_data_ood, _, _ = random_split(
        train_ood_data,
        [10000, 10000, 280000],
        generator=torch.Generator().manual_seed(123),
    )

    train_id_data.dataset.target_transform = lambda x: torch.tensor([1])

    log.info(f"ID {len(train_id_data)} OOD {len(train_data_ood)}")

    # train_data = ConcatDataset((train_ood_data, train_data))

    train_loader = DataLoader(
        train_id_data + train_data_ood,
        batch_size=8,
        shuffle=True,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    model = Model(cfg.backbone, num_classes=2).to(cfg.device)
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
        batch_size=8,
        shuffle=True,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    model = Model(cfg.backbone, num_classes=num_classes).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

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

            log.info(f"{labels.unique().tolist()}")
            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            bar.set_postfix({"loss": running_loss, "lr": sched.get_last_lr()[0]})
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

    for att in attributes:
        target_path = join(cfg.paths.models, f"model-{att}-{seed:05d}.pt")
        if os.path.exists(target_path):
            log.info(f"Loading model from {target_path}")
            model = torch.load(target_path, map_location="cpu")
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

        # 11788 images
        log.info(f"Dataset: {cfg.paths.root}")
        data = CUB200(root=cfg.paths.root, transform=trans, attributes=cfg.attributes)
        log.info(f"Class: {data[1000][1][0]}")

        train_data, val_data, test_data = random_split(
            data, [10000, 1000, 788], generator=torch.Generator().manual_seed(seed)
        )

        models = load_models(cfg, cfg.attributes + ["oe"], seed)

        for index, att in enumerate(cfg.attributes):
            if att in models:
                log.info(f"Model for {att} already exists, skipping...")
                continue

            num_classes = 2 if att != "class" else 200
            log.info(f"Training model for {att} ({index=}) with {num_classes=}")

            model = train_model(
                cfg,
                att_index=index,
                num_classes=num_classes,
                train_data=train_data,
                val_data=val_data,
            )
            models[att] = model.cpu()
            torch.save(model, join(cfg.paths.models, f"model-{att}-{seed:05d}.pt"))

        if "oe" not in models:
            model = train_oe_model(cfg, seed, trans=trans, train_id_data=train_data)
            models["oe"] = model.cpu()
            torch.save(model, join(cfg.paths.models, f"model-oe-{seed:05d}.pt"))

        log.info(f"Extracting...")

        output_path = join(cfg.paths.predictions, f"data-train-{seed:05d}.pt")
        extract_and_update_data(cfg, models, output_path, train_data)

        output_path = join(cfg.paths.predictions, f"data-val-{seed:05d}.pt")
        extract_and_update_data(cfg, models, output_path, val_data)

        output_path = join(cfg.paths.predictions, f"data-test-{seed:05d}.pt")
        extract_and_update_data(cfg, models, output_path, test_data)

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

            output_path = join(
                cfg.paths.predictions, f"data-{dataset_name}-{seed:05d}.pt"
            )

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
