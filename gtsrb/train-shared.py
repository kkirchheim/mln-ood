import logging
import os
from os.path import join

import hydra
import torch
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import ToRGB
from pytorch_ood.utils import fix_random_seed
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

from dataset import GTSRB

from train import (
    seed_worker,
    datasets,
    extract_and_save_model_data,
    prepare_dataloader,
    create_dataset_loader,
)

log = logging.getLogger(__name__)


def MultiHeadModel(num_classes=None, *args, **kwargs):
    model = WideResNet(*args, num_classes=1000, pretrained="imagenet32", **kwargs)

    for n, n_c in enumerate(num_classes):
        setattr(model, f"fc{n}", nn.Linear(model.fc.in_features, n_c))

    del model.fc

    return model


def train_model(cfg, num_classes, trans):
    """
    train a model for the given attribute index
    """
    data = GTSRB(root=cfg.paths.root, train=True, transforms=trans)
    train_data, val_data = random_split(
        data, [35000, 4209], generator=torch.Generator().manual_seed(123)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=10,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        worker_init_fn=seed_worker,
    )

    model = MultiHeadModel(num_classes=num_classes).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    sched = CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

    train_model_on_data(
        cfg, criterion, model, optimizer, sched, train_loader, val_loader
    )
    return model


def train_model_on_data(
    cfg, criterion, model, optimizer, sched, train_loader, val_loader
):
    """ """
    for epoch in range(cfg.epochs):
        running_loss = 0.0
        model.train()
        bar = tqdm(train_loader)
        for inputs, y in bar:
            labels = y
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

            optimizer.zero_grad()

            z = model.features(inputs)
            loss1 = criterion(model.fc0(z), labels[:, 0])
            loss2 = criterion(model.fc1(z), labels[:, 1])
            loss3 = criterion(model.fc2(z), labels[:, 2])
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()

            running_loss = 0.8 * running_loss + 0.2 * loss.item()
            bar.set_postfix({"loss": running_loss, "lr": sched.get_last_lr()[0]})
            sched.step()

        correct1 = 0
        correct2 = 0
        correct3 = 0
        total = 0

        with torch.no_grad():
            model.eval()

            for inputs, y in val_loader:
                labels = y
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)

                z = model.features(inputs)
                y1 = model.fc0(z).max(dim=1)[1]
                y2 = model.fc1(z).max(dim=1)[1]
                y3 = model.fc2(z).max(dim=1)[1]

                total += labels.size(0)
                correct1 += (y1 == labels[:, 0]).sum().item()
                correct2 += (y2 == labels[:, 1]).sum().item()
                correct3 += (y3 == labels[:, 2]).sum().item()

        log.info(
            f"Accuracy of the network on the test images: {correct1 / total:.2%} {correct2 / total:.2%} {correct3 / total:.2%}"
        )


class ModelWrapper(nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net = net
        self.head = head

    def features(self, x):
        return self.net.features(x)

    def forward(self, x):
        z = self.features(x)
        return self.head(z)


@hydra.main(config_path="config", config_name="train-shared.yaml", version_base="1.2")
def main(cfg):
    trans = Compose([ToRGB(), ToTensor(), Resize(cfg.image_size, antialias=True)])

    os.makedirs(cfg.paths.predictions)
    os.makedirs(cfg.paths.models)

    for seed in range(cfg.n_seeds):
        fix_random_seed(seed)

        net = train_model(cfg, num_classes=[43, 4, 5], trans=trans)

        log.info(f"Saving models...")
        torch.save(net, join(cfg.paths.models, f"model-shared-{seed:05d}.pt"))

        log.info(f"Extracting training stuff")
        net.eval()

        models = {
            "shape": ModelWrapper(net, net.fc1),
            "color": ModelWrapper(net, net.fc2),
            "label": ModelWrapper(net, net.fc0),
        }

        # Save trained models
        # log.info("Saving models...")
        # for model_name, model in models.items():
        #     torch.save(
        #         model, join(cfg.paths.models, f"model-{model_name}-{seed:05d}.pt")
        #     )

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
