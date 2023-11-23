from os.path import join

import hydra
import torch
from pytorch_ood.dataset.img import (
    TinyImageNetCrop,
    Textures,
    LSUNResize,
    LSUNCrop,
    TinyImageNetResize,
    iNaturalist,
    Places365,
    UniformNoise,
    GaussianNoise,
)
from pytorch_ood.utils import fix_random_seed, ToUnknown
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import os
from collections import deque
import time
import logging
from torch.optim import LBFGS
import functools

log = logging.getLogger(__name__)


def get_output_dir():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


def get_machine_lock_file():
    return f"/tmp/lock-{os.environ['SLURM_JOB_ID']}.lock"


class Timer:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.perf_counter() - self.start_time
        logging.info(f"{self.task_name} took {elapsed_time:.4f} seconds")


@torch.no_grad()
def extract_logits(model, loader, device):
    model.eval()
    model.to(device)
    ys = []
    y_hats = []

    for x, y in tqdm(loader, desc="Extracting Logits"):
        y_hats.append(model(x.to(device)))
        ys.append(y.to(device))

    y_hats = torch.cat(y_hats, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()
    return y_hats, ys


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    ys = []
    y_hats = []

    for x, y in tqdm(loader, desc="Extracting Features"):
        x = x.to(device)

        if hasattr(model, "features"):
            y_hats.append(model.features(x))
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

            y_hats.append(x)

        ys.append(y.to(device))

    y_hats = torch.cat(y_hats, dim=0).cpu()
    ys = torch.cat(ys, dim=0).cpu()
    return y_hats, ys


def seed_worker(worker_id):
    fix_random_seed(worker_id)


def extract_and_update_data(cfg, models, output_path, data):
    train_loader = DataLoader(
        data,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        worker_init_fn=seed_worker,
    )
    modified = False

    if not os.path.exists(output_path):
        data = {}
    else:
        data = torch.load(output_path)

    data = {k.lower(): v for k, v in data.items()}

    log.info(list(data.keys()))

    for att, model in models.items():
        if f"{att}-logits" not in data:
            modified = True
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

    if modified:
        torch.save(data, output_path)


def pseudo_log_prob(model, x):
    """
    Compute the pseudo log likelihood for a batch of states x.
    x: Tensor of shape (B, D)
    """
    B, D = x.shape
    pseudo_ll = 0.0
    for i in range(D):
        # Get the candidate values for dimension i as a tensor.
        domain_i = model.domain[i]
        candidate_tensor = torch.tensor(
            domain_i, dtype=x.dtype, device=x.device
        )  # shape (K,)
        K = candidate_tensor.shape[0]

        # Expand x to shape (B, K, D) and replace column i with all candidates.
        x_alt = x.unsqueeze(1).repeat(1, K, 1)  # (B, K, D)
        x_alt[:, :, i] = candidate_tensor.unsqueeze(0).repeat(B, 1)

        # Flatten to (B*K, D) and compute energies in one go.
        energies = model.energy(x_alt.view(B * K, D)).view(B, K)  # shape (B, K)

        # For each sample, determine which candidate matches the observed value.
        eq = x[:, i].unsqueeze(1) == candidate_tensor.unsqueeze(0)  # (B, K)
        indices = eq.float().argmax(dim=1)  # (B,)

        # Gather the energy corresponding to the observed value.
        energy_obs = energies.gather(dim=1, index=indices.unsqueeze(1)).squeeze(1)
        log_energy_obs = torch.log(energy_obs)
        log_den = torch.log(energies.sum(dim=1))
        log_p = log_energy_obs - log_den  # log p(x_i | x_{-i})
        pseudo_ll += log_p.sum()

    return pseudo_ll


def train_mln_pseudo(cfg, model, data, use_prog=True):
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.mln.batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.stack([y[0] for y in x]),
        num_workers=0,
    )

    max_iter = 20
    optimizer = LBFGS(model.parameters(), lr=cfg.mln.lr, max_iter=max_iter)
    model.train()
    model.to(cfg.device)

    if use_prog:
        bar = tqdm(total=cfg.mln.epochs * max_iter)

    for epoch in range(cfg.mln.epochs):
        for batch_no, x in enumerate(loader):
            x = x.to(cfg.device)

            def closure():
                optimizer.zero_grad()
                pll = pseudo_log_prob(model, x)
                loss = -pll / x.shape[0]  # average negative pseudo log likelihood
                loss.backward()

                if use_prog:
                    bar.set_postfix_str(
                        f"loss: {loss.item():.2f} epoch: {epoch} batch: {batch_no}"
                    )
                    bar.update(1)

                # log.info(f"{model.w}")
                return loss

            optimizer.step(closure)


def train_mln(cfg, model, data, use_prog=True):
    dataset = TensorDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.mln.batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.stack([y[0] for y in x]),
        num_workers=0,
    )

    max_iter = 20
    optimizer = LBFGS(model.parameters(), lr=cfg.mln.lr, max_iter=max_iter)

    model.train()
    model.to(cfg.device)

    if use_prog:
        bar = tqdm(range(cfg.mln.epochs), total=cfg.mln.epochs * max_iter)

    with logging_redirect_tqdm():
        for epoch in range(cfg.mln.epochs):
            total_loss = 0
            for x in loader:

                def closure():
                    optimizer.zero_grad()
                    prob = model.prob(x.to(cfg.device))
                    losses = -torch.log(prob)
                    loss = losses.mean()
                    loss.backward()

                    if use_prog:
                        bar.set_postfix_str(
                            f"loss: {losses.sum().item() / x.shape[0]:.2f}"
                        )
                        bar.update(1)

                    # log.info(f"{model.w}")

                    return loss

                optimizer.step(closure)

    return total_loss


class AccuracyMeter:
    def __init__(self, window_size=50):
        """
        Initializes the accuracy meter with a specified window size.

        Parameters:
        - window_size (int): The number of batches to keep in the moving window.
        """
        self.window_size = window_size
        self.correct_window = deque(maxlen=window_size)
        self.total_window = deque(maxlen=window_size)

    def update(self, preds, labels):
        """
        Updates the moving window with the latest batch's predictions and labels.

        Parameters:
        - preds (torch.Tensor): Predicted labels.
        - labels (torch.Tensor): True labels.
        """
        # Calculate the number of correct predictions
        correct_count = (preds == labels).sum().item()
        total_count = labels.size(0)

        # Update the deque with the new values
        self.correct_window.append(correct_count)
        self.total_window.append(total_count)

    def compute(self):
        """
        Computes the accuracy over the current window.

        Returns:
        - (float): Accuracy over the current window.
        """
        # Sum the values in the window to get total correct and total instances
        window_correct = sum(self.correct_window)
        window_total = sum(self.total_window)

        # Calculate accuracy, handling the case where window_total is zero
        return window_correct / window_total if window_total > 0 else 0.0

    def reset(self):
        """
        Resets the accuracy meter.
        """
        self.correct_window.clear()
        self.total_window.clear()


import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch import nn


class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, num_classes):
        super(ViTFeatureExtractor, self).__init__()
        self.trafo = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.trafo.heads[-1] = nn.Linear(768, num_classes)

    def features(self, x):
        x = self.trafo._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.trafo.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.trafo.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self.trafo._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.trafo.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.trafo.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.trafo.heads(x)

        return x


def file_lock(lock_filename, sleep_interval=0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to acquire the lock by creating the lock file atomically.
            while True:
                try:
                    fd = os.open(lock_filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    log.info(f"Lock file {lock_filename} acquired.")
                    break  # Lock acquired.
                except FileExistsError:
                    time.sleep(sleep_interval)  # Wait and retry if lock exists.
            try:
                return func(*args, **kwargs)
            finally:
                os.close(fd)
                os.remove(lock_filename)

        return wrapper

    return decorator


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
