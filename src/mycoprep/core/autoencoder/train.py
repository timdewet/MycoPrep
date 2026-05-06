"""Training loop for the cell morphology autoencoder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import AutoencoderConfig
from .dataset import HDF5CropDataset, split_by_fov

ProgressCB = Callable[[float, str], None]


def _detect_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collate_crops(batch):
    """Custom collate: stack crops, pass metadata as list of dicts."""
    crops = torch.stack([item[0] for item in batch])
    metas = [item[1] for item in batch]
    return crops, metas


def _gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    """Build a normalised 2D Gaussian kernel on *device*."""
    ksize = int(2 * round(3 * sigma) + 1)
    if ksize < 3:
        ksize = 3
    ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
    k1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    k1d = k1d / k1d.sum()
    return k1d.unsqueeze(1) @ k1d.unsqueeze(0)


def _augment_batch_gpu(
    crops: torch.Tensor, config: AutoencoderConfig
) -> torch.Tensor:
    """Batched augmentation on the GPU.

    Operates on a (B, C, H, W) tensor already on the device. Per-batch
    rotation/flip/blur (one random choice per batch — across many batches
    over training the augmentation distribution is still wide). Per-image
    brightness/contrast (cheap, vectorised).
    """
    if not config.augment:
        return crops

    if config.rotation:
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            crops = torch.rot90(crops, k, dims=(2, 3))

    if config.flip:
        if torch.rand(1).item() > 0.5:
            crops = crops.flip(dims=(2,))
        if torch.rand(1).item() > 0.5:
            crops = crops.flip(dims=(3,))

    b = crops.shape[0]
    if config.brightness_jitter > 0:
        delta = (
            (torch.rand(b, 1, 1, 1, device=crops.device) * 2 - 1)
            * config.brightness_jitter
        )
        crops = (crops + delta).clamp(0, 1)

    if config.contrast_jitter > 0:
        factor = 1.0 + (
            (torch.rand(b, 1, 1, 1, device=crops.device) * 2 - 1)
            * config.contrast_jitter
        )
        mean = crops.mean(dim=(2, 3), keepdim=True)
        crops = ((crops - mean) * factor + mean).clamp(0, 1)

    if config.gaussian_blur_sigma > 0:
        sigma = float(torch.rand(1).item() * config.gaussian_blur_sigma)
        if sigma > 0.1:
            kernel = _gaussian_kernel_2d(sigma, crops.device)
            ksize = kernel.shape[-1]
            c = crops.shape[1]
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(c, 1, -1, -1)
            crops = torch.nn.functional.conv2d(
                crops, kernel, padding=ksize // 2, groups=c
            )

    return crops


def train_autoencoder(
    h5_paths: Sequence[Path],
    output_dir: Path,
    config: AutoencoderConfig,
    *,
    device: Optional[str] = None,
    progress_cb: Optional[ProgressCB] = None,
    fine_tune: bool = False,
) -> dict:
    """Train (or fine-tune) the autoencoder.

    Args:
        h5_paths: HDF5 crop files to train on.
        output_dir: Where to save model checkpoints and artifacts.
        config: Training configuration.
        device: Torch device string (auto-detected if None).
        progress_cb: Called with (fraction, message) for GUI progress bars.
        fine_tune: If True, uses fine_tune_epochs/lr and loads config.model_path.

    Returns:
        Summary dict with training metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device(device)

    if progress_cb is None:
        progress_cb = lambda f, m: None

    epochs = config.fine_tune_epochs if fine_tune else config.epochs
    lr = config.fine_tune_lr if fine_tune else config.lr

    progress_cb(0.0, "Splitting data by FOV...")
    train_idx, val_idx = split_by_fov(h5_paths, config.val_fraction)

    train_ds = HDF5CropDataset(
        h5_paths,
        target_size=config.crop_size,
        in_channels=config.in_channels,
        augment=config.augment,
        rotation=config.rotation,
        flip=config.flip,
        brightness_jitter=config.brightness_jitter,
        contrast_jitter=config.contrast_jitter,
        gaussian_blur_sigma=config.gaussian_blur_sigma,
        indices=train_idx,
    )
    val_ds = HDF5CropDataset(
        h5_paths,
        target_size=config.crop_size,
        in_channels=config.in_channels,
        augment=False,
        indices=val_idx,
    )

    if getattr(train_ds, "_use_cache", False):
        cache_gb = getattr(train_ds, "_cache_bytes", 0) / 1e9
        progress_cb(
            0.02,
            f"Train: {len(train_ds)} cells, Val: {len(val_ds)} cells "
            f"(crops cached in RAM, {cache_gb:.1f} GB)",
        )
    else:
        progress_cb(
            0.02,
            f"Train: {len(train_ds)} cells, Val: {len(val_ds)} cells "
            f"(streaming from HDF5)",
        )

    # When the dataset is cached in RAM, ``__getitem__`` is just a numpy
    # slice — workers add multiprocessing overhead without any benefit.
    # When streaming from HDF5, multiple workers parallelise the slow
    # random-access reads.
    cached = bool(getattr(train_ds, "_use_cache", False))
    if cached:
        num_workers = 0
    else:
        import os
        cpu_count = os.cpu_count() or 4
        num_workers = max(2, min(8, cpu_count - 1))
    # pin_memory is a CUDA host→device optimisation; on MPS (unified
    # memory) it's pointless and can even slow things down.
    use_pin = device == "cuda"
    loader_kwargs: dict = dict(
        batch_size=config.batch_size,
        num_workers=num_workers,
        collate_fn=_collate_crops,
        pin_memory=use_pin,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Build or load model
    model = config.build_model()
    if fine_tune and config.model_path and config.model_path.exists():
        state = torch.load(str(config.model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        progress_cb(0.03, f"Loaded model from {config.model_path.name}")

    model = model.to(device)

    # Freeze strategy (full training only)
    if not fine_tune and config.freeze_epochs > 0 and hasattr(model, "freeze_early_layers"):
        model.freeze_early_layers()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.time()

    n_train_batches = len(train_loader)
    # Within an epoch, train is the bulk of the work; reserve a sliver for val
    # in the overall fraction calculation.
    train_frac_of_epoch = 0.85
    # ~4 train updates per epoch keeps the log readable; the per-epoch
    # summary at the end covers the rest. Validation gets no per-batch
    # updates — the summary line reports its loss.
    train_report_every = max(10, n_train_batches // 4)

    progress_cb(
        0.04,
        f"Starting training on {device.upper()} — "
        f"{n_train_batches} train batches × {epochs} epochs",
    )

    for epoch in range(epochs):
        # Unfreeze after freeze_epochs
        if (
            not fine_tune
            and epoch == config.freeze_epochs
            and hasattr(model, "unfreeze_all")
        ):
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=config.weight_decay
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - epoch)
            progress_cb(
                epoch / epochs, f"Epoch {epoch}: unfreezing all layers"
            )

        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_t0 = time.time()
        # Notify before the first batch so the user sees activity even if
        # the first batch takes a while (DataLoader workers warmup, MPS
        # compile, etc).
        progress_cb(
            (epoch + 0.0) / epochs,
            f"Epoch {epoch + 1}/{epochs}: starting training "
            f"({n_train_batches} batches)...",
        )
        for batch_idx, (crops, _) in enumerate(train_loader):
            crops = crops.to(device, non_blocking=True)
            crops = _augment_batch_gpu(crops, config)
            recon, _ = model(crops)
            loss = criterion(recon, crops)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # Per-batch progress: gives the user a heartbeat AND lets the
            # Stop button take effect (the runner's cb raises StopRequested
            # when stop has been requested).
            if (batch_idx + 1) % train_report_every == 0 or batch_idx + 1 == n_train_batches:
                batch_t = time.time() - epoch_t0
                avg_per_batch = batch_t / (batch_idx + 1)
                remaining_batches = n_train_batches - (batch_idx + 1)
                eta_epoch = remaining_batches * avg_per_batch
                # Total ETA: this epoch's remaining + (epochs - epoch - 1) full epochs
                eta_total = eta_epoch + avg_per_batch * n_train_batches * (epochs - epoch - 1)
                within = (batch_idx + 1) / n_train_batches * train_frac_of_epoch
                overall = (epoch + within) / epochs
                progress_cb(
                    overall,
                    f"Epoch {epoch + 1}/{epochs}: train "
                    f"{batch_idx + 1}/{n_train_batches} "
                    f"loss={loss.item():.4f} "
                    f"({batch_t:.0f}s, ETA epoch {eta_epoch:.0f}s, "
                    f"total {eta_total / 60:.1f}min)"
                )

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for crops, _ in val_loader:
                crops = crops.to(device, non_blocking=True)
                # No augmentation on val.
                recon, _ = model(crops)
                val_loss += criterion(recon, crops).item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)
        scheduler.step()

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        frac = (epoch + 1) / epochs
        elapsed = time.time() - t0
        progress_cb(
            0.05 + 0.90 * frac,
            f"Epoch {epoch + 1}/{epochs} done — "
            f"train: {train_loss:.5f}, val: {val_loss:.5f} "
            f"(epoch {time.time() - epoch_t0:.0f}s, total {elapsed / 60:.1f}min)",
        )

    # Save final model and artifacts
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    config_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config.__dict__.items()
    }
    config_dict["device"] = device
    config_dict["train_cells"] = len(train_ds)
    config_dict["val_cells"] = len(val_ds)
    config_dict["h5_paths"] = [str(p) for p in h5_paths]
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    progress_cb(0.97, "Saving training curves...")
    _save_training_curves(train_losses, val_losses, output_dir)

    progress_cb(0.99, "Saving reconstruction samples...")
    _save_reconstructions(model, val_loader, device, output_dir)

    progress_cb(1.0, "Training complete.")

    summary = {
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "epochs_trained": epochs,
        "train_cells": len(train_ds),
        "val_cells": len(val_ds),
        "model_path": str(output_dir / "best_model.pth"),
        "elapsed_seconds": time.time() - t0,
    }

    # Best-effort cleanup — runs even if the caller interrupts via
    # StopRequested raised from a progress callback above.
    for ds in (train_ds, val_ds):
        try:
            ds.close()
        except Exception:  # noqa: BLE001
            pass

    return summary


def _save_training_curves(
    train_losses: list[float], val_losses: list[float], output_dir: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_losses, label="Train")
        ax.plot(val_losses, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Autoencoder Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def _save_reconstructions(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    output_dir: Path,
    n_samples: int = 8,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model.eval()
        samples = []
        with torch.no_grad():
            for crops, _ in val_loader:
                crops = crops.to(device)
                recon, _ = model(crops)
                for i in range(min(n_samples - len(samples), crops.shape[0])):
                    samples.append((crops[i].cpu(), recon[i].cpu()))
                if len(samples) >= n_samples:
                    break

        if not samples:
            return

        n = len(samples)
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i, (orig, rec) in enumerate(samples):
            # Show first channel
            axes[0, i].imshow(orig[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[0, i].set_title("Input" if i == 0 else "")
            axes[0, i].axis("off")
            axes[1, i].imshow(rec[0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].set_title("Recon" if i == 0 else "")
            axes[1, i].axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / "reconstructions.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
