"""Supervised Contrastive Learning (SupCon) loss and training loop.

Reference: Khosla et al. 2020, "Supervised Contrastive Learning"
(https://arxiv.org/abs/2004.11362).

SupCon trains the encoder so that cells from the same class (here, same
gene knockdown) cluster together in the projection space and cells from
different classes are pushed apart. Compared to a vanilla autoencoder,
which optimises pixel reconstruction with no class signal, SupCon directly
shapes the embedding space for the discrimination task — at the cost of
needing labels.

Standard SupCon practice is to feed two augmented "views" of each image
through the encoder + projection head, then compute the loss treating any
two same-class samples (across both views) as a positive pair. After
training, the projection head is discarded and the 512-d encoder features
are used as the embedding for downstream analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import AutoencoderConfig
from .dataset import HDF5CropDataset, split_by_fov
from .train import _augment_batch_gpu, _collate_crops, _detect_device

ProgressCB = Callable[[float, str], None]


class SupConLoss(nn.Module):
    """Supervised contrastive loss.

    For each anchor, positives are all other samples sharing the anchor's
    class. The loss maximises log-prob of positive pairs against all other
    pairs in the batch (excluding self).

    Anchors with no positives present in the batch are dropped from the
    mean (they would otherwise contribute zero gradient and divide by zero).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalised projection vectors.
            labels:   (B,) integer class labels.
        """
        device = features.device
        B = features.shape[0]

        # (B, B) cosine-similarity matrix (features are already unit-norm).
        sim = features @ features.T / self.temperature

        # Numerical stability: subtract row-wise max before exp.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        labels = labels.contiguous().view(-1, 1)
        # (B, B) mask: 1 where labels match, 0 elsewhere.
        same_class = labels.eq(labels.T).float()
        # Exclude self-pairs.
        eye = torch.eye(B, device=device)
        positive_mask = same_class - eye  # 1 for positive pairs, 0 elsewhere
        non_self_mask = 1.0 - eye

        exp_sim = torch.exp(sim) * non_self_mask
        # Denominator = sum over all non-self pairs (positives + negatives).
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        log_prob = sim - log_denom  # (B, B)

        n_pos = positive_mask.sum(dim=1)
        # Per-anchor mean log-prob over its positives.
        # Avoid div-by-zero by clamping; we filter no-positive anchors below.
        mean_log_prob_pos = (
            (positive_mask * log_prob).sum(dim=1)
            / n_pos.clamp(min=1.0)
        )

        valid = n_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        return -mean_log_prob_pos[valid].mean()


def _peek_checkpoint_in_channels(state: dict) -> Optional[int]:
    """Return the input-channel count of a saved SupCon/AE checkpoint.

    Looks at the conv1 weight (shape ``(out_c, in_c, k, k)``) at any of
    the keys we use across architectures: ResNet-18 wraps the encoder
    in ``encoder.0`` (Sequential), the lightweight encoder uses
    ``encoder.0.0``. Returns None if no recognised key is present.
    """
    candidates = (
        "encoder.0.weight",       # ResNet-18 encoder Sequential[0] = conv1
        "encoder.0.0.weight",     # Lightweight encoder Sequential[0] = first block conv
    )
    for key in candidates:
        if key in state:
            shape = tuple(state[key].shape)
            if len(shape) == 4:
                return int(shape[1])
    return None


def _build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
):
    """Linear warmup → cosine decay learning-rate schedule.

    ``warmup_epochs == 0`` collapses to pure cosine. Mirrors the Mtb
    reference's ``warmup_epochs=5`` then cosine.
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    warmup = LambdaLR(
        optimizer,
        lr_lambda=lambda e: (e + 1) / float(warmup_epochs),
    )
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )


def _resolve_class_label(meta: dict, source: str = "auto") -> str:
    """Pick the SupCon class label for a single cell.

    ``source = "auto"`` (default) prefers the H5 ``genes`` field when
    populated, falls back to ``drugs`` (combined with concentration so
    each drug-dose is its own class), then to the full ``condition_label``.
    ``source = "gene"`` / ``"drug"`` forces the choice.
    """
    if source in ("auto", "gene"):
        g = str(meta.get("gene", "") or "").strip()
        if g:
            return g
        if source == "gene":
            return ""
    if source in ("auto", "drug"):
        d = str(meta.get("drug", "") or "").strip()
        if d:
            conc = str(meta.get("concentration", "") or "").strip()
            return f"{d}_{conc}" if conc else d
        if source == "drug":
            return ""
    # Last resort: the full condition label, NOT just the first token,
    # so labels like "ATc+ rpoB" don't collapse all "ATc+" cells into
    # one class.
    return str(meta.get("condition_label", "") or "").strip()


def _per_cell_class_labels(
    dataset: HDF5CropDataset, source: str = "auto",
) -> list[str]:
    """Return one class label per cell in the dataset's index order."""
    n = len(dataset)
    out: list[str] = [""] * n
    if hasattr(dataset, "_cached_meta"):
        for i in range(n):
            global_idx = int(dataset._indices[i])
            file_idx, local_idx = dataset._global_to_file(global_idx)
            mdict = dataset._cached_meta[file_idx]
            stub = {
                "condition_label": (
                    str(mdict["condition_labels"][local_idx])
                    if "condition_labels" in mdict else ""
                ),
                "gene": (
                    str(mdict["genes"][local_idx])
                    if "genes" in mdict else ""
                ),
                "drug": (
                    str(mdict["drugs"][local_idx])
                    if "drugs" in mdict else ""
                ),
                "concentration": (
                    str(mdict["concentrations"][local_idx])
                    if "concentrations" in mdict else ""
                ),
            }
            out[i] = _resolve_class_label(stub, source)
        return out
    for i in range(n):
        _, m = dataset[i]
        out[i] = _resolve_class_label(m, source)
    return out


def _build_label_map(
    dataset: HDF5CropDataset,
    source: str = "auto",
    min_cells_per_class: int = 1,
) -> tuple[dict[str, int], list[str]]:
    """Enumerate unique class labels and return ``(label_map, per_cell_labels)``.

    Classes with fewer than ``min_cells_per_class`` cells are dropped
    from the map; their cells get an empty-string label and the SupCon
    training loop will filter them out via ``class_kept_mask``.
    """
    per_cell = _per_cell_class_labels(dataset, source=source)
    counts: dict[str, int] = {}
    for c in per_cell:
        if not c:
            continue
        counts[c] = counts.get(c, 0) + 1
    keep = sorted(c for c, n in counts.items() if n >= min_cells_per_class)
    label_map = {c: i for i, c in enumerate(keep)}
    return label_map, per_cell


def train_supcon(
    h5_paths: Sequence[Path],
    output_dir: Path,
    config: AutoencoderConfig,
    *,
    device: Optional[str] = None,
    progress_cb: Optional[ProgressCB] = None,
    fine_tune: bool = False,
) -> dict:
    """Train a ResNet-18 + projection head with the SupCon loss.

    The class label for each cell is the gene token of its
    ``condition_label`` (e.g. ``"NT ATc+"`` → ``"NT"``). Two augmented
    views of each crop are fed through the encoder + projection head; the
    SupCon loss then pulls together same-gene projections and pushes apart
    different-gene projections across both views.

    Returns a summary dict with ``model_path`` pointing to the trained
    encoder weights — the projection head is discarded.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device(device)

    if progress_cb is None:
        progress_cb = lambda f, m: None

    epochs = config.fine_tune_epochs if fine_tune else config.epochs
    lr = config.fine_tune_lr if fine_tune else config.lr
    temperature = float(getattr(config, "temperature", 0.07))
    projection_dim = int(getattr(config, "projection_dim", 128))

    progress_cb(0.0, "Splitting data by FOV...")
    train_idx, val_idx = split_by_fov(h5_paths, config.val_fraction)

    train_ds = HDF5CropDataset(
        h5_paths,
        target_size=config.crop_size,
        in_channels=config.in_channels,
        image_channels=config.image_channels,
        include_mask=config.include_mask,
        augment=False,  # GPU-batched augmentation in the loop produces 2 views
        indices=train_idx,
    )
    val_ds = HDF5CropDataset(
        h5_paths,
        target_size=config.crop_size,
        in_channels=config.in_channels,
        image_channels=config.image_channels,
        include_mask=config.include_mask,
        augment=False,
        indices=val_idx,
    )

    label_source = str(getattr(config, "supcon_label_source", "auto"))
    min_cells = max(1, int(getattr(config, "min_cells_per_class", 50)))
    label_map, train_per_cell = _build_label_map(
        train_ds, source=label_source, min_cells_per_class=min_cells,
    )
    n_classes = len(label_map)
    if n_classes < 2:
        raise RuntimeError(
            f"SupCon needs ≥ 2 distinct class labels in the training data "
            f"(after dropping classes with < {min_cells} cells), found "
            f"{n_classes}. Check that the H5 'genes' / 'drugs' datasets are "
            f"populated, or lower min_cells_per_class."
        )

    # Per-cell integer labels for the kept classes; -1 marks cells
    # whose class was dropped, so they're excluded from sampling.
    train_int_labels = np.array(
        [label_map.get(c, -1) for c in train_per_cell], dtype=np.int64,
    )
    kept_mask = train_int_labels >= 0
    n_dropped = int((~kept_mask).sum())

    # Class-frequency stats — surfaced upfront so weird splits are visible.
    class_counts: dict[int, int] = {}
    for k in train_int_labels[kept_mask]:
        class_counts[int(k)] = class_counts.get(int(k), 0) + 1
    counts_arr = np.array(list(class_counts.values()), dtype=np.int64)
    progress_cb(
        0.02,
        f"Train: {len(train_ds)} cells ({n_dropped} dropped, "
        f"{kept_mask.sum()} kept), Val: {len(val_ds)} cells, "
        f"{n_classes} classes "
        f"(min={int(counts_arr.min())}, median={int(np.median(counts_arr))}, "
        f"max={int(counts_arr.max())}) "
        + ("[crops cached in RAM]" if getattr(train_ds, "_use_cache", False)
           else "[streaming]"),
    )

    # WeightedRandomSampler with inverse-frequency weights, capped at
    # ``max_samples_per_class`` so a rare class can't dominate via huge
    # weights and a common class can't drown out the batch. Mirrors the
    # Mtb reference's class-balancing strategy.
    max_per_class = max(1, int(getattr(config, "max_samples_per_class", 5000)))
    inv_freq = np.zeros(len(train_int_labels), dtype=np.float64)
    for k, cnt in class_counts.items():
        # Per-cell weight = 1/count, capped so the per-class total
        # doesn't exceed max_per_class units.
        w = 1.0 / cnt
        cap = max_per_class / max(1, cnt)
        per_cell_w = min(w, cap)
        inv_freq[(train_int_labels == k)] = per_cell_w
    inv_freq[~kept_mask] = 0.0
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(inv_freq, dtype=torch.double),
        num_samples=int(kept_mask.sum()),
        replacement=True,
    )

    # Worker count + loader settings: same logic as train_autoencoder.
    cached = bool(getattr(train_ds, "_use_cache", False))
    if cached:
        num_workers = 0
    else:
        import os
        cpu_count = os.cpu_count() or 4
        num_workers = max(2, min(8, cpu_count - 1))
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
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # Build model via the config dispatch — supports both
    # ``supcon_resnet18`` (ImageNet-pretrained ResNet-18 backbone) and
    # ``supcon_lightweight`` (from-scratch 5-block CNN).
    #
    # When fine-tuning, the checkpoint's conv1 input-channel count is
    # the source of truth — peek at it and override the config so a
    # change in include_mask / image_channels between runs doesn't break
    # the load with a "size mismatch for encoder.0.weight" error.
    if fine_tune and config.model_path and config.model_path.exists():
        state = torch.load(
            str(config.model_path), map_location="cpu", weights_only=True,
        )
        ckpt_in = _peek_checkpoint_in_channels(state)
        cfg_in = config.total_in_channels()
        if ckpt_in is not None and ckpt_in != cfg_in:
            progress_cb(
                0.025,
                f"Fine-tune checkpoint expects {ckpt_in}-channel input "
                f"({cfg_in} requested) — overriding config to match. "
                f"Train from scratch (untick 'Use existing model') if you "
                f"want to change the channel count.",
            )
            # Override at the field level the build_model() reads.
            if config.image_channels is not None:
                # When user has picked specific channels, drop the mask
                # to fit (or vice-versa). Cleanest is to fall back to the
                # legacy in_channels path.
                config.image_channels = None
            config.in_channels = max(1, ckpt_in - (1 if config.include_mask else 0))
            if config.total_in_channels() != ckpt_in:
                # If include_mask was off, just set in_channels to ckpt_in.
                config.in_channels = ckpt_in
                config.include_mask = False
        model = config.build_model()
        model.load_state_dict(state, strict=False)
        progress_cb(0.03, f"Loaded model from {config.model_path.name}")
    else:
        model = config.build_model()

    model = model.to(device)
    if not fine_tune and config.freeze_epochs > 0:
        model.freeze_early_layers()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.weight_decay,
    )
    warmup_epochs = max(0, int(getattr(config, "warmup_epochs", 5)))
    scheduler = _build_warmup_cosine(optimizer, warmup_epochs, epochs)
    criterion = SupConLoss(temperature=temperature)
    early_stop_patience = max(0, int(getattr(config, "early_stop_patience", 15)))

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []
    t0 = time.time()

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    train_report_every = max(10, n_train_batches // 4)

    progress_cb(
        0.04,
        f"Starting SupCon training on {device.upper()} — "
        f"{n_train_batches} train batches × {epochs} epochs, "
        f"τ={temperature}, proj_dim={projection_dim}",
    )

    def _labels_for_batch(metas) -> torch.Tensor:
        ys = []
        for m in metas:
            cls = _resolve_class_label(m, source=label_source)
            ys.append(label_map.get(cls, -1))
        return torch.as_tensor(ys, dtype=torch.long, device=device)

    epochs_no_improve = 0
    stopped_early_at: Optional[int] = None
    for epoch in range(epochs):
        if (
            not fine_tune
            and epoch == config.freeze_epochs
        ):
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=config.weight_decay,
            )
            # Re-arm warmup on the freshly-unfrozen network so the bigger
            # parameter set doesn't get hit with a full-magnitude lr step
            # immediately.
            scheduler = _build_warmup_cosine(
                optimizer,
                min(warmup_epochs, max(0, epochs - epoch - 1)),
                epochs - epoch,
            )
            progress_cb(epoch / epochs, f"Epoch {epoch}: unfreezing all layers")

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_t0 = time.time()
        progress_cb(
            (epoch + 0.0) / epochs,
            f"Epoch {epoch + 1}/{epochs}: starting training "
            f"({n_train_batches} batches)...",
        )
        for batch_idx, (crops, metas) in enumerate(train_loader):
            crops = crops.to(device, non_blocking=True)
            labels = _labels_for_batch(metas)

            # Two independent augmented views for the SupCon objective.
            view_a = _augment_batch_gpu(crops, config)
            view_b = _augment_batch_gpu(crops, config)
            stacked = torch.cat([view_a, view_b], dim=0)
            labels_2v = torch.cat([labels, labels], dim=0)

            proj, _ = model(stacked)
            loss = criterion(proj, labels_2v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (
                (batch_idx + 1) % train_report_every == 0
                or batch_idx + 1 == n_train_batches
            ):
                batch_t = time.time() - epoch_t0
                avg_per_batch = batch_t / (batch_idx + 1)
                rem = n_train_batches - (batch_idx + 1)
                eta_epoch = rem * avg_per_batch
                eta_total = eta_epoch + avg_per_batch * n_train_batches * (epochs - epoch - 1)
                overall = (epoch + (batch_idx + 1) / n_train_batches * 0.85) / epochs
                progress_cb(
                    overall,
                    f"Epoch {epoch + 1}/{epochs}: train "
                    f"{batch_idx + 1}/{n_train_batches} "
                    f"loss={loss.item():.4f} "
                    f"({batch_t:.0f}s, ETA epoch {eta_epoch:.0f}s, "
                    f"total {eta_total/60:.1f}min)"
                )

        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        # Validation: same loss on held-out FOVs (with augmentation since
        # the loss requires variation — without augmentation the two views
        # would be identical and the loss collapses).
        model.eval()
        v_loss = 0.0
        n_v = 0
        with torch.no_grad():
            for crops, metas in val_loader:
                crops = crops.to(device, non_blocking=True)
                labels = _labels_for_batch(metas)
                view_a = _augment_batch_gpu(crops, config)
                view_b = _augment_batch_gpu(crops, config)
                stacked = torch.cat([view_a, view_b], dim=0)
                labels_2v = torch.cat([labels, labels], dim=0)
                proj, _ = model(stacked)
                v_loss += criterion(proj, labels_2v).item()
                n_v += 1
        v_loss = v_loss / max(n_v, 1)
        val_losses.append(v_loss)
        scheduler.step()

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            # Save the full model (encoder + head) so fine-tuning can resume;
            # extract.py loads only the encoder portion via .encode().
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        progress_cb(
            0.05 + 0.90 * (epoch + 1) / epochs,
            f"Epoch {epoch + 1}/{epochs} done — "
            f"train: {train_loss:.5f}, val: {v_loss:.5f} "
            f"(epoch {time.time() - epoch_t0:.0f}s, total {elapsed/60:.1f}min)",
        )

        if (
            early_stop_patience > 0
            and epochs_no_improve >= early_stop_patience
        ):
            stopped_early_at = epoch + 1
            progress_cb(
                0.05 + 0.90 * (epoch + 1) / epochs,
                f"Early stopping at epoch {epoch + 1} — "
                f"val loss hasn't improved in {early_stop_patience} epochs "
                f"(best={best_val_loss:.5f}).",
            )
            break

    torch.save(model.state_dict(), output_dir / "final_model.pth")

    cfg_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in config.__dict__.items()
    }
    cfg_dict["device"] = device
    cfg_dict["train_cells"] = len(train_ds)
    cfg_dict["val_cells"] = len(val_ds)
    cfg_dict["n_classes"] = n_classes
    cfg_dict["temperature"] = temperature
    cfg_dict["projection_dim"] = projection_dim
    cfg_dict["loss"] = "supcon"
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Save label map so we can map cells → classes again at inference time.
    with open(output_dir / "supcon_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    # Reuse the autoencoder's training-curve plot.
    from .train import _save_training_curves
    _save_training_curves(train_losses, val_losses, output_dir)

    summary = {
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "epochs_trained": stopped_early_at if stopped_early_at else epochs,
        "stopped_early_at": stopped_early_at,
        "train_cells": len(train_ds),
        "val_cells": len(val_ds),
        "n_classes": n_classes,
        "label_source": label_source,
        "min_cells_per_class": min_cells,
        "model_path": str(output_dir / "best_model.pth"),
        "elapsed_seconds": time.time() - t0,
        "loss": "supcon",
    }

    for ds in (train_ds, val_ds):
        try:
            ds.close()
        except Exception:  # noqa: BLE001
            pass

    return summary
