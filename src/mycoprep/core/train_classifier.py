#!/usr/bin/env python3
"""
Train Cell Quality Classifier
==============================
Trains the CNN on labeled cell crops produced by label_cells.py.

Features:
    - Automatic train/validation split (stratified by class)
    - Heavy data augmentation for small datasets
    - Class-weighted loss to handle imbalanced labels
    - Early stopping with best-model checkpointing
    - Training curves and confusion matrix saved as PNGs
    - Per-class precision, recall, F1 reported

Requirements:
    pip install torch torchvision numpy scikit-learn matplotlib

Usage:
    python train_classifier.py --data labeled_data/ --output models/

    # With custom hyperparameters
    python train_classifier.py --data labeled_data/ --output models/ \\
        --epochs 100 --lr 1e-3 --batch-size 32 --val-fraction 0.2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def make_augmentation_transforms():
    """
    Build data augmentation pipeline using torchvision transforms.

    Augmentations suitable for microscopy:
        - Random horizontal + vertical flips
        - Random 90° rotations
        - Small random affine (translation, scale)
        - Gaussian noise
        - Random brightness/contrast adjustment
    """
    import torch
    import torchvision.transforms.v2 as T

    class PerChannelNormalize:
        """Normalize each channel of a tensor to [0, 1] independently."""
        def __call__(self, x):
            for c in range(x.shape[0]):
                cmin, cmax = x[c].min(), x[c].max()
                if cmax > cmin:
                    x[c] = (x[c] - cmin) / (cmax - cmin)
            return x

    class AddGaussianNoise:
        """Add Gaussian noise with random std."""
        def __init__(self, max_std=0.05):
            self.max_std = max_std

        def __call__(self, x):
            import torch as t
            std = t.rand(1).item() * self.max_std
            noise = t.randn_like(x) * std
            return (x + noise).clamp(0, 1)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5),
        T.RandomApply([
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.3),
        PerChannelNormalize(),
        AddGaussianNoise(max_std=0.03),
    ])

    val_transform = T.Compose([
        PerChannelNormalize(),
    ])

    return train_transform, val_transform


class CellCropDataset:
    """
    PyTorch-compatible dataset for cell crops.

    Wraps numpy arrays with optional transforms.
    """

    def __init__(self, crops, labels, transform=None):
        """
        Args:
            crops:     (N, C, H, W) float32 numpy array
            labels:    (N,) int64 numpy array
            transform: callable or None
        """
        import torch

        self.crops = torch.from_numpy(crops)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        crop = self.crops[idx].clone()
        label = self.labels[idx]

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label


def compute_class_weights(labels, num_classes):
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Returns:
        weights: (num_classes,) float32 tensor
    """
    import torch

    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return torch.from_numpy(weights)


def train(
    data_dir,
    output_dir,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    val_fraction=0.2,
    patience=15,
    device=None,
    phase_only=True,
    phase_channel_in_crop=0,
    pretrained_path=None,
    progress_cb=None,
):
    """
    Train the cell quality CNN.

    Args:
        data_dir:       path to labeled_data/ (with crops/ and labels.csv)
        output_dir:     path to save model and training artifacts
        epochs:         maximum training epochs
        batch_size:     training batch size
        lr:             initial learning rate
        val_fraction:   fraction of data for validation
        patience:       early stopping patience (epochs without improvement)
        device:         torch device (None = auto)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    from .cell_quality_classifier import (
        CLASS_NAMES,
        NUM_CLASSES,
        load_crop_dataset,
        _build_model,
    )

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        print(f"Device: CUDA GPU ({gpu_name})")
    elif device == "mps":
        print(f"Device: Apple Silicon GPU (MPS)")
    else:
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_cuda or has_mps:
            print(f"Device: CPU  (GPU available — pass --gpu to use it)")
        else:
            print(f"Device: CPU  (no GPU detected)")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    crops, labels, meta = load_crop_dataset(
        data_dir, phase_only=phase_only,
        phase_channel_in_crop=phase_channel_in_crop,
    )
    n_samples = len(labels)
    in_channels = crops.shape[1]

    print(f"  {n_samples} labeled cells, {in_channels} channels per crop")
    for i, name in enumerate(CLASS_NAMES):
        count = (labels == i).sum()
        print(f"    {name}: {count}")

    if n_samples < 20:
        print("WARNING: Very few labeled cells. Consider labeling more for better results.")

    # ── Train/val split (stratified) ──────────────────────────────────────
    train_idx, val_idx = train_test_split(
        np.arange(n_samples),
        test_size=val_fraction,
        stratify=labels,
        random_state=42,
    )

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

    train_transform, val_transform = make_augmentation_transforms()

    train_ds = CellCropDataset(crops[train_idx], labels[train_idx], train_transform)
    val_ds = CellCropDataset(crops[val_idx], labels[val_idx], val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # ── Model, loss, optimizer ────────────────────────────────────────────
    model = _build_model(in_channels, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} trainable parameters")

    if pretrained_path is not None:
        try:
            state = torch.load(str(pretrained_path), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"  Loaded pretrained weights from {pretrained_path}")
            if missing:    print(f"    missing keys:    {missing}")
            if unexpected: print(f"    unexpected keys: {unexpected}")
        except Exception as e:
            print(f"  WARNING: could not load pretrained weights ({e}); training from scratch")

    class_weights = compute_class_weights(labels, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_crops, batch_labels in train_loader:
            batch_crops = batch_crops.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_crops)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_labels)
            train_correct += (logits.argmax(1) == batch_labels).sum().item()
            train_total += len(batch_labels)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_crops, batch_labels in val_loader:
                batch_crops = batch_crops.to(device)
                batch_labels = batch_labels.to(device)

                logits = model(batch_crops)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item() * len(batch_labels)
                preds = logits.argmax(1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Logging ───────────────────────────────────────────────────
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"LR={lr_now:.2e}"
        )
        if progress_cb is not None:
            progress_cb(
                (epoch + 1) / epochs,
                f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}",
            )

        # ── Checkpointing ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            if progress_cb is not None:
                progress_cb(
                    1.0,
                    f"Early stopped at epoch {epoch + 1} (no improvement for {patience} epochs). "
                    f"Best val acc {best_val_acc:.3f}.",
                )
            break
    else:
        # Loop completed all epochs without early-stopping.
        if progress_cb is not None:
            progress_cb(
                1.0, f"Done · best val acc {best_val_acc:.3f} at epoch {best_epoch}",
            )

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Load best model for final metrics
    model.load_state_dict(
        torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    all_preds = []
    all_true = []
    all_probs = []
    with torch.no_grad():
        for batch_crops, batch_labels in val_loader:
            batch_crops = batch_crops.to(device)
            logits = model(batch_crops)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_true.extend(batch_labels.numpy())
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    all_probs = np.concatenate(all_probs, axis=0)  # (N, num_classes)

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix

    present_classes = sorted(set(all_true) | set(all_preds))
    target_names = [CLASS_NAMES[i] for i in present_classes]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (validation set)")
    print("=" * 60)
    print(classification_report(all_true, all_preds,
                                labels=present_classes,
                                target_names=target_names,
                                zero_division=0))

    # ── Save training artifacts ───────────────────────────────────────────
    # Save final model too
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Save training config
    config = {
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "best_val_acc": float(best_val_acc),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ── Plot training curves ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["train_loss"], label="Train")
        ax1.plot(history["val_loss"], label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history["train_acc"], label="Train")
        ax2.plot(history["val_acc"], label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=150)
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(all_true, all_preds, labels=present_classes)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha="right")
        ax.set_yticklabels(target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Validation)")

        # Annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

        # ── ROC curve ─────────────────────────────────────────────────
        try:
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

            fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

            if NUM_CLASSES == 2:
                # Binary: use P(bad) as the score for the positive class
                scores = all_probs[:, 1]

                # ROC
                fpr, tpr, roc_thresholds = roc_curve(all_true, scores, pos_label=1)
                roc_auc = auc(fpr, tpr)

                ax_roc.plot(fpr, tpr, color="#2166ac", lw=2,
                            label=f"AUC = {roc_auc:.3f}")
                ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

                # Mark the operating point closest to default threshold (0.5)
                default_idx = np.argmin(np.abs(roc_thresholds - 0.5))
                ax_roc.plot(fpr[default_idx], tpr[default_idx], "ro", ms=8,
                            label=f"Threshold=0.5 (FPR={fpr[default_idx]:.2f}, "
                                  f"TPR={tpr[default_idx]:.2f})")

                # Youden's J statistic — optimal threshold
                j_scores = tpr - fpr
                best_j_idx = np.argmax(j_scores)
                best_thresh = roc_thresholds[best_j_idx]
                ax_roc.plot(fpr[best_j_idx], tpr[best_j_idx], "g^", ms=10,
                            label=f"Optimal threshold={best_thresh:.2f} "
                                  f"(FPR={fpr[best_j_idx]:.2f}, TPR={tpr[best_j_idx]:.2f})")

                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve (good vs bad)")
                ax_roc.legend(loc="lower right", fontsize=8)
                ax_roc.grid(True, alpha=0.3)

                # Precision-Recall
                precision, recall, pr_thresholds = precision_recall_curve(
                    all_true, scores, pos_label=1)
                ap = average_precision_score(all_true, scores, pos_label=1)

                ax_pr.plot(recall, precision, color="#b2182b", lw=2,
                           label=f"AP = {ap:.3f}")

                # Baseline: fraction of positives
                baseline = (all_true == 1).mean()
                ax_pr.axhline(baseline, color="k", ls="--", lw=1, alpha=0.5,
                              label=f"Baseline ({baseline:.2f})")

                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall Curve")
                ax_pr.legend(loc="lower left", fontsize=8)
                ax_pr.grid(True, alpha=0.3)

                print(f"\n  ROC AUC: {roc_auc:.3f}")
                print(f"  Average Precision: {ap:.3f}")
                print(f"  Optimal threshold (Youden's J): {best_thresh:.3f}")

            else:
                # Multi-class: one-vs-rest ROC for each class
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(all_true, classes=list(range(NUM_CLASSES)))
                colors = plt.cm.Set1(np.linspace(0, 1, NUM_CLASSES))

                for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
                    if i >= all_probs.shape[1]:
                        continue
                    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                    auc_i = auc(fpr_i, tpr_i)
                    ax_roc.plot(fpr_i, tpr_i, color=color, lw=2,
                                label=f"{name} (AUC={auc_i:.3f})")

                    prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], all_probs[:, i])
                    ap_i = average_precision_score(y_bin[:, i], all_probs[:, i])
                    ax_pr.plot(rec_i, prec_i, color=color, lw=2,
                               label=f"{name} (AP={ap_i:.3f})")

                ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curves (one-vs-rest)")
                ax_roc.legend(fontsize=8)
                ax_roc.grid(True, alpha=0.3)

                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.set_title("Precision-Recall Curves")
                ax_pr.legend(fontsize=8)
                ax_pr.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "roc_pr_curves.png", dpi=150)
            plt.close()

            # ── Raw curves → JSON for interactive GUI overlays ────────────
            try:
                import json as _json
                curves: dict = {"num_classes": int(NUM_CLASSES),
                                "class_names": list(CLASS_NAMES)}
                if NUM_CLASSES == 2:
                    curves["binary"] = {
                        "roc":  {"fpr": fpr.tolist(),
                                 "tpr": tpr.tolist(),
                                 "thresholds": roc_thresholds.tolist(),
                                 "auc": float(roc_auc)},
                        "pr":   {"precision": precision.tolist(),
                                 "recall":    recall.tolist(),
                                 "thresholds": pr_thresholds.tolist(),
                                 "ap": float(ap),
                                 "baseline": float(baseline)},
                        "optimal_threshold": float(best_thresh),
                    }
                (output_dir / "metrics.json").write_text(_json.dumps(curves, indent=2))
                print(f"Raw ROC/PR curves: {output_dir / 'metrics.json'}")
            except Exception as _e:  # noqa: BLE001
                print(f"  (could not write raw metrics.json: {_e})")
            print(f"ROC & PR curves: {output_dir / 'roc_pr_curves.png'}")

        except ImportError:
            print("(sklearn ROC/PR metrics not available — skipped ROC plot)")

        print(f"\nTraining curves: {output_dir / 'training_curves.png'}")
        print(f"Confusion matrix: {output_dir / 'confusion_matrix.png'}")

    except ImportError:
        print("(matplotlib not available — skipped plot generation)")

    print(f"\nModel saved: {output_dir / 'best_model.pth'}")
    print(f"Config saved: {output_dir / 'training_config.json'}")
    print("\nTo use in the pipeline:")
    print(f"  python cellpose_pipeline.py -i <input> -o <output> "
          f"--classify {output_dir / 'best_model.pth'}")

    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "model_path": str(output_dir / "best_model.pth"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train the cell quality CNN classifier."
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to labeled data directory (from label_cells.py)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to save trained model and artifacts"
    )
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation split")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--phase-only", action="store_true", default=True,
                        help="Train on phase + mask channels only (2-ch input). "
                             "Makes the model generalisable to images without "
                             "fluorescence. Enabled by default.")
    parser.add_argument("--all-channels", action="store_true",
                        help="Train on all imaging channels + mask (overrides --phase-only). "
                             "The resulting model will only work on images with the same "
                             "number of channels it was trained on.")
    parser.add_argument("--phase-channel-in-crop", type=int, default=0,
                        help="Index of the phase channel within the saved crop arrays "
                             "(default: 0, i.e. the first imaging channel). The mask is "
                             "always the last channel.")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training. Selects CUDA on Linux/Windows or "
                             "MPS (Metal) on Apple Silicon automatically.")
    parser.add_argument("--device", type=str, default=None,
                        help="Explicit torch device, e.g. 'cuda', 'mps', 'cpu', 'cuda:1'. "
                             "Overrides --gpu if both are provided.")

    args = parser.parse_args()

    # --gpu picks the best available GPU (CUDA → MPS → warn)
    device = args.device
    if device is None and args.gpu:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            print("Warning: --gpu requested but no GPU found (CUDA or MPS). Falling back to CPU.")
            device = "cpu"

    phase_only = args.phase_only and not args.all_channels

    train(
        args.data, args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        patience=args.patience,
        device=device,
        phase_only=phase_only,
        phase_channel_in_crop=args.phase_channel_in_crop,
    )


if __name__ == "__main__":
    main()
