"""Configuration dataclass for autoencoder training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AutoencoderConfig:
    # Model architecture
    # "resnet18" / "lightweight" — autoencoders, MSE reconstruction loss.
    # "supcon_resnet18" / "supcon_lightweight" — encoder + projection head,
    #     SupCon contrastive loss using gene labels. Sharper class clusters
    #     at the cost of needing labels. The lightweight variant is a much
    #     smaller from-scratch CNN — often the right choice for bacteria,
    #     where ResNet-18's capacity and ImageNet priors don't pay off.
    model_type: str = "resnet18"
    latent_dim: int = 512
    crop_size: int = 128
    in_channels: int = 1  # 1 = brightfield only, 2 = brightfield + fluorescence

    # Training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"  # "cosine" | "step"
    freeze_epochs: int = 10
    perceptual_loss_weight: float = 0.0
    val_fraction: float = 0.1

    # SupCon-specific (ignored when model_type is an autoencoder)
    temperature: float = 0.07           # SupCon contrastive temperature
    projection_dim: int = 128           # SupCon projection head output dim

    # Augmentation
    augment: bool = True
    rotation: bool = True
    flip: bool = True
    brightness_jitter: float = 0.1
    contrast_jitter: float = 0.1
    gaussian_blur_sigma: float = 1.5  # max sigma for binning robustness

    # Fine-tuning (incremental training on new runs)
    fine_tune_epochs: int = 10
    fine_tune_lr: float = 1e-5

    # Paths
    output_dir: Optional[Path] = None
    model_path: Optional[Path] = None  # existing .pth for inference or fine-tuning

    @property
    def is_supcon(self) -> bool:
        return self.model_type.startswith("supcon")

    def build_model(self):
        """Instantiate the configured model architecture."""
        from .models import (
            LightweightAutoencoder,
            ResNet18Autoencoder,
            SupConLightweight,
            SupConResNet18,
        )

        if self.model_type == "resnet18":
            return ResNet18Autoencoder(
                in_channels=self.in_channels, latent_dim=self.latent_dim
            )
        elif self.model_type == "lightweight":
            return LightweightAutoencoder(
                in_channels=self.in_channels, latent_dim=self.latent_dim
            )
        elif self.model_type == "supcon_resnet18":
            return SupConResNet18(
                in_channels=self.in_channels,
                latent_dim=self.latent_dim,
                projection_dim=self.projection_dim,
            )
        elif self.model_type == "supcon_lightweight":
            return SupConLightweight(
                in_channels=self.in_channels,
                latent_dim=self.latent_dim,
                projection_dim=self.projection_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")
