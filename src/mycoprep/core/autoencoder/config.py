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
    # Number of non-mask image channels to feed in. Used as a fallback
    # when ``image_channels`` is None (default) — picks the first N
    # non-mask channels from the H5. When ``image_channels`` is provided,
    # this field is ignored.
    in_channels: int = 1
    # Explicit per-channel selection by name. Names matched case-
    # insensitively against the H5 ``channel_names`` attribute. None =
    # take the first ``in_channels`` non-mask channels.
    image_channels: Optional[list[str]] = None
    # Append the segmentation mask as an additional input channel
    # (binarised at 0.5). The Mtb reference SupCon used phase + ParB +
    # mask — the mask gives the encoder the segmentation prior for free
    # and the ResNet conv1 happily adapts to ``len(image_channels) + 1``
    # input channels. Default ON.
    include_mask: bool = True

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
    # Per-image Gaussian noise σ ∈ [0, ``gaussian_noise_sigma``]. Mirrors
    # the Mtb reference's GaussNoise(std=0..0.08, p=0.3) — adds invariance
    # to sensor and shot noise.
    gaussian_noise_sigma: float = 0.08
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

    def total_in_channels(self) -> int:
        """Number of input channels the model will see.

        ``image_channels`` (when provided) defines the imaging channels;
        otherwise the legacy ``in_channels`` count is used as the number
        of non-mask channels. The mask is appended when ``include_mask``
        is True.
        """
        if self.image_channels is not None:
            n_img = len(self.image_channels)
        else:
            n_img = self.in_channels
        return n_img + (1 if self.include_mask else 0)

    def build_model(self):
        """Instantiate the configured model architecture."""
        from .models import (
            LightweightAutoencoder,
            ResNet18Autoencoder,
            SupConLightweight,
            SupConResNet18,
        )

        n_in = self.total_in_channels()
        if self.model_type == "resnet18":
            return ResNet18Autoencoder(
                in_channels=n_in, latent_dim=self.latent_dim
            )
        elif self.model_type == "lightweight":
            return LightweightAutoencoder(
                in_channels=n_in, latent_dim=self.latent_dim
            )
        elif self.model_type == "supcon_resnet18":
            return SupConResNet18(
                in_channels=n_in,
                latent_dim=self.latent_dim,
                projection_dim=self.projection_dim,
            )
        elif self.model_type == "supcon_lightweight":
            return SupConLightweight(
                in_channels=n_in,
                latent_dim=self.latent_dim,
                projection_dim=self.projection_dim,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")
