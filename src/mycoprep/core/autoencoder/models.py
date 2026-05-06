"""Encoder architectures for cell morphology embedding.

Two paradigms supported:

- **Autoencoder** — encoder + decoder, MSE reconstruction loss. Self-supervised,
  label-free, captures the full morphological manifold but tends to produce
  smooth embeddings that aren't strongly class-discriminative.
- **SupCon** — encoder + projection head, supervised contrastive loss
  (Khosla et al. 2020). Needs class labels (gene knockdown identity). Pulls
  same-class together, pushes different-class apart. Sharper class-level
  clusters than the autoencoder when labels are available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Autoencoder(nn.Module):
    """ResNet-18 encoder with transposed-conv decoder.

    The encoder is pretrained on ImageNet with the first conv adapted to
    ``in_channels``.  The 512-d avgpool output is the latent vector.
    """

    def __init__(self, in_channels: int = 1, latent_dim: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv from 3 → in_channels
        old_conv = encoder.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
            elif in_channels == 2:
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
                new_conv.weight[:, 1] = old_conv.weight[:, 1]
            else:
                for c in range(in_channels):
                    new_conv.weight[:, c] = old_conv.weight[:, c % 3]
        encoder.conv1 = new_conv

        # Remove classification head — keep everything up to avgpool
        self.encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
            encoder.avgpool,
        )

        # Projection if latent_dim != 512
        if latent_dim != 512:
            self.proj = nn.Linear(512, latent_dim)
        else:
            self.proj = nn.Identity()

        # Decoder: latent → (in_channels, 128, 128)
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 4→8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 8→16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16→32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32→64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # 64→128
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent vector. Returns (B, latent_dim)."""
        h = self.encoder(x)  # (B, 512, 1, 1)
        h = h.flatten(1)  # (B, 512)
        return self.proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction. Returns (B, C, 128, 128)."""
        h = self.decoder_fc(z)  # (B, 512*4*4)
        h = h.view(-1, 512, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_vector)."""
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def freeze_early_layers(self) -> None:
        """Freeze encoder blocks 1-2 for initial training."""
        for name, param in self.encoder.named_parameters():
            # Indices 4,5 in the Sequential are layer1, layer2
            # Also freeze conv1/bn1 (indices 0,1)
            if any(f"{i}." in name or name.startswith(f"{i}.") for i in range(6)):
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


class LightweightAutoencoder(nn.Module):
    """Simple 5-layer convolutional autoencoder as a comparison baseline."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder: 128→64→32→16→8→4 spatial
        self.encoder_conv = nn.Sequential(
            self._enc_block(in_channels, 32),
            self._enc_block(32, 64),
            self._enc_block(64, 128),
            self._enc_block(128, 256),
            self._enc_block(256, 512),
        )
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        if latent_dim != 512:
            self.proj = nn.Linear(512, latent_dim)
        else:
            self.proj = nn.Identity()

        # Decoder: latent → (in_channels, 128, 128)
        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            self._dec_block(512, 256),   # 4→8
            self._dec_block(256, 128),   # 8→16
            self._dec_block(128, 64),    # 16→32
            self._dec_block(64, 32),     # 32→64
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),  # 64→128
            nn.Sigmoid(),
        )

    @staticmethod
    def _enc_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = self.encoder_pool(h).flatten(1)
        return self.proj(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(-1, 512, 4, 4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class SupConResNet18(nn.Module):
    """ResNet-18 encoder + small projection head for supervised contrastive
    learning.

    The encoder is identical to :class:`ResNet18Autoencoder`'s — pretrained
    on ImageNet, first conv adapted to ``in_channels``, output is the 512-d
    avgpool feature. The projection head is a 2-layer MLP that maps 512 → a
    smaller normalised space used for the SupCon loss only. After training
    the head is discarded; downstream embedding/inference uses the 512-d
    encoder features (the `.encode()` method) directly.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Adapt first conv from 3 → in_channels (same logic as autoencoder).
        old_conv = encoder.conv1
        new_conv = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
            elif in_channels == 2:
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
                new_conv.weight[:, 1] = old_conv.weight[:, 1]
            else:
                for c in range(in_channels):
                    new_conv.weight[:, c] = old_conv.weight[:, c % 3]
        encoder.conv1 = new_conv

        self.encoder = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
            encoder.avgpool,
        )

        if latent_dim != 512:
            self.proj = nn.Linear(512, latent_dim)
        else:
            self.proj = nn.Identity()

        # Projection head — standard SupCon practice is 2-layer MLP with
        # hidden dim = encoder dim and BN+ReLU between, output L2-normalised
        # in the loss (not in the head itself).
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d encoder features. Use this for downstream
        inference / strain profiles after training."""
        h = self.encoder(x)
        h = h.flatten(1)
        return self.proj(h)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoder features to the contrastive space and L2-normalise.
        Used during training only."""
        p = self.projection_head(z)
        return F.normalize(p, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (projection, encoder_features). Projection is L2-normalised."""
        z = self.encode(x)
        p = self.project(z)
        return p, z

    def freeze_early_layers(self) -> None:
        """Freeze encoder blocks 1-2 for initial training."""
        for name, param in self.encoder.named_parameters():
            if any(f"{i}." in name or name.startswith(f"{i}.") for i in range(6)):
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        for param in self.parameters():
            param.requires_grad = True


class SupConLightweight(nn.Module):
    """Lightweight (5-block strided-conv) encoder + projection head for SupCon.

    Mirrors :class:`LightweightAutoencoder`'s encoder exactly — five
    Conv2d(k=4, stride=2) blocks with BN + LeakyReLU, ending in
    AdaptiveAvgPool2d → 512-d. The projection head is identical to
    :class:`SupConResNet18`'s.

    Smaller param budget than ResNet-18 (~few hundred K params vs 11M),
    no ImageNet pretraining. Often the right choice for bacteria where
    ResNet's capacity and natural-image priors don't translate well.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 512,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

        self.encoder_conv = nn.Sequential(
            LightweightAutoencoder._enc_block(in_channels, 32),
            LightweightAutoencoder._enc_block(32, 64),
            LightweightAutoencoder._enc_block(64, 128),
            LightweightAutoencoder._enc_block(128, 256),
            LightweightAutoencoder._enc_block(256, 512),
        )
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        if latent_dim != 512:
            self.proj = nn.Linear(512, latent_dim)
        else:
            self.proj = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 512-d encoder features (post-pool, pre-projection-head)."""
        h = self.encoder_conv(x)
        h = self.encoder_pool(h).flatten(1)
        return self.proj(h)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        p = self.projection_head(z)
        return F.normalize(p, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (projection, encoder_features). Projection is L2-normalised."""
        z = self.encode(x)
        p = self.project(z)
        return p, z

    def freeze_early_layers(self) -> None:
        """Freeze the first two conv blocks (out of five) for initial training."""
        for i, block in enumerate(self.encoder_conv):
            if i < 2:
                for p in block.parameters():
                    p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
