"""
VAYU-NET — INSAT-3D SELF-SUPERVISED MASKED AUTOENCODER
======================================================
Problem Statement: SIH 2026 PS 26070
Module: Self-Supervised Masked Spatial Reconstruction for Satellite Imagery

Architecture:
  - Encoder: Exactly matches CompactSpatialCNN (in_channels -> 16 -> 32 -> 64 -> 64-dim latent).
  - Decoder: Symmetric progressive upsampling decoder (64-dim latent -> [in_channels, 72, 116]).
  - Self-Supervised Objective: Random patch masking (35% area masked) with MSE reconstruction loss.
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.multisource_decoupled import CompactSpatialCNN


class InsatDecoder(nn.Module):
    """
    Symmetric progressive upsampling decoder for reconstructing [C, 72, 116] satellite maps.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 64 * 9 * 15)
        self.conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, 64]
        B = z.shape[0]
        x = F.gelu(self.fc(z)).view(B, 64, 9, 15)
        x = F.interpolate(x, size=(18, 29), mode="bilinear", align_corners=False)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, size=(36, 58), mode="bilinear", align_corners=False)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.interpolate(x, size=(72, 116), mode="bilinear", align_corners=False)
        out = self.conv3(x)
        return out


def generate_random_patch_mask(batch_size: int,
                               height: int = 72,
                               width: int = 116,
                               patch_size: int = 8,
                               mask_ratio: float = 0.35,
                               device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates binary patch masks of shape [B, 1, H, W] where 1 = unmasked, 0 = masked.
    """
    n_h = height // patch_size
    n_w = width // patch_size
    n_patches = n_h * n_w
    n_masked = int(n_patches * mask_ratio)

    mask_patches = torch.ones((batch_size, n_patches), device=device)
    for b in range(batch_size):
        perm = torch.randperm(n_patches, device=device)
        mask_patches[b, perm[:n_masked]] = 0.0

    mask = mask_patches.view(batch_size, 1, n_h, n_w)
    mask = F.interpolate(mask, size=(height, width), mode="nearest")
    return mask


class InsatMaskedAutoencoder(nn.Module):
    """
    Self-supervised spatial reconstruction network for INSAT-3D channels.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 64, mask_ratio: float = 0.35):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.encoder = CompactSpatialCNN(in_channels=in_channels, embed_dim=embed_dim)
        self.decoder = InsatDecoder(in_channels=in_channels, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: [B, C, 72, 116]
            mask: Optional [B, 1, 72, 116] binary mask
        """
        B, C, H, W = x.shape
        if mask is None:
            mask = generate_random_patch_mask(B, H, W, patch_size=8, mask_ratio=self.mask_ratio, device=x.device)

        # Apply mask to input (0 = missing/occluded)
        x_masked = x * mask

        # Encode masked image into latent representation
        z = self.encoder(x_masked)

        # Decode latent into full spatial reconstruction
        x_rec = self.decoder(z)

        # Calculate reconstruction loss (weighted toward masked patches)
        loss_total = F.mse_loss(x_rec, x)
        loss_masked = F.mse_loss(x_rec * (1.0 - mask), x * (1.0 - mask))
        loss = loss_total + 2.0 * loss_masked

        return {
            "loss": loss,
            "loss_total": loss_total,
            "loss_masked": loss_masked,
            "reconstruction": x_rec,
            "latent": z,
            "mask": mask,
            "x_masked": x_masked
        }

    def get_encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns encoder state dict ready for direct injection into CompactSpatialCNN."""
        return {k: v.cpu().clone() for k, v in self.encoder.state_dict().items()}
