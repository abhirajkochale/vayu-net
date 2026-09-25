"""
VAYU-NET: EXP-M1 Multi-Task Architecture.
Configurations:
  - M1A (GridSat-only): 1-channel IR sequence [B, 6, 1, 72, 116]
  - M1B (IMERG-only):   1-channel IMERG precipitation sequence [B, 6, 1, 72, 116]
  - M1C (Multimodal):   Separate spatial encoders for GridSat and IMERG, fused before temporal GRU

Prediction Heads:
  1. Center localization: [lat, lon] at t0 (direct coordinate regression in [0, 1])
  2. Intensity classification: 7-class IMD cyclonic category logits
  3. Wind speed regression: Maximum sustained wind in knots
  4. Track prediction: Multi-horizon displacements (+12h, +24h, +48h)
"""

import math
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical NIO Spatial Bounds
LAT_MIN = -5.0
LAT_SPAN = 40.0   # lat in [-5.0, 35.0]
LON_MIN = 40.0
LON_SPAN = 65.0   # lon in [40.0, 105.0]

# Canonical 7-class IMD Category Mapping
CATEGORY_TO_IDX = {
    "D": 0, "DD": 1, "CS": 2, "SCS": 3, "VSCS": 4, "ESCS": 5, "SuCS": 6
}
IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}

# Default TRAIN wind normalization statistics
DEFAULT_TRAIN_WIND_MEAN_KT = 36.88362
DEFAULT_TRAIN_WIND_STD_KT = 17.97539


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(max(0.0, min(1.0, a))), math.sqrt(max(0.0, min(1.0, 1.0 - a))))
    return R * c


class CompactSpatialCNN(nn.Module):
    """
    3-stage CNN spatial encoder mapping [B, C, 72, 116] to spatial embedding [B, embed_dim].
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(64 * 2 * 2, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        flat = torch.flatten(x, 1)
        return F.gelu(self.ln(self.fc(flat)))


class ExpM1Model(nn.Module):
    """
    Controlled Ablation Model for EXP-M1:
      - mode='m1a_gridsat': GridSat-only
      - mode='m1b_imerg':   IMERG-only
      - mode='m1c_fusion':  GridSat + IMERG
    """
    def __init__(self,
                 mode: str = "m1c_fusion",
                 spatial_dim: int = 64,
                 gru_dim: int = 64,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 wind_mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                 wind_std: float = DEFAULT_TRAIN_WIND_STD_KT):
        super().__init__()
        if mode not in ["m1a_gridsat", "m1b_imerg", "m1c_fusion"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'm1a_gridsat', 'm1b_imerg', or 'm1c_fusion'.")

        self.mode = mode
        self.spatial_dim = spatial_dim
        self.gru_dim = gru_dim
        self.latent_dim = latent_dim
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        # Modality Spatial Encoders
        if mode in ["m1a_gridsat", "m1c_fusion"]:
            self.gridsat_spatial = CompactSpatialCNN(in_channels=1, embed_dim=spatial_dim)
        else:
            self.gridsat_spatial = None

        if mode in ["m1b_imerg", "m1c_fusion"]:
            self.imerg_spatial = CompactSpatialCNN(in_channels=1, embed_dim=spatial_dim)
        else:
            self.imerg_spatial = None

        # Modality Fusion Layer before GRU (for M1C)
        if mode == "m1c_fusion":
            self.spatial_fusion = nn.Sequential(
                nn.Linear(spatial_dim * 2, spatial_dim),
                nn.LayerNorm(spatial_dim),
                nn.GELU()
            )
        else:
            self.spatial_fusion = None

        # Temporal GRU: always processes sequence [B, 6, spatial_dim]
        self.temporal_gru = nn.GRU(spatial_dim, gru_dim, num_layers=1, batch_first=True)

        # Projection from temporal latent to multi-task shared representation
        self.proj = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-Task Prediction Heads
        # 1. Center Localization Head -> [u_lat, u_lon] in [0, 1]
        self.head_center = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

        # 2. Intensity Classification Head -> 7 category logits
        self.head_class = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 7)
        )

        # 3. Wind Speed Regression Head -> Normalized continuous wind
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # 4. Multi-Horizon Track Prediction Heads -> [u_lat, u_lon] in [0, 1] for +12h, +24h, +48h
        self.head_track_12h = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        self.head_track_24h = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        self.head_track_48h = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def get_parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }

    def decode_normalized_coordinates(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """Converts normalized coordinates [u_lat, u_lon] in [0, 1] to geographic [lat, lon] in degrees."""
        lat = LAT_MIN + norm_coords[..., 0] * LAT_SPAN
        lon = LON_MIN + norm_coords[..., 1] * LON_SPAN
        return torch.stack([lat, lon], dim=-1)

    def unnormalize_wind(self, norm_wind: torch.Tensor) -> torch.Tensor:
        """Converts normalized wind speed to knots."""
        return norm_wind * self.wind_std + self.wind_mean

    def forward(self,
                gridsat_seq: Optional[torch.Tensor] = None,
                imerg_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            gridsat_seq: Optional [B, 6, 72, 116] or [B, 6, 1, 72, 116]
            imerg_seq:   Optional [B, 6, 72, 116] or [B, 6, 1, 72, 116]
        """
        # Ensure 5D tensor: [B, 6, 1, 72, 116]
        if gridsat_seq is not None and gridsat_seq.ndim == 4:
            gridsat_seq = gridsat_seq.unsqueeze(2)
        if imerg_seq is not None and imerg_seq.ndim == 4:
            imerg_seq = imerg_seq.unsqueeze(2)

        if self.mode == "m1a_gridsat":
            if gridsat_seq is None:
                raise ValueError("gridsat_seq must be provided when mode='m1a_gridsat'")
            B, T, C, H, W = gridsat_seq.shape
            x_flat = gridsat_seq.view(B * T, C, H, W)
            spatial_feats = self.gridsat_spatial(x_flat).view(B, T, self.spatial_dim)
            temporal_input = spatial_feats

        elif self.mode == "m1b_imerg":
            if imerg_seq is None:
                raise ValueError("imerg_seq must be provided when mode='m1b_imerg'")
            B, T, C, H, W = imerg_seq.shape
            x_flat = imerg_seq.view(B * T, C, H, W)
            spatial_feats = self.imerg_spatial(x_flat).view(B, T, self.spatial_dim)
            temporal_input = spatial_feats

        else: # m1c_fusion
            if gridsat_seq is None or imerg_seq is None:
                raise ValueError("Both gridsat_seq and imerg_seq must be provided when mode='m1c_fusion'")
            B, T, C_g, H, W = gridsat_seq.shape
            _, _, C_i, _, _ = imerg_seq.shape
            
            g_flat = gridsat_seq.view(B * T, C_g, H, W)
            i_flat = imerg_seq.view(B * T, C_i, H, W)
            
            g_feats = self.gridsat_spatial(g_flat).view(B, T, self.spatial_dim)
            i_feats = self.imerg_spatial(i_flat).view(B, T, self.spatial_dim)
            
            # Fuse spatial features before temporal GRU
            fused_spatial = torch.cat([g_feats, i_feats], dim=-1)  # [B, 6, 128]
            temporal_input = self.spatial_fusion(fused_spatial)    # [B, 6, 64]

        # Temporal modeling over 6 steps
        gru_out, _ = self.temporal_gru(temporal_input)  # [B, 6, gru_dim]
        t0_latent = gru_out[:, -1, :]                   # Final state at t0 [B, gru_dim]

        shared = self.proj(t0_latent)                   # [B, latent_dim]

        # Multi-task heads
        norm_center = self.head_center(shared)
        class_logits = self.head_class(shared)
        norm_wind = self.head_wind(shared)
        norm_t12 = self.head_track_12h(shared)
        norm_t24 = self.head_track_24h(shared)
        norm_t48 = self.head_track_48h(shared)

        return {
            "center_norm": norm_center,
            "center_deg": self.decode_normalized_coordinates(norm_center),
            "class_logits": class_logits,
            "wind_norm": norm_wind,
            "wind_kt": self.unnormalize_wind(norm_wind),
            "track_12h_norm": norm_t12,
            "track_12h_deg": self.decode_normalized_coordinates(norm_t12),
            "track_24h_norm": norm_t24,
            "track_24h_deg": self.decode_normalized_coordinates(norm_t24),
            "track_48h_norm": norm_t48,
            "track_48h_deg": self.decode_normalized_coordinates(norm_t48),
            "shared_latent": shared
        }