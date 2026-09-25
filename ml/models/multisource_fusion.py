"""
VAYU-NET — MULTI-SOURCE SATELLITE FUSION MODEL ARCHITECTURE
============================================================
Problem Statement: SIH 2026 PS 26070
Module: Multi-Source Satellite Deep Learning & Modality Ablation

Architectures:
  - Model A (GridSat Only): 6-frame GridSat-B1 sequence (1-channel IRWIN 11um)
  - Model B (INSAT Only): 6-frame INSAT-3D sequence (3-channel TIR1, TIR2, WV)
  - Model C (GridSat + INSAT Fusion): Joint multi-source sequence modeling

Heads:
  1. Center Localization: Current cyclone center [lat, lon]
  2. Intensity Classification: 7-class IMD cyclone intensity category
  3. Wind Speed Regression: Maximum sustained surface wind in knots
  4. Track Prediction: Multi-horizon displacements (+12h, +24h, +48h)
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
    "D": 0,      # Depression (17-27 kt)
    "DD": 1,     # Deep Depression (28-33 kt)
    "CS": 2,     # Cyclonic Storm (34-47 kt)
    "SCS": 3,    # Severe Cyclonic Storm (48-63 kt)
    "VSCS": 4,   # Very Severe Cyclonic Storm (64-89 kt)
    "ESCS": 5,   # Extremely Severe Cyclonic Storm (90-119 kt)
    "SuCS": 6    # Super Cyclonic Storm (>= 120 kt)
}
IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}

# TRAIN-only default wind normalization statistics
DEFAULT_TRAIN_WIND_MEAN_KT = 36.88362
DEFAULT_TRAIN_WIND_STD_KT = 17.97539


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two coordinates in kilometers."""
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
    Lightweight 3-stage CNN spatial encoder for small-sample regime (175 train samples).
    Reduces 2D satellite maps to compact spatial embedding vector.
    """
    def __init__(self, in_channels: int, embed_dim: int = 64):
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


class BranchEncoder(nn.Module):
    """
    End-to-end spatiotemporal encoder for a 6-frame satellite modality sequence.
    Applies shared SpatialCNN across all 6 timesteps, followed by temporal GRU.
    """
    def __init__(self, in_channels: int, spatial_dim: int = 64, gru_dim: int = 64):
        super().__init__()
        self.spatial = CompactSpatialCNN(in_channels, spatial_dim)
        self.gru = nn.GRU(spatial_dim, gru_dim, num_layers=1, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, 6, C, H, W]
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        feats = self.spatial(x_flat).view(B, T, -1)  # [B, 6, spatial_dim]
        out, _ = self.gru(feats)                     # [B, 6, gru_dim]
        return out[:, -1, :]                         # Final temporal state at t0 [B, gru_dim]


class MultisourceFusionModel(nn.Module):
    """
    Modular Multi-Source Satellite Architecture supporting:
      - mode='gridsat': Model A (GridSat only, 1-channel sequence)
      - mode='insat': Model B (INSAT-3D only, 3-channel sequence TIR1, TIR2, WV)
      - mode='fusion': Model C (GridSat + INSAT-3D Joint Multi-Source Fusion)
    """
    def __init__(self,
                 mode: str = "fusion",
                 spatial_dim: int = 64,
                 gru_dim: int = 64,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 wind_mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                 wind_std: float = DEFAULT_TRAIN_WIND_STD_KT):
        super().__init__()
        if mode not in ["gridsat", "insat", "fusion"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'gridsat', 'insat', or 'fusion'.")

        self.mode = mode
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.latent_dim = latent_dim

        # Modality Encoders
        if mode in ["gridsat", "fusion"]:
            self.gridsat_branch = BranchEncoder(in_channels=1, spatial_dim=spatial_dim, gru_dim=gru_dim)
        else:
            self.gridsat_branch = None

        if mode in ["insat", "fusion"]:
            self.insat_branch = BranchEncoder(in_channels=3, spatial_dim=spatial_dim, gru_dim=gru_dim)
        else:
            self.insat_branch = None

        # Fusion Projection
        proj_in = (gru_dim * 2) if mode == "fusion" else gru_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, latent_dim),
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

        # 4. Track Prediction Heads -> [u_lat, u_lon] in [0, 1] for +12h, +24h, +48h
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
        """Returns total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": frozen
        }

    def decode_normalized_coordinates(self, norm_coords: torch.Tensor) -> torch.Tensor:
        """
        Converts normalized coordinates [u_lat, u_lon] in [0, 1] to geographic [lat, lon] in degrees.
        """
        lat = LAT_MIN + norm_coords[..., 0] * LAT_SPAN
        lon = LON_MIN + norm_coords[..., 1] * LON_SPAN
        return torch.stack([lat, lon], dim=-1)

    def unnormalize_wind(self, norm_wind: torch.Tensor) -> torch.Tensor:
        """Converts normalized wind speed to knots."""
        return norm_wind * self.wind_std + self.wind_mean

    def forward(self,
                gridsat_seq: Optional[torch.Tensor] = None,
                insat_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            gridsat_seq: Optional [B, 6, 1, H, W]
            insat_seq: Optional [B, 6, 3, H, W]
        Returns:
            Dictionary containing model predictions for all 4 tasks.
        """
        if self.mode == "gridsat":
            if gridsat_seq is None:
                raise ValueError("gridsat_seq must be provided when mode='gridsat'")
            lat = self.gridsat_branch(gridsat_seq)
        elif self.mode == "insat":
            if insat_seq is None:
                raise ValueError("insat_seq must be provided when mode='insat'")
            lat = self.insat_branch(insat_seq)
        else: # fusion
            if gridsat_seq is None or insat_seq is None:
                raise ValueError("Both gridsat_seq and insat_seq must be provided when mode='fusion'")
            g_lat = self.gridsat_branch(gridsat_seq)
            i_lat = self.insat_branch(insat_seq)
            lat = torch.cat([g_lat, i_lat], dim=-1)

        shared = self.proj(lat)

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
