"""
VAYU-NET — MULTI-SOURCE SATELLITE TRANSFER LEARNING ARCHITECTURE
=================================================================
Problem Statement: SIH 2026 PS 26070
Module: Pretrained GridSat Representation Transfer + Lightweight INSAT Fusion

Architectures:
  - EXP-1: Frozen pretrained GridSat temporal encoder + trainable multi-task heads
  - EXP-2: Frozen pretrained GridSat + lightweight INSAT-3D branch + fusion projection
  - EXP-3: Partially unfrozen GridSat last-stage encoder + lightweight INSAT-3D branch + fusion
  - Channel Ablation: TIR1 only, TIR1+TIR2, TIR1+TIR2+WV
"""

import math
from typing import Dict, Any, Optional, Tuple, List
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
    Lightweight 3-stage CNN spatial encoder for small-sample regime.
    Reduces 2D satellite maps to compact spatial embedding vector.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(64 * 4, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        flat = torch.flatten(x, 1)
        return F.gelu(self.ln(self.fc(flat)))


class LightweightInsatBranch(nn.Module):
    """
    End-to-end spatiotemporal encoder for 6-frame INSAT-3D sequence.
    """
    def __init__(self, in_channels: int = 3, spatial_dim: int = 64, gru_dim: int = 64):
        super().__init__()
        self.spatial = CompactSpatialCNN(in_channels=in_channels, embed_dim=spatial_dim)
        self.gru = nn.GRU(spatial_dim, gru_dim, num_layers=1, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, 6, C, H, W]
        B, T, C, H, W = x_seq.shape
        x_flat = x_seq.view(B * T, C, H, W)
        feats = self.spatial(x_flat).view(B, T, -1)
        out, _ = self.gru(feats)
        return out[:, -1, :]  # [B, gru_dim]


class MultisourceTransferFusionModel(nn.Module):
    """
    Transfer Learning Multi-Source Architecture.
    Reuses pretrained GridSat temporal representation and fuses lightweight INSAT branch.
    """
    def __init__(self,
                 mode: str = "exp2_fusion_frozen",
                 in_channels_insat: int = 3,
                 pretrained_ckpt_path: str = "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
                 latent_dim: int = 128,
                 dropout: float = 0.1,
                 wind_mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                 wind_std: float = DEFAULT_TRAIN_WIND_STD_KT):
        super().__init__()
        self.mode = mode
        self.wind_mean = wind_mean
        self.wind_std = wind_std
        self.latent_dim = latent_dim

        # 1. Pretrained GridSat Temporal Encoder (2-layer GRU from Phase 4A)
        self.gridsat_gru = nn.GRU(132, 128, num_layers=2, batch_first=True, dropout=dropout)
        
        # Load weights
        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        gru_weights = {k.replace("gru.", ""): v for k, v in sd.items() if k.startswith("gru.")}
        self.gridsat_gru.load_state_dict(gru_weights)

        # Apply Freezing Strategy
        if mode in ["exp1_gridsat_frozen", "exp2_fusion_frozen"]:
            # Completely freeze pretrained GridSat encoder
            for p in self.gridsat_gru.parameters():
                p.requires_grad = False
        elif mode == "exp3_fusion_partial_unfreeze":
            # Partial unfreeze: freeze layer 0, unfreeze only layer 1 (last stage)
            for name, param in self.gridsat_gru.named_parameters():
                if "_l0" in name:
                    param.requires_grad = False
                elif "_l1" in name:
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Lightweight INSAT-3D Branch
        if mode != "exp1_gridsat_frozen":
            self.insat_branch = LightweightInsatBranch(in_channels=in_channels_insat, spatial_dim=64, gru_dim=64)
            proj_in = 128 + 64
        else:
            self.insat_branch = None
            proj_in = 128

        # 3. Fusion & Shared Latent Projection
        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 4. Multi-Task Prediction Heads
        self.head_center = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.head_class = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 7)
        )
        self.head_wind = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.head_track_12h = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.head_track_24h = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.head_track_48h = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
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
        """Maps [u_lat, u_lon] in [0, 1] to geographic [lat, lon] in degrees."""
        lat = LAT_MIN + norm_coords[..., 0] * LAT_SPAN
        lon = LON_MIN + norm_coords[..., 1] * LON_SPAN
        return torch.stack([lat, lon], dim=-1)

    def unnormalize_wind(self, norm_wind: torch.Tensor) -> torch.Tensor:
        """Converts normalized wind to knots."""
        return norm_wind * self.wind_std + self.wind_mean

    def forward(self,
                gridsat_feat_seq: torch.Tensor,
                insat_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            gridsat_feat_seq: [B, 6, 132]
            insat_seq: Optional [B, 6, C, 72, 116]
        """
        # Pretrained GridSat GRU
        gru_out, _ = self.gridsat_gru(gridsat_feat_seq)
        g_lat = gru_out[:, -1, :]  # [B, 128]

        if self.insat_branch is not None:
            if insat_seq is None:
                raise ValueError("insat_seq must be provided when INSAT branch is active.")
            i_lat = self.insat_branch(insat_seq)  # [B, 64]
            fused = torch.cat([g_lat, i_lat], dim=-1)
        else:
            fused = g_lat

        shared = self.proj(fused)

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
