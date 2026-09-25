"""
VAYU-NET — DECOUPLED MULTI-SOURCE SATELLITE ARCHITECTURE
=========================================================
Problem Statement: SIH 2026 PS 26070
Module: Decoupled Multi-Source Transfer Learning & Selective Routing Suite

Hypothesis:
  Late concatenation of all INSAT channels into one shared latent bottleneck degrades
  spatial trajectory representations. Selective routing provides dedicated task-specific
  pathways:
    - TRACK: Pretrained GridSat (+ optional lightweight TIR1+TIR2 auxiliary branch)
    - CENTER: Pretrained GridSat (+ optional lightweight TIR1+TIR2 auxiliary branch)
    - INTENSITY/WIND: Pretrained GridSat + lightweight TIR1+TIR2+WV auxiliary branch
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


class MultisourceDecoupledModel(nn.Module):
    """
    Decoupled Multi-Source Architecture.
    Allows selective routing of INSAT channels and GridSat representations to dedicated task heads.

    Supported Modes:
      - 'baseline_gridsat': GridSat representation only to all heads.
      - 'arch_a_decoupled_track': Track head receives GridSat + TIR1/TIR2 auxiliary branch. Center & Intensity receive GridSat only.
      - 'arch_b_decoupled_intensity': Intensity heads receive GridSat + TIR1/TIR2/WV auxiliary branch. Center & Track receive GridSat only.
      - 'arch_c_decoupled_full': Center & Track receive GridSat + TIR1/TIR2. Intensity receives GridSat + TIR1/TIR2/WV.
      - 'decoupled_center': Center receives GridSat + TIR1/TIR2. Track & Intensity receive GridSat only.
      - 'ablation_i2_decoupled_intensity_tir': Intensity receives GridSat + TIR1/TIR2. Center & Track receive GridSat only.
    """
    def __init__(self,
                 mode: str = "arch_c_decoupled_full",
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

        # Strictly freeze pretrained GridSat encoder
        for p in self.gridsat_gru.parameters():
            p.requires_grad = False

        # 2. Auxiliary INSAT branches based on architecture mode
        # IR branch: channels 0, 1 (TIR1 + TIR2) -> 64-dim
        needs_ir_branch = mode in [
            "arch_a_decoupled_track",
            "arch_c_decoupled_full",
            "decoupled_center",
            "ablation_i2_decoupled_intensity_tir"
        ]
        if needs_ir_branch:
            self.insat_ir_branch = LightweightInsatBranch(in_channels=2, spatial_dim=64, gru_dim=64)
        else:
            self.insat_ir_branch = None

        # WV/3-channel branch: channels 0, 1, 2 (TIR1 + TIR2 + WV) -> 64-dim
        needs_wv_branch = mode in [
            "arch_b_decoupled_intensity",
            "arch_c_decoupled_full"
        ]
        if needs_wv_branch:
            self.insat_wv_branch = LightweightInsatBranch(in_channels=3, spatial_dim=64, gru_dim=64)
        else:
            self.insat_wv_branch = None

        # 3. Head-Specific Projections (Controlled Fusion)
        # Center projection
        if mode in ["arch_c_decoupled_full", "decoupled_center"]:
            self.proj_center = nn.Sequential(
                nn.LayerNorm(128 + 64),
                nn.Linear(128 + 64, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.proj_center = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Track projection
        if mode in ["arch_a_decoupled_track", "arch_c_decoupled_full"]:
            self.proj_track = nn.Sequential(
                nn.LayerNorm(128 + 64),
                nn.Linear(128 + 64, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.proj_track = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # Intensity/Wind projection
        if mode in ["arch_b_decoupled_intensity", "arch_c_decoupled_full", "ablation_i2_decoupled_intensity_tir"]:
            self.proj_intensity = nn.Sequential(
                nn.LayerNorm(128 + 64),
                nn.Linear(128 + 64, latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.proj_intensity = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, latent_dim),
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
        Forward pass with decoupled selective routing.
        Args:
            gridsat_feat_seq: [B, 6, 132]
            insat_seq: Optional [B, 6, 3, 72, 116] (ch0: TIR1, ch1: TIR2, ch2: WV)
        """
        # 1. Pretrained GridSat GRU representation
        gru_out, _ = self.gridsat_gru(gridsat_feat_seq)
        g_lat = gru_out[:, -1, :]  # [B, 128]

        # 2. Extract auxiliary features if needed
        i_ir = None
        if self.insat_ir_branch is not None:
            if insat_seq is None:
                raise ValueError("insat_seq required for IR branch")
            # Slice channels 0 and 1 (TIR1 and TIR2)
            i_ir = self.insat_ir_branch(insat_seq[:, :, [0, 1], :, :])  # [B, 64]

        i_wv = None
        if self.insat_wv_branch is not None:
            if insat_seq is None:
                raise ValueError("insat_seq required for WV branch")
            # Slice all 3 channels (TIR1, TIR2, WV)
            i_wv = self.insat_wv_branch(insat_seq[:, :, [0, 1, 2], :, :])  # [B, 64]

        # 3. Route to task-specific representations
        # Center latent
        if self.mode in ["arch_c_decoupled_full", "decoupled_center"]:
            center_latent = self.proj_center(torch.cat([g_lat, i_ir], dim=-1))
        else:
            center_latent = self.proj_center(g_lat)

        # Track latent
        if self.mode in ["arch_a_decoupled_track", "arch_c_decoupled_full"]:
            track_latent = self.proj_track(torch.cat([g_lat, i_ir], dim=-1))
        else:
            track_latent = self.proj_track(g_lat)

        # Intensity latent
        if self.mode in ["arch_b_decoupled_intensity", "arch_c_decoupled_full"]:
            intensity_latent = self.proj_intensity(torch.cat([g_lat, i_wv], dim=-1))
        elif self.mode == "ablation_i2_decoupled_intensity_tir":
            intensity_latent = self.proj_intensity(torch.cat([g_lat, i_ir], dim=-1))
        else:
            intensity_latent = self.proj_intensity(g_lat)

        # 4. Predict from dedicated representations
        norm_center = self.head_center(center_latent)
        class_logits = self.head_class(intensity_latent)
        norm_wind = self.head_wind(intensity_latent)
        norm_t12 = self.head_track_12h(track_latent)
        norm_t24 = self.head_track_24h(track_latent)
        norm_t48 = self.head_track_48h(track_latent)

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
            "center_latent": center_latent,
            "track_latent": track_latent,
            "intensity_latent": intensity_latent
        }
