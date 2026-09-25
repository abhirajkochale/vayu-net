"""
VAYU-NET: EXP-M3 Multi-Task Architecture with Granular Spatially Adaptive Fusion.

Configurations:
  - M3A (GridSat-only): 1-channel IR sequence [B, 6, 1, 72, 116] (Reference Control, batch_size=16)
  - M3B (IMERG-only):   1-channel IMERG precipitation sequence [B, 6, 1, 72, 116] (Reference Control, batch_size=16)
  - M3C (Spatially Adaptive Fusion):
      Separate GridSat and IMERG feature maps G_t, I_t in R^(64 x 9 x 15).
      Concatenated map X_t = concat(G_t, I_t) in R^(128 x 9 x 15).
      Lightweight 1x1 Conv gate -> A_t in R^(64 x 9 x 15) in [0, 1].
      Element-wise fusion: F_t = A_t * G_t + (1 - A_t) * I_t in R^(64 x 9 x 15).
      Pooling/projection -> 64-dim embedding per timestep -> temporal unidirectional GRU.

Prediction Heads (Preserving exact EXP-M1/M2 target formulation):
  1. Center localization: [u_lat, u_lon] in [0, 1] at t0 (absolute coordinates)
  2. Intensity classification: 7-class IMD category logits
  3. Wind speed regression: Standardized continuous wind in knots
  4. Track prediction: Absolute normalized [u_lat, u_lon] in [0, 1] for +12h, +24h, +48h
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


class SpatialFeatureExtractor(nn.Module):
    """
    3-stage convolutional backbone mapping [B, 1, 72, 116] to spatial feature map [B, 64, 9, 15].
    Architecture is identical to the convolutional stages of EXP-M1/M2 CompactSpatialCNN.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, 64, 9, 15]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        return x


class SpatialPoolingHead(nn.Module):
    """
    Pools and projects a spatial feature map [B, 64, 9, 15] to embedding [B, embed_dim].
    Matches the pooling and linear projection of EXP-M1/M2 CompactSpatialCNN.
    """
    def __init__(self, in_channels: int = 64, embed_dim: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(in_channels * 2 * 2, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        # feat_map: [B, 64, 9, 15]
        p = self.pool(feat_map)
        flat = torch.flatten(p, 1)
        return F.gelu(self.ln(self.fc(flat)))


class SpatialGatingUnit(nn.Module):
    """
    Lightweight 1x1 Conv spatial gating network for Granular Fusion.
    Takes concatenated feature maps X_t = concat(G_t, I_t) in R^(128 x 9 x 15).
    Produces spatial gate map A_t in R^(64 x 9 x 15) in [0, 1].
    Combines: F_t = A_t * G_t + (1 - A_t) * I_t.
    """
    def __init__(self, channels: int = 64, hidden_channels: int = 32):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self,
                sat_map: torch.Tensor,
                imerg_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sat_map:   [B, 64, 9, 15]
            imerg_map: [B, 64, 9, 15]
        Returns:
            fused_map: [B, 64, 9, 15]
            alpha_map: [B, 64, 9, 15] (GridSat spatial weight; IMERG weight is 1 - alpha)
        """
        x_concat = torch.cat([sat_map, imerg_map], dim=1)       # [B, 128, 9, 15]
        alpha = self.gate_net(x_concat)                          # [B, 64, 9, 15]
        fused = alpha * sat_map + (1.0 - alpha) * imerg_map     # [B, 64, 9, 15]
        return fused, alpha


class ExpM3Model(nn.Module):
    """
    Controlled Ablation Model for EXP-M3:
      - mode='m3a_gridsat': GridSat-only control (matching M1A architecture)
      - mode='m3b_imerg':   IMERG-only control (matching M1B architecture)
      - mode='m3c_spatial': Granular Spatially-Adaptive Multimodal Fusion
    """
    def __init__(self,
                 mode: str = "m3c_spatial",
                 spatial_dim: int = 64,
                 gru_dim: int = 64,
                 latent_dim: int = 64,
                 dropout: float = 0.1,
                 wind_mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                 wind_std: float = DEFAULT_TRAIN_WIND_STD_KT):
        super().__init__()
        if mode not in ["m3a_gridsat", "m3b_imerg", "m3c_spatial"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'm3a_gridsat', 'm3b_imerg', or 'm3c_spatial'.")

        self.mode = mode
        self.spatial_dim = spatial_dim
        self.gru_dim = gru_dim
        self.latent_dim = latent_dim
        self.wind_mean = wind_mean
        self.wind_std = wind_std

        # Modality Feature Extractors
        if mode in ["m3a_gridsat", "m3c_spatial"]:
            self.gridsat_extractor = SpatialFeatureExtractor(in_channels=1, out_channels=spatial_dim)
        else:
            self.gridsat_extractor = None

        if mode in ["m3b_imerg", "m3c_spatial"]:
            self.imerg_extractor = SpatialFeatureExtractor(in_channels=1, out_channels=spatial_dim)
        else:
            self.imerg_extractor = None

        # Spatial Gating Unit (M3C only)
        if mode == "m3c_spatial":
            self.spatial_gate = SpatialGatingUnit(channels=spatial_dim, hidden_channels=32)
        else:
            self.spatial_gate = None

        # Spatial Pooling Head (Maps fused/unimodal map [B, 64, 9, 15] to embedding [B, 64])
        self.pooling_head = SpatialPoolingHead(in_channels=spatial_dim, embed_dim=spatial_dim)

        # Temporal GRU: always processes sequence [B, 6, spatial_dim]
        # Unidirectional single-layer GRU
        self.temporal_gru = nn.GRU(spatial_dim, gru_dim, num_layers=1, batch_first=True)

        # Shared Projection from temporal latent
        self.proj = nn.Sequential(
            nn.LayerNorm(gru_dim),
            nn.Linear(gru_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Multi-Task Prediction Heads (Identical to EXP-M1/M2)
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
        if gridsat_seq is not None and gridsat_seq.ndim == 4:
            gridsat_seq = gridsat_seq.unsqueeze(2)
        if imerg_seq is not None and imerg_seq.ndim == 4:
            imerg_seq = imerg_seq.unsqueeze(2)

        spatial_gate_maps = None

        if self.mode == "m3a_gridsat":
            if gridsat_seq is None:
                raise ValueError("gridsat_seq must be provided when mode='m3a_gridsat'")
            B, T, C, H, W = gridsat_seq.shape
            x_flat = gridsat_seq.view(B * T, C, H, W)
            feat_maps = self.gridsat_extractor(x_flat)                       # [B*T, 64, 9, 15]
            spatial_embeddings = self.pooling_head(feat_maps).view(B, T, -1) # [B, T, 64]
            temporal_input = spatial_embeddings

        elif self.mode == "m3b_imerg":
            if imerg_seq is None:
                raise ValueError("imerg_seq must be provided when mode='m3b_imerg'")
            B, T, C, H, W = imerg_seq.shape
            x_flat = imerg_seq.view(B * T, C, H, W)
            feat_maps = self.imerg_extractor(x_flat)                         # [B*T, 64, 9, 15]
            spatial_embeddings = self.pooling_head(feat_maps).view(B, T, -1) # [B, T, 64]
            temporal_input = spatial_embeddings

        else: # m3c_spatial
            if gridsat_seq is None or imerg_seq is None:
                raise ValueError("Both gridsat_seq and imerg_seq must be provided when mode='m3c_spatial'")
            B, T, C_g, H, W = gridsat_seq.shape
            _, _, C_i, _, _ = imerg_seq.shape

            g_flat = gridsat_seq.view(B * T, C_g, H, W)
            i_flat = imerg_seq.view(B * T, C_i, H, W)

            g_maps = self.gridsat_extractor(g_flat)                          # [B*T, 64, 9, 15]
            i_maps = self.imerg_extractor(i_flat)                            # [B*T, 64, 9, 15]

            # Granular Spatially Adaptive Gating
            fused_maps, alpha_maps = self.spatial_gate(g_maps, i_maps)       # [B*T, 64, 9, 15], [B*T, 64, 9, 15]
            spatial_embeddings = self.pooling_head(fused_maps).view(B, T, -1)# [B, T, 64]
            temporal_input = spatial_embeddings
            spatial_gate_maps = alpha_maps.view(B, T, self.spatial_dim, 9, 15)

        # Temporal modeling over 6 steps using unidirectional GRU
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

        out = {
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
        if spatial_gate_maps is not None:
            out["spatial_gate_alpha"] = spatial_gate_maps          # [B, 6, 64, 9, 15] (GridSat spatial weight)
            out["gridsat_spatial_weight"] = spatial_gate_maps
            out["imerg_spatial_weight"] = 1.0 - spatial_gate_maps
        return out
