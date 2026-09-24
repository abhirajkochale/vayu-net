"""
VAYU-NET PHASE 5B — ENVIRONMENT-AWARE HYBRID RESIDUAL TRACK MODEL
==================================================================
Joint architecture predicting physical residual displacements (km)
relative to a constant-velocity kinematic forecast:
  FINAL_FORECAST_h = KINEMATIC_FORECAST_h + PREDICTED_RESIDUAL_h

Inputs:
  1. Satellite Sequence: [B, 6, 132] (from frozen Phase 3C spatial encoder)
  2. Environmental Sequence: [B, 6, 8, 41, 66] (ERA5 4-level U/V wind fields)
  3. Causal Motion Context: [B, 5] (recent velocity, speed, anchor position)
  4. Kinematic Forecast Anchor: [B, 2] per horizon (+12h, +24h, +48h)

Coordinate System:
  Local tangent-plane geodesic displacement [res_north_km, res_east_km]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

R_EARTH_KM = 6371.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RESIDUAL_SCALE_KM = 100.0  # Optimization scaling factor

LAT_MIN = -5.0
LAT_SPAN = 40.0
LON_MIN = 40.0
LON_SPAN = 65.0

class EnvironmentSpatialCNN(nn.Module):
    def __init__(self, in_channels=8, embed_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.proj = nn.Sequential(
            nn.Linear(64 * 2 * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        flat = torch.flatten(x, 1)
        return self.proj(flat)


class EnvironmentTemporalGRU(nn.Module):
    def __init__(self, in_channels=8, spatial_dim=64, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.spatial_encoder = EnvironmentSpatialCNN(in_channels=in_channels, embed_dim=spatial_dim)
        self.gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.hidden_dim = hidden_dim

    def forward(self, env_seq):
        # env_seq: [B, T=6, 8, 41, 66]
        B, T, C, H, W = env_seq.shape
        flat_seq = env_seq.view(B * T, C, H, W)
        spatial_feats = self.spatial_encoder(flat_seq)
        temporal_feats = spatial_feats.view(B, T, -1)
        _, h_n = self.gru(temporal_feats)
        return h_n[-1] # [B, hidden_dim]


class Phase5BHybridResidualModel(nn.Module):
    """
    Multimodal Hybrid Residual Track Model fusing Satellite features,
    ERA5 atmospheric wind fields, and kinematic motion context.
    """
    def __init__(
        self,
        sat_feat_dim=132,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        motion_in_dim=5,
        motion_emb_dim=16,
        fusion_dim=128,
        dropout=0.1
    ):
        super().__init__()
        
        # 1. Satellite GRU
        self.sat_gru = nn.GRU(
            input_size=sat_feat_dim,
            hidden_size=sat_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # 2. Environmental GRU
        self.env_gru = EnvironmentTemporalGRU(
            in_channels=env_channels,
            spatial_dim=env_spatial_dim,
            hidden_dim=env_hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # 3. Motion Feature Projection
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_in_dim, motion_emb_dim),
            nn.LayerNorm(motion_emb_dim),
            nn.GELU()
        )
        
        # 4. Multimodal Fusion
        total_fused_dim = sat_hidden_dim + env_hidden_dim + motion_emb_dim # 128 + 64 + 16 = 208
        self.fusion = nn.Sequential(
            nn.Linear(total_fused_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 5. Multi-Horizon Residual Heads (outputs normalized residuals [r_north, r_east])
        self.res_head_12 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.res_head_24 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.res_head_48 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, sat_seq, env_seq, motion_ctx, kin_12=None, kin_24=None, kin_48=None):
        """
        Forward pass predicting residuals and reconstructed coordinates.
        sat_seq: [B, 6, 132]
        env_seq: [B, 6, 8, 41, 66]
        motion_ctx: [B, 5]
        kin_12, kin_24, kin_48: Optional [B, 2] kinematic forecasts
        """
        # Satellite state
        _, h_sat_n = self.sat_gru(sat_seq)
        h_sat = h_sat_n[-1] # [B, 128]
        
        # Environmental state
        h_env = self.env_gru(env_seq) # [B, 64]
        
        # Motion context
        h_mot = self.motion_proj(motion_ctx) # [B, 16]
        
        # Fuse
        fused = torch.cat([h_sat, h_env, h_mot], dim=-1) # [B, 208]
        h_fused = self.fusion(fused) # [B, 128]
        
        # Predict normalized residuals
        r12_norm = self.res_head_12(h_fused)
        r24_norm = self.res_head_24(h_fused)
        r48_norm = self.res_head_48(h_fused)
        
        # Scale to km
        r12_km = r12_norm * RESIDUAL_SCALE_KM
        r24_km = r24_norm * RESIDUAL_SCALE_KM
        r48_km = r48_norm * RESIDUAL_SCALE_KM
        
        out = {
            'r12_norm': r12_norm,
            'r24_norm': r24_norm,
            'r48_norm': r48_norm,
            'r12_km': r12_km,
            'r24_km': r24_km,
            'r48_km': r48_km,
            'h_sat': h_sat,
            'h_env': h_env,
            'h_fused': h_fused
        }
        
        # If kinematic anchors supplied, compute reconstructed physical coordinates
        if kin_12 is not None and kin_24 is not None and kin_48 is not None:
            out['pred_12'] = self.residual_km_to_latlon(kin_12, r12_km)
            out['pred_24'] = self.residual_km_to_latlon(kin_24, r24_km)
            out['pred_48'] = self.residual_km_to_latlon(kin_48, r48_km)
            
        return out

    @staticmethod
    def latlon_to_residual_km(kin_coords, true_coords):
        """
        Calculates local tangent-plane geodesic displacement [res_north_km, res_east_km]
        from kinematic base coordinate to ground-truth coordinate.
        """
        kin_lat = kin_coords[..., 0]
        kin_lon = kin_coords[..., 1]
        true_lat = true_coords[..., 0]
        true_lon = true_coords[..., 1]
        
        res_north = (true_lat - kin_lat) * DEG_TO_RAD * R_EARTH_KM
        cos_lat = torch.cos(kin_lat * DEG_TO_RAD)
        res_east = (true_lon - kin_lon) * cos_lat * DEG_TO_RAD * R_EARTH_KM
        return torch.stack([res_north, res_east], dim=-1)

    @staticmethod
    def residual_km_to_latlon(kin_coords, residual_km, clip_bounds=True):
        """
        Invertibly reconstructs physical latitude/longitude from kinematic anchor
        and local displacement vector [res_north_km, res_east_km].
        """
        kin_lat = kin_coords[..., 0]
        kin_lon = kin_coords[..., 1]
        res_north = residual_km[..., 0]
        res_east = residual_km[..., 1]
        
        pred_lat = kin_lat + (res_north / (R_EARTH_KM * DEG_TO_RAD))
        cos_lat = torch.clamp(torch.cos(kin_lat * DEG_TO_RAD), min=0.01)
        pred_lon = kin_lon + (res_east / (R_EARTH_KM * cos_lat * DEG_TO_RAD))
        
        if clip_bounds:
            pred_lat = torch.clamp(pred_lat, min=LAT_MIN, max=LAT_MIN + LAT_SPAN)
            pred_lon = torch.clamp(pred_lon, min=LON_MIN, max=LON_MIN + LON_SPAN)
            
        return torch.stack([pred_lat, pred_lon], dim=-1)
