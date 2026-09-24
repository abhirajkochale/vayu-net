"""
VAYU-NET Phase 4B — Hybrid Kinematic + Satellite Residual Track Prediction Model
(HybridKinematicSatelliteGRU)

Architecture:
  1. Shared Frozen Spatial Encoder (Phase 3C DedicatedCenterLocalizationResNet):
     Extracts spatial representation, CoordConv features, and soft-argmax center coordinates.
  2. Frame-level Spatial Feature Aggregation:
     AdaptiveAvgPool2d + Linear projection -> 128-dim spatial embedding.
     Per-frame feature vector: [spatial_emb (128), norm_lat (1), norm_lon (1), delta_lat (1), delta_lon (1)] = 132 dims.
  3. Causal Sequence Modeling:
     2-layer GRU across 6 consecutive timestamps (t-15h to t0).
     Extracts final temporal hidden state at t0: h_6 [B, 128].
  4. Kinematic Context Fusion:
     Appends normalized recent motion features:
     [v_north_km_h, v_east_km_h, speed_km_h, anchor_lat_norm, anchor_lon_norm] -> 5 dims.
     Combined context representation: 128 + 5 = 133 dims.
  5. Multi-Horizon Residual Heads:
     - Head +12h: Linear(133, 64) -> ReLU -> Linear(64, 2) -> [res_north_km, res_east_km]
     - Head +24h: Linear(133, 64) -> ReLU -> Linear(64, 2) -> [res_north_km, res_east_km]
     - Head +48h: Linear(133, 64) -> ReLU -> Linear(64, 2) -> [res_north_km, res_east_km]
  6. Invertible Local Tangent-Plane Geodesic Reconstruction:
     Reconstructed Coordinates:
       lat_hat = lat_kin + res_north_km / (R_EARTH * DEG_TO_RAD)
       lon_hat = lon_kin + res_east_km / (R_EARTH * cos(lat_kin) * DEG_TO_RAD)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.center_localization_cnn import (
    DedicatedCenterLocalizationResNet,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN
)

R_EARTH_KM = 6371.0
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
RESIDUAL_SCALE_KM = 100.0  # Normalized residual scale factor for optimization stability

class HybridKinematicSatelliteGRU(nn.Module):
    """
    Hybrid Kinematic + Satellite Residual Track Forecasting Network.
    Predicts physical residual corrections (northward and eastward in km)
    relative to a constant-velocity baseline forecast.
    """
    def __init__(self,
                 spatial_checkpoint_path=None,
                 freeze_spatial_encoder=True,
                 spatial_emb_dim=128,
                 gru_hidden_dim=128,
                 gru_num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.spatial_emb_dim = spatial_emb_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        self.freeze_spatial_encoder = freeze_spatial_encoder
        
        # 1. Shared Spatial Encoder (Transferred from Phase 3C)
        self.spatial_encoder = DedicatedCenterLocalizationResNet()
        if spatial_checkpoint_path is not None:
            self.load_spatial_encoder_weights(spatial_checkpoint_path)
        if self.freeze_spatial_encoder:
            self._set_spatial_encoder_grad(False)
            
        # 2. Spatial Aggregation Layer
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 3)) # [B, 128, 2, 3] = 768 dims
        self.spatial_proj = nn.Sequential(
            nn.Linear(128 * 2 * 3, spatial_emb_dim),
            nn.BatchNorm1d(spatial_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Frame feature dimension: 128 + 2 (norm_center) + 2 (delta_center) = 132
        self.frame_feat_dim = spatial_emb_dim + 4
        
        # 3. Temporal Sequence GRU
        self.gru = nn.GRU(
            input_size=self.frame_feat_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0
        )
        
        # 4. Kinematic Motion Context Integration
        # Kinematic vector: [v_north_km_h, v_east_km_h, speed_km_h, anchor_lat_norm, anchor_lon_norm] = 5 dims
        self.kin_feat_dim = 5
        self.fused_dim = gru_hidden_dim + self.kin_feat_dim
        
        # 5. Multi-Horizon Residual Heads
        # Each head predicts normalized residual [r_north / 100, r_east / 100]
        self.head_12h = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.head_24h = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )
        self.head_48h = nn.Sequential(
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def _set_spatial_encoder_grad(self, requires_grad):
        """Enables or disables gradients for the shared spatial encoder."""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = requires_grad

    def load_spatial_encoder_weights(self, checkpoint_path):
        """Transfers trained spatial weights from Phase 3C checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        missing_keys, unexpected_keys = self.spatial_encoder.load_state_dict(state_dict, strict=False)
        print(f"[HybridModel] Transferred spatial weights from: {checkpoint_path}")
        print(f"  Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if self.freeze_spatial_encoder:
            self._set_spatial_encoder_grad(False)

    def extract_single_frame_feature(self, x):
        """
        Extracts spatial embedding and estimated center from a single frame [B, 1, 572, 929].
        """
        B = x.size(0)
        x1 = self.spatial_encoder.relu(self.spatial_encoder.bn1(self.spatial_encoder.conv1(x)))
        mp = self.spatial_encoder.maxpool(x1)
        l1 = self.spatial_encoder.layer1(mp)
        l2 = self.spatial_encoder.layer2(l1)
        
        pooled = self.spatial_pool(l2).view(B, -1)
        spatial_emb = self.spatial_proj(pooled)
        
        coords = self.spatial_encoder.coord_channels.expand(B, -1, -1, -1)
        feat_spatial = torch.cat([l2, coords], dim=1)
        heatmap_logits = self.spatial_encoder.center_decoder(feat_spatial)
        norm_center, _ = self.spatial_encoder.decode_soft_argmax(heatmap_logits)
        return spatial_emb, norm_center

    def forward_features(self, feat_sequence, kin_features):
        """
        Forward pass directly from pre-computed feature sequence [B, 6, 132]
        and kinematic context vector [B, 5].
        Returns:
            predicted_residuals_km: dict of tensors with shape [B, 2] in physical km.
            predicted_residuals_norm: dict of tensors with shape [B, 2] in scaled units.
        """
        B, T, D = feat_sequence.shape
        gru_out, _ = self.gru(feat_sequence)
        final_state = gru_out[:, -1, :] # [B, gru_hidden_dim]
        
        # Fuse with kinematic context
        fused = torch.cat([final_state, kin_features], dim=-1) # [B, fused_dim]
        
        # Raw heads predict normalized residual units
        r12_norm = self.head_12h(fused)
        r24_norm = self.head_24h(fused)
        r48_norm = self.head_48h(fused)
        
        # Rescale to physical kilometers
        r12_km = r12_norm * RESIDUAL_SCALE_KM
        r24_km = r24_norm * RESIDUAL_SCALE_KM
        r48_km = r48_norm * RESIDUAL_SCALE_KM
        
        return {
            "r12_norm": r12_norm,
            "r24_norm": r24_norm,
            "r48_norm": r48_norm,
            "r12_km": r12_km,
            "r24_km": r24_km,
            "r48_km": r48_km
        }

    @staticmethod
    def latlon_to_residual_km(kin_coords, true_coords):
        """
        Computes local tangent-plane residual displacement [res_north_km, res_east_km]
        from kinematic coordinate to true coordinate.
        Args:
            kin_coords: Tensor [..., 2] of [lat, lon]
            true_coords: Tensor [..., 2] of [lat, lon]
        Returns:
            residual_km: Tensor [..., 2] of [res_north_km, res_east_km]
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
        Reconstructs physical latitude/longitude by adding local tangent-plane residual
        displacement [res_north_km, res_east_km] to kinematic base coordinate.
        Args:
            kin_coords: Tensor [..., 2] of [lat, lon]
            residual_km: Tensor [..., 2] of [res_north_km, res_east_km]
            clip_bounds: If True, safely clamps final coordinates within NIO geographic domain
        Returns:
            reconstructed_coords: Tensor [..., 2] of [lat, lon]
        """
        kin_lat = kin_coords[..., 0]
        kin_lon = kin_coords[..., 1]
        res_north = residual_km[..., 0]
        res_east = residual_km[..., 1]
        
        pred_lat = kin_lat + (res_north / (R_EARTH_KM * DEG_TO_RAD))
        cos_lat = torch.clamp(torch.cos(kin_lat * DEG_TO_RAD), min=0.01) # Avoid div by zero near poles
        pred_lon = kin_lon + (res_east / (R_EARTH_KM * cos_lat * DEG_TO_RAD))
        
        if clip_bounds:
            pred_lat = torch.clamp(pred_lat, min=LAT_MIN, max=LAT_MIN + LAT_SPAN)
            pred_lon = torch.clamp(pred_lon, min=LON_MIN, max=LON_MIN + LON_SPAN)
            
        return torch.stack([pred_lat, pred_lon], dim=-1)
