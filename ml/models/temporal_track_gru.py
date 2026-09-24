"""
VAYU-NET Phase 4A — Temporal Cyclone Track Prediction Model (TemporalTrackGRU)

Architecture:
  1. Shared Spatial Encoder:
     DedicatedCenterLocalizationResNet backbone (transferred from Phase 3C checkpoint).
     Extracts spatial representation, CoordConv-augmented features, and estimated center coordinates.
  2. Frame-level Spatial Feature Aggregation:
     Adaptive pooling + linear projection -> compact spatial embedding (128 dims).
     Appends soft-argmax estimated center [lat, lon] and frame-to-frame displacement [delta_lat, delta_lon].
     Resulting frame feature vector: 132 dims per timestep.
  3. Temporal Sequence Modeling:
     2-layer bidirectional or unidirectional GRU across 6 consecutive timestamps (t-15h to t0).
     Sequence shape: [B, 6, 132].
  4. Multi-Horizon Forecast Heads:
     From the final temporal hidden state at t0:
     - Head +12h: Linear -> ReLU -> Linear -> [B, 2] (normalized lat, lon in [0, 1])
     - Head +24h: Linear -> ReLU -> Linear -> [B, 2] (normalized lat, lon in [0, 1])
     - Head +48h: Linear -> ReLU -> Linear -> [B, 2] (normalized lat, lon in [0, 1])
  5. Strict Anti-Leakage Invariants:
     Temporal sequence terminates at t0. Zero future satellite frames or labels enter the model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.center_localization_cnn import (
    DedicatedCenterLocalizationResNet,
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    HEATMAP_H, HEATMAP_W
)

class TemporalTrackGRU(nn.Module):
    """
    Temporal Track Prediction Network using Shared Spatial Encoder + GRU.
    """
    def __init__(self,
                 spatial_checkpoint_path=None,
                 freeze_spatial_encoder=True,
                 use_center_features=True,
                 spatial_emb_dim=128,
                 gru_hidden_dim=128,
                 gru_num_layers=2,
                 dropout=0.1):
        super().__init__()
        
        self.use_center_features = use_center_features
        self.spatial_emb_dim = spatial_emb_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_num_layers = gru_num_layers
        
        # 1. Shared Spatial Encoder (Phase 3C Architecture)
        self.spatial_encoder = DedicatedCenterLocalizationResNet()
        self.freeze_spatial_encoder = freeze_spatial_encoder
        
        if spatial_checkpoint_path is not None:
            self.load_spatial_encoder_weights(spatial_checkpoint_path)
            
        if self.freeze_spatial_encoder:
            self._set_spatial_encoder_grad(False)
            
        # 2. Spatial Aggregation Layer
        # layer2 features are [B, 128, 72, 117]. Adaptive pool to [B, 128, 2, 3] = 768 dims
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 3))
        self.spatial_proj = nn.Sequential(
            nn.Linear(128 * 2 * 3, spatial_emb_dim),
            nn.BatchNorm1d(spatial_emb_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature dimension per frame:
        # spatial_emb_dim (128) + 2 (norm_lat, norm_lon) + 2 (delta_lat, delta_lon) = 132
        self.frame_feat_dim = spatial_emb_dim + (4 if use_center_features else 0)
        
        # 3. Temporal Sequence GRU
        self.gru = nn.GRU(
            input_size=self.frame_feat_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout if gru_num_layers > 1 else 0.0
        )
        
        # 4. Multi-Horizon Track Prediction Heads
        # Each head predicts normalized coordinates [u_lat, u_lon] in [0, 1]
        self.head_12h = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.head_24h = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        self.head_48h = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def _set_spatial_encoder_grad(self, requires_grad):
        """Enables or disables gradients for the shared spatial encoder."""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = requires_grad
            
    def load_spatial_encoder_weights(self, checkpoint_path):
        """Transfers trained spatial weights from Phase 3C checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Load weights into spatial encoder
        missing_keys, unexpected_keys = self.spatial_encoder.load_state_dict(state_dict, strict=False)
        print(f"[TemporalTrackGRU] Loaded Phase 3C weights from: {checkpoint_path}")
        print(f"  Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if self.freeze_spatial_encoder:
            self._set_spatial_encoder_grad(False)

    def extract_single_frame_feature(self, x):
        """
        Extracts spatial embedding and estimated center from a single frame [B, 1, 572, 929].
        Returns:
            spatial_emb: [B, spatial_emb_dim]
            norm_center: [B, 2] in [0, 1]
        """
        B = x.size(0)
        
        # Backbone intermediate stages
        x1 = self.spatial_encoder.conv1(x)
        x1 = self.spatial_encoder.bn1(x1)
        x1 = self.spatial_encoder.relu(x1)
        mp = self.spatial_encoder.maxpool(x1)
        
        l1 = self.spatial_encoder.layer1(mp)     # [B, 64, 143, 233]
        l2 = self.spatial_encoder.layer2(l1)     # [B, 128, 72, 117]
        
        # Spatial pooling for compact embedding
        pooled = self.spatial_pool(l2).view(B, -1) # [B, 768]
        spatial_emb = self.spatial_proj(pooled)    # [B, spatial_emb_dim]
        
        # Center Spatial Localization Branch
        coords = self.spatial_encoder.coord_channels.expand(B, -1, -1, -1)
        feat_spatial = torch.cat([l2, coords], dim=1)
        heatmap_logits = self.spatial_encoder.center_decoder(feat_spatial)
        norm_center, _ = self.spatial_encoder.decode_soft_argmax(heatmap_logits) # [B, 2]
        
        return spatial_emb, norm_center

    def extract_sequence_features(self, sequence_tensor):
        """
        Extracts temporal feature representations across all 6 frames.
        Args:
            sequence_tensor: [B, 6, 572, 929]
        Returns:
            feat_sequence: [B, 6, frame_feat_dim]
            estimated_centers: [B, 6, 2]
        """
        B, T, H, W = sequence_tensor.shape
        assert T == 6, f"Expected sequence length 6, got {T}"
        
        frame_features = []
        estimated_centers = []
        
        prev_center = None
        for t in range(T):
            frame_t = sequence_tensor[:, t:t+1, :, :] # [B, 1, H, W]
            emb_t, center_t = self.extract_single_frame_feature(frame_t)
            estimated_centers.append(center_t)
            
            if self.use_center_features:
                if prev_center is None:
                    delta_t = torch.zeros_like(center_t)
                else:
                    delta_t = center_t - prev_center
                prev_center = center_t
                
                feat_t = torch.cat([emb_t, center_t, delta_t], dim=-1) # [B, 132]
            else:
                feat_t = emb_t # [B, 128]
                
            frame_features.append(feat_t)
            
        feat_sequence = torch.stack(frame_features, dim=1)      # [B, 6, frame_feat_dim]
        estimated_centers = torch.stack(estimated_centers, dim=1) # [B, 6, 2]
        return feat_sequence, estimated_centers

    def forward_features(self, feat_sequence):
        """
        Forward pass directly from pre-computed feature sequence [B, 6, frame_feat_dim].
        Enables high-throughput training when spatial encoder is frozen.
        """
        B, T, D = feat_sequence.shape
        gru_out, h_n = self.gru(feat_sequence)
        final_state = gru_out[:, -1, :] # Final state at t0 [B, gru_hidden_dim]
        
        pred_12h = self.head_12h(final_state) # [B, 2]
        pred_24h = self.head_24h(final_state) # [B, 2]
        pred_48h = self.head_48h(final_state) # [B, 2]
        
        return {
            "pred_norm_12h": pred_12h,
            "pred_norm_24h": pred_24h,
            "pred_norm_48h": pred_48h,
            "final_state": final_state
        }

    def forward(self, sequence_tensor):
        """
        End-to-end forward pass from raw 6-frame satellite tensor [B, 6, 572, 929].
        """
        if self.freeze_spatial_encoder:
            with torch.no_grad():
                feat_seq, est_centers = self.extract_sequence_features(sequence_tensor)
        else:
            feat_seq, est_centers = self.extract_sequence_features(sequence_tensor)
            
        out = self.forward_features(feat_seq)
        out["estimated_historical_centers"] = est_centers
        return out

    @staticmethod
    def denormalize_coords(norm_coords):
        """Converts normalized [u_lat, u_lon] in [0, 1] to physical degrees."""
        u_lat = norm_coords[..., 0]
        u_lon = norm_coords[..., 1]
        lat = LAT_MIN + u_lat * LAT_SPAN
        lon = LON_MIN + u_lon * LON_SPAN
        return torch.stack([lat, lon], dim=-1)

    @staticmethod
    def normalize_coords(deg_coords):
        """Converts physical degrees [lat, lon] to normalized [0, 1]."""
        lat = deg_coords[..., 0]
        lon = deg_coords[..., 1]
        u_lat = (lat - LAT_MIN) / LAT_SPAN
        u_lon = (lon - LON_MIN) / LON_SPAN
        return torch.stack([u_lat, u_lon], dim=-1)
