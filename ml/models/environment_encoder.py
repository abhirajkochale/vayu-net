"""
VAYU-NET PHASE 5A — ENVIRONMENTAL ENCODER ARCHITECTURE
=======================================================
Processes multi-level atmospheric wind fields from ERA5:
  - 8 input channels: (u, v) at (850, 700, 500, 300) hPa
  - Grid: [B, 6, 8, 41, 66]
  - Spatial CNN: 3-stage lightweight CNN with adaptive pooling -> 64-d embedding
  - Temporal GRU: 2-layer causal recurrent network -> terminal representation e(t0)
  - Standalone Environment-Only Track Model for Experiment B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnvironmentSpatialCNN(nn.Module):
    """
    Lightweight 2D CNN encoder for single-timestamp 8-channel ERA5 wind field.
    Input: [B, 8, H, W] where H~41, W~66
    Output: [B, embed_dim] (default 64)
    """
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
        # x: [B, 8, 41, 66]
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x) # [B, 32, 20, 33]
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x) # [B, 64, 10, 16]
        
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x) # [B, 64, 2, 2]
        
        flat = torch.flatten(x, 1) # [B, 256]
        embed = self.proj(flat) # [B, embed_dim]
        return embed


class EnvironmentTemporalGRU(nn.Module):
    """
    Temporal sequence encoder for 6 ERA5 environmental frames.
    Input: [B, 6, 8, 41, 66]
    Output: terminal hidden state h_env(t0) of shape [B, hidden_dim]
    """
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
        spatial_feats = self.spatial_encoder(flat_seq) # [B*T, spatial_dim]
        temporal_feats = spatial_feats.view(B, T, -1) # [B, T, spatial_dim]
        
        gru_out, h_n = self.gru(temporal_feats) # h_n: [num_layers, B, hidden_dim]
        # Return terminal state at t0
        terminal_state = h_n[-1] # [B, hidden_dim]
        return terminal_state, temporal_feats


class EnvironmentOnlyTrackModel(nn.Module):
    """
    Experiment B: Standalone environmental model predicting future cyclone tracks
    using ONLY 6-frame ERA5 atmospheric steering wind fields.
    Outputs +12h, +24h, +48h (lat, lon) coordinates directly.
    """
    def __init__(self, in_channels=8, spatial_dim=64, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.env_gru = EnvironmentTemporalGRU(
            in_channels=in_channels,
            spatial_dim=spatial_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Geographic multi-horizon track regression heads
        self.head_12 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2) # [lat, lon]
        )
        self.head_24 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.head_48 = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, env_seq):
        # env_seq: [B, 6, 8, 41, 66]
        h_env, _ = self.env_gru(env_seq) # [B, hidden_dim]
        
        pred_12 = self.head_12(h_env)
        pred_24 = self.head_24(h_env)
        pred_48 = self.head_48(h_env)
        
        return {
            'pred_12': pred_12,
            'pred_24': pred_24,
            'pred_48': pred_48,
            'h_env': h_env
        }
