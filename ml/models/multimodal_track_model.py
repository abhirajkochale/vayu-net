"""
VAYU-NET PHASE 5A — MULTIMODAL TRACK PREDICTION MODEL
======================================================
Joint architecture integrating:
1. SATELLITE BRANCH:
   - 6-frame GridSat-B1 sequence [B, 6, 1, 572, 929]
   - Frozen Phase 3C ResNet-18 spatial encoder (11.26M params)
   - Spatial projection + 2-layer satellite GRU (128 hidden) -> h_sat(t0)
2. ENVIRONMENTAL BRANCH:
   - 6-frame ERA5 sequence [B, 6, 8, 41, 66]
   - 3-stage environmental CNN + 2-layer environmental GRU (64 hidden) -> h_env(t0)
3. MULTIMODAL FUSION & DECODER:
   - Late concatenation fusion: [h_sat, h_env] (192-d)
   - Multi-horizon track regression heads for +12h, +24h, +48h
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from ml.models.environment_encoder import EnvironmentTemporalGRU

PHASE3C_CHECKPOINT_PATH = "data/interim/ml/checkpoints/best_center_localization_cnn.pt"

class MultimodalTrackModel(nn.Module):
    def __init__(
        self,
        phase3c_checkpoint=PHASE3C_CHECKPOINT_PATH,
        freeze_satellite_backbone=True,
        sat_feat_dim=64,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        fusion_dim=128,
        dropout=0.1
    ):
        super().__init__()
        
        # ---------------------------------------------------------------------
        # 1. SATELLITE SPATIAL BACKBONE (Phase 3C Transfer)
        # ---------------------------------------------------------------------
        base_resnet = resnet18(weights=None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4

        # Load weights from Phase 3C if available
        if phase3c_checkpoint and os.path.exists(phase3c_checkpoint):
            ckpt = torch.load(phase3c_checkpoint, map_location='cpu', weights_only=False)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            
            backbone_state = {}
            for k, v in state_dict.items():
                if any(k.startswith(p) for p in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                    backbone_state[k] = v
            self.load_state_dict(backbone_state, strict=False)
            print(f"[MultimodalTrackModel] Loaded Phase 3C spatial backbone from {phase3c_checkpoint}")

        if freeze_satellite_backbone:
            for p in self.satellite_backbone_parameters():
                p.requires_grad = False
            print("[MultimodalTrackModel] Frozen satellite spatial backbone parameters.")

        self.sat_adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.sat_proj = nn.Sequential(
            nn.Linear(512 * 2 * 2, sat_feat_dim),
            nn.LayerNorm(sat_feat_dim),
            nn.GELU()
        )
        
        self.sat_gru = nn.GRU(
            input_size=sat_feat_dim,
            hidden_size=sat_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # ---------------------------------------------------------------------
        # 2. ENVIRONMENTAL BRANCH (ERA5 Atmospheric Wind Sequence)
        # ---------------------------------------------------------------------
        self.env_gru = EnvironmentTemporalGRU(
            in_channels=env_channels,
            spatial_dim=env_spatial_dim,
            hidden_dim=env_hidden_dim,
            num_layers=2,
            dropout=dropout
        )

        # ---------------------------------------------------------------------
        # 3. MULTIMODAL FUSION & TRACK DECODER
        # ---------------------------------------------------------------------
        multimodal_dim = sat_hidden_dim + env_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(multimodal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head_12 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.head_24 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.head_48 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def satellite_backbone_parameters(self):
        for m in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                yield p

    def forward_satellite(self, sat_seq):
        # sat_seq: [B, T=6, 1, 572, 929]
        B, T, C, H, W = sat_seq.shape
        flat_x = sat_seq.view(B * T, C, H, W)
        
        x = self.conv1(flat_x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) # [B*T, 512, 18, 30]
        
        x = self.sat_adaptive_pool(x) # [B*T, 512, 2, 2]
        flat_feats = torch.flatten(x, 1) # [B*T, 2048]
        proj_feats = self.sat_proj(flat_feats) # [B*T, sat_feat_dim]
        
        seq_feats = proj_feats.view(B, T, -1)
        _, h_n = self.sat_gru(seq_feats)
        h_sat = h_n[-1] # [B, sat_hidden_dim]
        return h_sat

    def forward(self, sat_seq, env_seq):
        # sat_seq: [B, 6, 1, 572, 929]
        # env_seq: [B, 6, 8, 41, 66]
        h_sat = self.forward_satellite(sat_seq) # [B, 128]
        h_env, _ = self.env_gru(env_seq) # [B, 64]
        
        # Late fusion
        fused = torch.cat([h_sat, h_env], dim=-1) # [B, 192]
        h_fused = self.fusion(fused) # [B, fusion_dim]
        
        pred_12 = self.head_12(h_fused)
        pred_24 = self.head_24(h_fused)
        pred_48 = self.head_48(h_fused)
        
        return {
            'pred_12': pred_12,
            'pred_24': pred_24,
            'pred_48': pred_48,
            'h_sat': h_sat,
            'h_env': h_env,
            'h_fused': h_fused
        }
