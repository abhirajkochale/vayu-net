"""
VAYU-NET PHASE 6 — MULTI-TASK CYCLONE INTENSITY & WIND PREDICTION ARCHITECTURE
=============================================================================
Problem Statement: SIH 2026 PS 26070
Module: Current Cyclone Intensity Classification + Maximum Sustained Wind Regression

Architecture Components:
1. Encoders:
   - EXP-1 (Single-Frame): Linear projection & MLP on t0 spatial features [132]
   - EXP-2 (Temporal Satellite): 2-layer temporal GRU on 6-frame satellite sequence [6, 132]
   - EXP-3 (Multimodal): Joint satellite sequence GRU [6, 132] + ERA5 environmental sequence CNN-GRU [6, 8, 41, 66]
2. Shared Latent Representation:
   - Dimension: 128 (LayerNorm + GELU + Dropout)
3. Multi-Task Heads:
   - Intensity Classification Head: 7-class IMD category logits (D, DD, CS, SCS, VSCS, ESCS, SuCS)
     with temperature scaling for probability calibration.
   - Wind Speed Regression Head: Continuous maximum sustained wind in knots (normalized Smooth L1 target).
   - Auxiliary Pressure Head (optional): Central pressure regression in hPa.
4. Empirical Uncertainty:
   - Validation residual-based error bounds for wind speed.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical 7-class IMD category mapping
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

# TRAIN-only default normalization statistics
DEFAULT_TRAIN_WIND_MEAN_KT = 36.88362
DEFAULT_TRAIN_WIND_STD_KT = 17.97539
DEFAULT_TRAIN_PRES_MEAN_HPA = 993.7256
DEFAULT_TRAIN_PRES_STD_HPA = 10.99026


class EnvironmentSpatialCNN(nn.Module):
    """
    Lightweight 2D CNN encoder for single-timestamp 8-channel ERA5 wind field.
    Input: [B, 8, 41, 66] -> Output: [B, embed_dim] (default 64)
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
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        flat = torch.flatten(x, 1)
        return self.proj(flat)


class Phase6IntensityWindModel(nn.Module):
    """
    Multi-Task Cyclone Intensity Classification and Wind Speed Regression Model.
    Supports single_frame (EXP-1), temporal_sat (EXP-2), and multimodal_era5 (EXP-3).
    """
    def __init__(self,
                 mode="temporal_sat",
                 sat_feat_dim=132,
                 latent_dim=128,
                 gru_num_layers=2,
                 dropout=0.1,
                 num_classes=7,
                 train_wind_mean=DEFAULT_TRAIN_WIND_MEAN_KT,
                 train_wind_std=DEFAULT_TRAIN_WIND_STD_KT,
                 train_pres_mean=DEFAULT_TRAIN_PRES_MEAN_HPA,
                 train_pres_std=DEFAULT_TRAIN_PRES_STD_HPA):
        super().__init__()

        if mode not in ["single_frame", "temporal_sat", "multimodal_era5"]:
            raise ValueError(f"Unknown mode: {mode}. Expected 'single_frame', 'temporal_sat', or 'multimodal_era5'")

        self.mode = mode
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Register normalization buffers
        self.register_buffer("train_wind_mean", torch.tensor(train_wind_mean, dtype=torch.float32))
        self.register_buffer("train_wind_std", torch.tensor(train_wind_std, dtype=torch.float32))
        self.register_buffer("train_pres_mean", torch.tensor(train_pres_mean, dtype=torch.float32))
        self.register_buffer("train_pres_std", torch.tensor(train_pres_std, dtype=torch.float32))

        # Temperature parameter for probability calibration (initialized to 1.0)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        # Empirical wind uncertainty bands (populated from validation residuals)
        self.register_buffer("wind_median_ae", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("wind_p80_ae", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("wind_p90_ae", torch.tensor(0.0, dtype=torch.float32))

        # 1. Encoders by Mode
        if self.mode == "single_frame":
            # Input: sat_t0 [B, 132]
            self.sat_encoder = nn.Sequential(
                nn.Linear(sat_feat_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU()
            )
        elif self.mode == "temporal_sat":
            # Input: sat_seq [B, 6, 132]
            self.sat_gru = nn.GRU(
                input_size=sat_feat_dim,
                hidden_size=latent_dim,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0.0
            )
            self.sat_norm = nn.LayerNorm(latent_dim)
        elif self.mode == "multimodal_era5":
            # Input: sat_seq [B, 6, 132] + env_seq [B, 6, 8, 41, 66]
            self.sat_gru = nn.GRU(
                input_size=sat_feat_dim,
                hidden_size=latent_dim,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0.0
            )
            self.env_spatial_cnn = EnvironmentSpatialCNN(in_channels=8, embed_dim=64)
            self.env_gru = nn.GRU(
                input_size=64,
                hidden_size=64,
                num_layers=gru_num_layers,
                batch_first=True,
                dropout=dropout if gru_num_layers > 1 else 0.0
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(latent_dim + 64, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # 2. Multi-Task Prediction Heads
        # Intensity Classification Head
        self.category_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        # Wind Speed Regression Head (predicts normalized wind z_wind)
        self.wind_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Auxiliary Pressure Head (optional, predicts normalized pressure z_pres)
        self.pressure_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def extract_latent(self, sat_seq=None, sat_t0=None, env_seq=None):
        """Extracts shared latent representation z from input modalities."""
        if self.mode == "single_frame":
            if sat_t0 is None:
                if sat_seq is not None:
                    sat_t0 = sat_seq[:, -1, :]  # Extract terminal t0 frame
                else:
                    raise ValueError("sat_t0 or sat_seq must be provided for single_frame mode")
            z = self.sat_encoder(sat_t0)
        elif self.mode == "temporal_sat":
            if sat_seq is None:
                raise ValueError("sat_seq must be provided for temporal_sat mode")
            gru_out, h_n = self.sat_gru(sat_seq)  # gru_out: [B, 6, latent_dim]
            z = self.sat_norm(gru_out[:, -1, :]) # Terminal step t0
        elif self.mode == "multimodal_era5":
            if sat_seq is None or env_seq is None:
                raise ValueError("Both sat_seq and env_seq must be provided for multimodal_era5 mode")
            # Satellite representation
            sat_out, _ = self.sat_gru(sat_seq)
            h_sat = sat_out[:, -1, :]  # [B, latent_dim]

            # Environmental representation
            B, S, C, H, W = env_seq.shape
            env_flat = env_seq.view(B * S, C, H, W)
            env_spatial_feats = self.env_spatial_cnn(env_flat).view(B, S, 64)
            env_out, _ = self.env_gru(env_spatial_feats)
            h_env = env_out[:, -1, :]  # [B, 64]

            # Fusion
            h_joint = torch.cat([h_sat, h_env], dim=-1)
            z = self.fusion_layer(h_joint)

        return z

    def forward(self, sat_seq=None, sat_t0=None, env_seq=None):
        """
        Forward pass producing multi-task predictions:
        Returns:
            dict containing:
                'latent': [B, latent_dim]
                'category_logits': [B, 7] (uncalibrated logits)
                'category_probs': [B, 7] (temperature-calibrated softmax probabilities)
                'category_pred': [B] (predicted class index 0..6)
                'norm_wind': [B] (normalized wind speed regression)
                'pred_wind_kt': [B] (physical wind speed in knots)
                'wind_uncertainty_p80_kt': [B] (empirical P80 error band in knots)
                'wind_uncertainty_p90_kt': [B] (empirical P90 error band in knots)
                'norm_pressure': [B] (normalized pressure regression)
                'pred_pressure_hpa': [B] (physical pressure in hPa)
        """
        z = self.extract_latent(sat_seq=sat_seq, sat_t0=sat_t0, env_seq=env_seq)

        # 1. Intensity Category Classification
        cat_logits = self.category_head(z)
        temp = torch.clamp(self.temperature, min=0.01)
        cat_probs = F.softmax(cat_logits / temp, dim=-1)
        cat_pred = torch.argmax(cat_probs, dim=-1)

        # 2. Wind Speed Regression
        norm_wind = self.wind_head(z).squeeze(-1) # [B]
        pred_wind_kt = norm_wind * self.train_wind_std + self.train_wind_mean

        # Empirical uncertainty bounds
        wind_unc_p80 = self.wind_p80_ae.expand_as(pred_wind_kt)
        wind_unc_p90 = self.wind_p90_ae.expand_as(pred_wind_kt)

        # 3. Auxiliary Pressure Regression
        norm_pres = self.pressure_head(z).squeeze(-1) # [B]
        pred_pres_hpa = norm_pres * self.train_pres_std + self.train_pres_mean

        return {
            "latent": z,
            "category_logits": cat_logits,
            "category_probs": cat_probs,
            "category_pred": cat_pred,
            "norm_wind": norm_wind,
            "pred_wind_kt": pred_wind_kt,
            "wind_uncertainty_p80_kt": wind_unc_p80,
            "wind_uncertainty_p90_kt": wind_unc_p90,
            "norm_pressure": norm_pres,
            "pred_pressure_hpa": pred_pres_hpa
        }

    def set_temperature(self, temp_val):
        """Sets temperature scaling parameter for calibration."""
        self.temperature.data.fill_(float(temp_val))

    def set_uncertainty_bounds(self, median_ae, p80_ae, p90_ae):
        """Sets empirical uncertainty quantiles computed from validation residuals."""
        self.wind_median_ae.data.fill_(float(median_ae))
        self.wind_p80_ae.data.fill_(float(p80_ae))
        self.wind_p90_ae.data.fill_(float(p90_ae))
