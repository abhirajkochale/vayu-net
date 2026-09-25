"""
VAYU-NET: EXP-M1 Multi-Task Loss Functions.
Implements masked multi-task loss covering:
  - Center localization (Smooth L1 on normalized coordinates)
  - Wind speed regression (Masked Smooth L1 on normalized wind)
  - Intensity classification (Masked Cross-Entropy on 7 IMD categories)
  - Track prediction (Masked Smooth L1 at +12h, +24h, +48h)
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.experiments.exp_m1.model import (
    LAT_MIN, LAT_SPAN, LON_MIN, LON_SPAN,
    DEFAULT_TRAIN_WIND_MEAN_KT, DEFAULT_TRAIN_WIND_STD_KT
)


def encode_coordinates(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> torch.Tensor:
    """Normalizes geographic [lat, lon] to [0, 1] relative to the NIO synoptic domain."""
    u_lat = (lat_deg - LAT_MIN) / LAT_SPAN
    u_lon = (lon_deg - LON_MIN) / LON_SPAN
    return torch.stack([u_lat, u_lon], dim=-1)


def normalize_wind(wind_kt: torch.Tensor,
                   mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                   std: float = DEFAULT_TRAIN_WIND_STD_KT) -> torch.Tensor:
    """Standardizes wind speed using TRAIN-only statistics."""
    return (wind_kt - mean) / std


class MaskedMultiTaskLoss(nn.Module):
    """
    Computes masked multi-task loss for cyclone identification, classification,
    wind intensity, and track forecasting.
    """
    def __init__(self,
                 lambda_center: float = 1.0,
                 lambda_wind: float = 1.0,
                 lambda_class: float = 1.0,
                 lambda_track: float = 1.0,
                 wind_mean: float = DEFAULT_TRAIN_WIND_MEAN_KT,
                 wind_std: float = DEFAULT_TRAIN_WIND_STD_KT):
        super().__init__()
        self.lambda_center = lambda_center
        self.lambda_wind = lambda_wind
        self.lambda_class = lambda_class
        self.lambda_track = lambda_track
        self.wind_mean = wind_mean
        self.wind_std = wind_std

    def forward(self,
                preds: Dict[str, torch.Tensor],
                batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates weighted masked multi-task loss.
        """
        device = preds["center_norm"].device

        # 1. Center Localization Loss at t0
        c0_deg = batch["center_t0"].to(device)  # [B, 2]
        c0_norm = encode_coordinates(c0_deg[:, 0], c0_deg[:, 1])
        loss_center = F.smooth_l1_loss(preds["center_norm"], c0_norm)

        # 2. Maximum Sustained Wind Loss at t0
        w0_kt = batch["wind_t0"].to(device)       # [B]
        w0_mask = batch["wind_t0_mask"].to(device) # [B]
        w0_norm = normalize_wind(w0_kt, self.wind_mean, self.wind_std).unsqueeze(-1)
        w_err = F.smooth_l1_loss(preds["wind_norm"], w0_norm, reduction="none").squeeze(-1)
        loss_wind = (w_err * w0_mask).sum() / (w0_mask.sum() + 1e-6)

        # 3. Intensity Classification Loss at t0
        cat0 = batch["category_t0"].to(device)        # [B]
        cat0_mask = batch["category_t0_mask"].to(device) # [B]
        valid_cat = cat0_mask > 0.5
        if valid_cat.any():
            loss_class = F.cross_entropy(preds["class_logits"][valid_cat], cat0[valid_cat])
        else:
            loss_class = torch.tensor(0.0, device=device)

        # 4. Multi-Horizon Track Prediction Losses (+12h, +24h, +48h)
        def compute_track_loss(pred_norm: torch.Tensor,
                               true_deg: torch.Tensor,
                               mask: torch.Tensor) -> torch.Tensor:
            true_norm = encode_coordinates(true_deg[:, 0], true_deg[:, 1])
            err = F.smooth_l1_loss(pred_norm, true_norm, reduction="none")  # [B, 2]
            masked_err = (err * mask.unsqueeze(-1)).sum()
            return masked_err / (mask.sum() * 2.0 + 1e-6)

        c12_deg = batch["center_12h"].to(device)
        m12 = batch["wind_12h_mask"].to(device)  # 1.0 if target present at +12h
        loss_12 = compute_track_loss(preds["track_12h_norm"], c12_deg, m12)

        c24_deg = batch["center_24h"].to(device)
        m24 = batch["wind_24h_mask"].to(device)
        loss_24 = compute_track_loss(preds["track_24h_norm"], c24_deg, m24)

        c48_deg = batch["center_48h"].to(device)
        m48 = batch["wind_48h_mask"].to(device)
        loss_48 = compute_track_loss(preds["track_48h_norm"], c48_deg, m48)

        loss_track = (loss_12 + loss_24 + loss_48) / 3.0

        # Total Weighted Multi-Task Loss
        total_loss = (self.lambda_center * loss_center +
                      self.lambda_wind * loss_wind +
                      self.lambda_class * loss_class +
                      self.lambda_track * loss_track)

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_center": float(loss_center.item()),
            "loss_wind": float(loss_wind.item()),
            "loss_class": float(loss_class.item()),
            "loss_12": float(loss_12.item()),
            "loss_24": float(loss_24.item()),
            "loss_48": float(loss_48.item()),
            "loss_track": float(loss_track.item())
        }

        return total_loss, loss_dict