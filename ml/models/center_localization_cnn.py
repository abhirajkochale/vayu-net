"""
VAYU-NET PHASE 3C — DEDICATED CYCLONE CENTER LOCALIZATION MODEL
===============================================================
Architecture: DedicatedCenterLocalizationResNet
Backbone: 1-channel ResNet-18 (from-scratch, weights=None)
Spatial Feature Tap: layer2 feature map [B, 128, 72, 117]
Coordinate Injection: Normalized CoordConv channels [B, 2, 72, 117] in [-1, 1]
Spatial Decoder: Fully convolutional heatmap decoder [B, 130, 72, 117] -> [B, 1, 72, 117]
Coordinate Decoding: Differentiable 2D spatial soft-argmax for continuous geographic expectation
Diagnostic Peak: Discrete 2D argmax for discrete peak tracking

Zero Global Average Pooling. Zero Spatial Bottlenecks. Isolated Center Localization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Locked NIO spatial bounds
LAT_MIN = -5.0
LAT_SPAN = 40.0   # lat in [-5, 35]
LON_MIN = 40.0
LON_SPAN = 65.0   # lon in [40, 105]

# Heatmap spatial dimensions (derived from layer2 of 572x929 input)
HEATMAP_H = 72
HEATMAP_W = 117


class DedicatedCenterLocalizationResNet(nn.Module):
    """
    Dedicated Single-Frame Cyclone Center Localization Model.
    Isolates spatial feature learning from competing intensity objectives.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.heatmap_h = HEATMAP_H
        self.heatmap_w = HEATMAP_W
        
        # 1. Backbone: ResNet-18 from-scratch (weights=None)
        base = models.resnet18(weights=None)
        
        # Adapt first convolution to 1-channel Gridsat IR input (instead of 3-channel RGB)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        # Residual stages
        self.layer1 = base.layer1  # [B, 64, 143, 233]
        self.layer2 = base.layer2  # [B, 128, 72, 117]
        self.layer3 = base.layer3  # [B, 256, 36, 59]
        self.layer4 = base.layer4  # [B, 512, 18, 30]
        
        # 2. Normalized Coordinate Channels Buffer in [-1, 1]
        # Row 0 is South (-1), Row 71 is North (+1)
        # Col 0 is West (-1), Col 116 is East (+1)
        grid_y = torch.linspace(-1.0, 1.0, HEATMAP_H).view(1, 1, HEATMAP_H, 1)
        grid_x = torch.linspace(-1.0, 1.0, HEATMAP_W).view(1, 1, 1, HEATMAP_W)
        coord_grid = torch.cat([
            grid_y.expand(1, 1, HEATMAP_H, HEATMAP_W),
            grid_x.expand(1, 1, HEATMAP_H, HEATMAP_W)
        ], dim=1) # [1, 2, H, W]
        self.register_buffer("coord_channels", coord_grid)
        
        # Normalized coordinate grids for soft-argmax in [0, 1]
        grid_norm_y = torch.linspace(0.0, 1.0, HEATMAP_H).view(1, 1, HEATMAP_H, 1)
        grid_norm_x = torch.linspace(0.0, 1.0, HEATMAP_W).view(1, 1, 1, HEATMAP_W)
        self.register_buffer("grid_norm_y", grid_norm_y)
        self.register_buffer("grid_norm_x", grid_norm_x)
        
        # 3. Center Heatmap Decoder Head
        # Input: layer2 features (128 channels) + 2 coordinate channels = 130 channels
        self.center_decoder = nn.Sequential(
            nn.Conv2d(130, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # Raw heatmap logits [B, 1, 72, 117]
        )

    def decode_soft_argmax(self, heatmap_logits):
        """
        Differentiable spatial expectation (soft-argmax) converting 2D logits
        into normalized continuous coordinates (u_lat, u_lon) in [0, 1]^2.
        """
        B, C, H, W = heatmap_logits.shape
        flat_logits = heatmap_logits.view(B, -1) / self.temperature
        weights = F.softmax(flat_logits, dim=-1).view(B, 1, H, W)
        
        # Expectation along spatial dimensions
        pred_u_lat = (weights * self.grid_norm_y).sum(dim=[2, 3]).squeeze(-1) # [B, 1] -> [B]
        pred_u_lon = (weights * self.grid_norm_x).sum(dim=[2, 3]).squeeze(-1) # [B, 1] -> [B]
        
        norm_center = torch.stack([pred_u_lat, pred_u_lon], dim=-1) # [B, 2]
        return norm_center, weights

    def extract_discrete_peak(self, heatmap_logits):
        """Extracts discrete peak coordinate (argmax) for diagnostic debugging."""
        B, C, H, W = heatmap_logits.shape
        flat = heatmap_logits.view(B, -1)
        max_idx = torch.argmax(flat, dim=-1)
        peak_y = (max_idx // W).float() / (H - 1)
        peak_x = (max_idx % W).float() / (W - 1)
        return torch.stack([peak_y, peak_x], dim=-1) # [B, 2] in [0, 1]

    def forward(self, x):
        """
        Forward pass for dedicated center localization.
        Args:
            x: Tensor of shape [B, 1, 572, 929]
        Returns:
            dict containing:
                'heatmap_logits': [B, 1, 72, 117]
                'heatmap_prob':   [B, 1, 72, 117] (Sigmoid probabilities)
                'norm_center':    [B, 2] in [0, 1] (soft-argmax continuous coords)
                'peak_center':    [B, 2] in [0, 1] (discrete peak coords)
                'feature_shapes': dict of intermediate feature dimensions
        """
        B = x.size(0)
        
        # Backbone intermediate stages
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        mp = self.maxpool(x1)
        
        l1 = self.layer1(mp)     # [B, 64, 143, 233]
        l2 = self.layer2(l1)     # [B, 128, 72, 117]
        
        feature_shapes = {
            "input": list(x.shape),
            "conv1": list(x1.shape),
            "maxpool": list(mp.shape),
            "layer1": list(l1.shape),
            "layer2": list(l2.shape)
        }
        
        # Center Spatial Localization Branch
        coords = self.coord_channels.expand(B, -1, -1, -1) # [B, 2, 72, 117]
        feat_spatial = torch.cat([l2, coords], dim=1)      # [B, 130, 72, 117]
        heatmap_logits = self.center_decoder(feat_spatial) # [B, 1, 72, 117]
        heatmap_prob = torch.sigmoid(heatmap_logits)
        
        # Differentiable soft-argmax decoding
        norm_center, spatial_weights = self.decode_soft_argmax(heatmap_logits)
        peak_center = self.extract_discrete_peak(heatmap_logits)
        
        return {
            "heatmap_logits": heatmap_logits,
            "heatmap_prob": heatmap_prob,
            "norm_center": norm_center,
            "peak_center": peak_center,
            "spatial_weights": spatial_weights,
            "feature_shapes": feature_shapes
        }

    @staticmethod
    def generate_gaussian_target(target_norm_center, height=HEATMAP_H, width=HEATMAP_W, sigma=1.5, device=None):
        """
        Generates continuous 2D Gaussian heatmaps centered at target_norm_center.
        Args:
            target_norm_center: Tensor of shape [B, 2] with (u_lat, u_lon) in [0, 1]
            height, width: heatmap grid resolution (72, 117)
            sigma: Gaussian standard deviation in grid cells (default: 1.5 cells ~ 93 km)
        Returns:
            Tensor of shape [B, 1, height, width] with values in [0, 1]
        """
        B = target_norm_center.size(0)
        if device is None:
            device = target_norm_center.device
            
        u_lat = target_norm_center[:, 0].view(B, 1, 1, 1)
        u_lon = target_norm_center[:, 1].view(B, 1, 1, 1)
        
        y_star = u_lat * (height - 1)
        x_star = u_lon * (width - 1)
        
        grid_y = torch.arange(height, dtype=torch.float32, device=device).view(1, 1, height, 1)
        grid_x = torch.arange(width, dtype=torch.float32, device=device).view(1, 1, 1, width)
        
        dist_sq = (grid_y - y_star)**2 + (grid_x - x_star)**2
        target_heatmap = torch.exp(-dist_sq / (2.0 * sigma**2))
        return target_heatmap

    @staticmethod
    def denormalize_center(norm_center):
        """Converts normalized [0, 1] coordinates back to physical degrees."""
        u_lat = norm_center[:, 0]
        u_lon = norm_center[:, 1]
        lat_deg = LAT_MIN + u_lat * LAT_SPAN
        lon_deg = LON_MIN + u_lon * LON_SPAN
        return torch.stack([lat_deg, lon_deg], dim=-1)

    @staticmethod
    def normalize_center(center_deg):
        """Converts physical degrees to normalized [0, 1] coordinates."""
        lat = center_deg[:, 0]
        lon = center_deg[:, 1]
        u_lat = (lat - LAT_MIN) / LAT_SPAN
        u_lon = (lon - LON_MIN) / LON_SPAN
        return torch.stack([u_lat, u_lon], dim=-1)
