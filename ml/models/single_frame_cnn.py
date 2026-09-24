"""
VAYU-NET — Single-Frame CNN Baseline Model (ResNet-18)
Ablation model processing exclusively the t0 satellite observation (1 channel).
From-scratch baseline (weights=None) with multi-task prediction heads:
1. Current track center (normalized lat/lon in [0, 1]^2 via sigmoid)
2. Current maximum sustained wind (normalized continuous regression)
3. Current cyclonic category (7-class discrete classification)
"""

import torch
import torch.nn as nn
import torchvision.models as models

# Locked NIO spatial bounds
LAT_MIN = -5.0
LAT_SPAN = 40.0   # lat in [-5, 35]
LON_MIN = 40.0
LON_SPAN = 65.0   # lon in [40, 105]

class SingleFrameResNet(nn.Module):
    """
    1-Channel ResNet-18 backbone with multi-task estimation heads for current cyclone state.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Instantiate ResNet-18 without pre-trained weights (from-scratch baseline)
        base_resnet = models.resnet18(weights=None)
        
        # Adapt first convolution to 1-channel Gridsat IR input (instead of 3-channel RGB)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        
        # ResNet residual stages
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension is 512
        feat_dim = 512
        
        # 1. Track Center Head (Sigmoid outputs normalized [0, 1]^2)
        self.center_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
        # 2. Maximum Sustained Wind Head (normalized regression)
        self.wind_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        
        # 3. Cyclonic Category Head (7 classification logits)
        self.category_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        """Passes 1-channel input through ResNet-18 backbone"""
        # x shape: [B, 1, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        return feat

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, 1, 572, 929]
        Returns:
            dict with:
                'norm_center': [B, 2] in [0, 1] (u_lat, u_lon)
                'norm_wind': [B] (normalized wind scalar)
                'category_logits': [B, 7]
        """
        feat = self.extract_features(x)
        
        norm_center = self.center_head(feat)
        norm_wind = self.wind_head(feat).squeeze(-1)
        category_logits = self.category_head(feat)
        
        return {
            "norm_center": norm_center,
            "norm_wind": norm_wind,
            "category_logits": category_logits
        }

    @staticmethod
    def denormalize_center(norm_center):
        """
        Converts normalized [0, 1] coordinates back to physical degrees.
        lat = -5.0 + 40.0 * u_lat
        lon = 40.0 + 65.0 * u_lon
        """
        u_lat = norm_center[:, 0]
        u_lon = norm_center[:, 1]
        lat_deg = LAT_MIN + u_lat * LAT_SPAN
        lon_deg = LON_MIN + u_lon * LON_SPAN
        return torch.stack([lat_deg, lon_deg], dim=-1)

    @staticmethod
    def normalize_center(center_deg):
        """
        Converts physical degrees to normalized [0, 1] coordinates.
        u_lat = (lat - (-5.0)) / 40.0
        u_lon = (lon - 40.0) / 65.0
        """
        lat = center_deg[:, 0]
        lon = center_deg[:, 1]
        u_lat = (lat - LAT_MIN) / LAT_SPAN
        u_lon = (lon - LON_MIN) / LON_SPAN
        return torch.stack([u_lat, u_lon], dim=-1)

    @staticmethod
    def denormalize_wind(norm_wind, mean_kt, std_kt):
        """Converts normalized wind back to knots"""
        return norm_wind * std_kt + mean_kt

    @staticmethod
    def normalize_wind(wind_kt, mean_kt, std_kt):
        """Converts knots to normalized wind using TRAIN statistics"""
        return (wind_kt - mean_kt) / std_kt
