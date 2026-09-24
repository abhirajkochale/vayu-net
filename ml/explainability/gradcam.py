"""VAYU-NET WP-08: Grad-CAM Saliency Explainability Module.

Provides model interpretability for IMD intensity classification by computing
gradient-weighted class activation mapping (Grad-CAM) over spatial convolutional
feature representations of the t0 satellite observation.

IMPORTANT INTERPRETATION NOTICE:
The saliency/Grad-CAM output is an interpretation aid showing regions that
contributed most strongly to the model's intensity classification.
It must NOT be described or presented as a causal meteorological explanation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.phase6_intensity_wind import (
    Phase6IntensityWindModel,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
)
from ml.models.temporal_track_gru import TemporalTrackGRU


class GradCAMExplainer:
    """Computes Grad-CAM saliency heatmaps for VAYU-NET Phase 6 intensity classification.
    
    Hooks into the spatial convolutional feature map of the frozen Phase 3C ResNet-18
    backbone (layer2: [B, 128, 72, 117]) which provides the spatial visual tokens
    for the sequence model at t0.
    """

    def __init__(
        self,
        spatial_checkpoint_path: Union[str, Path] = "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
        temporal_checkpoint_path: Union[str, Path] = "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
        phase6_checkpoint_path: Union[str, Path] = "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt",
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.spatial_checkpoint_path = Path(spatial_checkpoint_path)
        self.temporal_checkpoint_path = Path(temporal_checkpoint_path)
        self.phase6_checkpoint_path = Path(phase6_checkpoint_path)

        # 1. Initialize Spatial Feature Extractor (TemporalTrackGRU components)
        self.spatial_extractor = TemporalTrackGRU(
            spatial_checkpoint_path=str(self.spatial_checkpoint_path),
            freeze_spatial_encoder=False,
            use_center_features=True,
            spatial_emb_dim=128,
            gru_hidden_dim=128,
            gru_num_layers=2,
            dropout=0.1,
        ).to(self.device)

        if self.temporal_checkpoint_path.exists():
            gru_ckpt = torch.load(self.temporal_checkpoint_path, map_location=self.device)
            spatial_state = {
                k: v
                for k, v in gru_ckpt.get("model_state_dict", {}).items()
                if k.startswith("spatial_")
            }
            if spatial_state:
                self.spatial_extractor.load_state_dict(spatial_state, strict=False)

        self.spatial_extractor.eval()

        # 2. Load Phase 6 Multi-Task Model
        p6_ckpt = torch.load(self.phase6_checkpoint_path, map_location=self.device)
        self.p6_config = p6_ckpt.get("config", {})
        self.intensity_classes = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]

        self.p6_model = Phase6IntensityWindModel(
            mode=p6_ckpt.get("mode", "multimodal_era5"),
            sat_feat_dim=132,
            latent_dim=self.p6_config.get("latent_dim", 128),
            gru_num_layers=self.p6_config.get("gru_num_layers", 2),
            dropout=self.p6_config.get("dropout", 0.1),
            num_classes=len(self.intensity_classes),
        ).to(self.device)
        self.p6_model.load_state_dict(p6_ckpt["model_state_dict"])
        self.p6_model.eval()

        # Target layer for Grad-CAM: ResNet-18 layer2 [B, 128, 72, 117]
        self.target_layer = self.spatial_extractor.spatial_encoder.layer2
        self.target_layer_name = "spatial_encoder.layer2"

        # Hooks
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target convolutional layer."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h_fwd, h_bwd])

    def encode_t0_frame(
        self,
        t0_img_tensor: torch.Tensor,
        prev_center_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Differentiably extracts the 132-dim visual token from the raw t0 satellite image.
        
        Args:
            t0_img_tensor: [1, 1, H, W] normalized satellite image at t0.
            prev_center_norm: Optional [1, 2] center at t_{-3h} for delta calculation.
            
        Returns:
            token_132: [1, 132] feature embedding for the t0 frame.
        """
        self.spatial_extractor.eval()
        emb_t0, center_t0 = self.spatial_extractor.extract_single_frame_feature(t0_img_tensor)

        if prev_center_norm is not None:
            delta_t0 = center_t0 - prev_center_norm.to(self.device)
        else:
            delta_t0 = torch.zeros_like(center_t0)

        token_132 = torch.cat([emb_t0, center_t0, delta_t0], dim=-1)
        return token_132

    def generate_saliency(
        self,
        t0_img_tensor: torch.Tensor,
        cached_seq_132: torch.Tensor,
        env_seq_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generates Grad-CAM saliency for the t0 satellite observation.
        
        Args:
            t0_img_tensor: [1, 1, H, W] or [H, W] raw normalized satellite frame at t0.
            cached_seq_132: [1, 6, 132] 6-frame satellite feature sequence.
            env_seq_tensor: [1, 6, 8, 41, 66] 6-step ERA5 environmental sequence.
            target_class_idx: Optional class index to explain (defaults to argmax predicted class).
            
        Returns:
            Dict containing:
                - cam_raw: 2D numpy array [72, 117]
                - cam_resized: 2D numpy array [H, W] normalized in [0, 1]
                - predicted_class: string label
                - predicted_class_idx: int
                - confidence: float
                - max_activation_loc: Tuple[int, int] (row, col)
                - activation_centroid: Tuple[float, float] (row, col)
                - logits: numpy array
                - probs: numpy array
                - target_layer: string name of target layer
        """
        self.spatial_extractor.eval()
        self.p6_model.eval()

        if t0_img_tensor.dim() == 2:
            t0_img_tensor = t0_img_tensor.unsqueeze(0).unsqueeze(0)
        elif t0_img_tensor.dim() == 3:
            t0_img_tensor = t0_img_tensor.unsqueeze(0)

        t0_img_tensor = t0_img_tensor.to(self.device).float()
        cached_seq_132 = cached_seq_132.to(self.device).float()
        env_seq_tensor = env_seq_tensor.to(self.device).float()

        # Zero gradients & reset hook buffers
        self.spatial_extractor.zero_grad()
        self.p6_model.zero_grad()
        self.gradients = None
        self.activations = None

        # Extract prev_center from t_{-3h} (step index 4) if available
        prev_center = cached_seq_132[:, 4, 128:130] if cached_seq_132.size(1) >= 5 else None

        # Compute t0 token differentiably through spatial_encoder
        token_t0 = self.encode_t0_frame(t0_img_tensor, prev_center_norm=prev_center)

        # Substitute t0 slot (index 5)
        seq_assembled = cached_seq_132.clone()
        seq_assembled[:, 5, :] = token_t0

        # Forward pass through Phase 6 multi-task model
        out = self.p6_model(sat_seq=seq_assembled, env_seq=env_seq_tensor)
        class_logits = out["category_logits"]
        probs = out["category_probs"]

        pred_idx = int(torch.argmax(class_logits, dim=-1).item())
        confidence = float(probs[0, pred_idx].item())

        if target_class_idx is None:
            target_class_idx = pred_idx

        # Backward pass on the target class score
        score = class_logits[0, target_class_idx]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Compute channel-wise weights alpha_k = Global Average Pooling of gradients
        # activations: [1, 128, 72, 117], gradients: [1, 128, 72, 117]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, 128, 1, 1]
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, 72, 117]
        cam = F.relu(cam)  # Apply ReLU to isolate positive contributions to target class

        # Upsample to original image resolution [H, W] (572 x 929)
        orig_h, orig_w = t0_img_tensor.shape[2], t0_img_tensor.shape[3]
        cam_upsampled = F.interpolate(
            cam, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

        cam_raw_np = cam.squeeze().detach().cpu().numpy()
        cam_resized_np = cam_upsampled.squeeze().detach().cpu().numpy()

        # Min-Max Normalization to [0, 1]
        c_min = float(np.min(cam_resized_np))
        c_max = float(np.max(cam_resized_np))
        if c_max - c_min > 1e-8:
            cam_norm = (cam_resized_np - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_resized_np)

        # Max activation location and activation centroid
        max_idx = np.unravel_index(np.argmax(cam_norm), cam_norm.shape)
        total_mass = float(np.sum(cam_norm))
        if total_mass > 1e-8:
            y_indices, x_indices = np.indices(cam_norm.shape)
            centroid_y = float(np.sum(y_indices * cam_norm) / total_mass)
            centroid_x = float(np.sum(x_indices * cam_norm) / total_mass)
        else:
            centroid_y = float(max_idx[0])
            centroid_x = float(max_idx[1])

        target_class_name = (
            self.intensity_classes[target_class_idx]
            if target_class_idx < len(self.intensity_classes)
            else f"Class_{target_class_idx}"
        )

        return {
            "cam_raw": cam_raw_np,
            "cam_resized": cam_norm,
            "predicted_class": target_class_name,
            "predicted_class_idx": target_class_idx,
            "confidence": confidence,
            "max_activation_loc": (int(max_idx[0]), int(max_idx[1])),
            "activation_centroid": (centroid_y, centroid_x),
            "logits": class_logits.detach().cpu().numpy()[0],
            "probs": probs.detach().cpu().numpy()[0],
            "target_layer": self.target_layer_name,
        }

    @staticmethod
    def render_overlay(
        original_img: np.ndarray,
        heatmap_norm: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "turbo",
    ) -> np.ndarray:
        """Renders an RGB overlay of the Grad-CAM heatmap over the normalized satellite image.
        
        Args:
            original_img: 2D numpy array [H, W] normalized in [0, 1].
            heatmap_norm: 2D numpy array [H, W] normalized in [0, 1].
            alpha: Heatmap blend weight (0 to 1).
            colormap_name: Matplotlib colormap (default: 'turbo').
            
        Returns:
            overlay_rgb: 3D numpy array [H, W, 3] uint8 (0-255).
        """
        cmap = plt.get_cmap(colormap_name)
        heatmap_rgb = cmap(heatmap_norm)[:, :, :3]  # [H, W, 3] float in [0, 1]

        # Convert grayscale satellite image to RGB
        orig_clipped = np.clip(original_img, 0.0, 1.0)
        orig_rgb = np.stack([orig_clipped, orig_clipped, orig_clipped], axis=-1)

        # Blend
        blended = (1.0 - alpha) * orig_rgb + alpha * heatmap_rgb
        blended = np.clip(blended, 0.0, 1.0)
        overlay_uint8 = (blended * 255.0).astype(np.uint8)
        return overlay_uint8

    def close(self) -> None:
        """Remove PyTorch hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
