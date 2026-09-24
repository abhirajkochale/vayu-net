"""VAYU-NET — Production Model Inference & Checkpoint Loading Service
==================================================================
SIH 2026 Problem Statement 26070
Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)

Provides:
- CPU-safe lazy loading and caching of PyTorch checkpoints.
- Environment-configurable paths (no hardcoded absolute machine paths).
- Cryptographic SHA-256 verification when configured.
- Clean execution of:
    * Phase 5B Variant A (Observed Center Anchor)
    * Phase 5B Variant B (Satellite-Derived Anchor)
    * Phase 6 Multi-Task Intensity & Wind Speed Model
- Explicit provenance tagging: prediction_source = "MODEL_INFERENCE".
- Strict separation between prediction inputs and verification targets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ml.models.center_localization_cnn import (
    DedicatedCenterLocalizationResNet,
    LAT_MIN,
    LAT_SPAN,
    LON_MIN,
    LON_SPAN,
)
from ml.models.phase5b_hybrid_residual import Phase5BHybridResidualModel
from ml.models.phase6_intensity_wind import (
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    Phase6IntensityWindModel,
)

logger = logging.getLogger("vayu.inference.model_service")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default relative paths anchored to PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ModelService:
    """Production runtime model loading and inference service."""

    # Expected SHA-256 checksums from repository audit
    EXPECTED_SHA256 = {
        "best_center_localization_cnn.pt": "5ef11b596ef94ece71079c1329eab1e50191127eb7b9b1ff13f7dcdd3185f1c9",
        "best_phase5b_variant_a.pt": "c1a51fb9d2e55a6043c4b903a4b4696d4c98cb9034082e4825fb6682fefd9216",
        "best_phase5b_variant_b.pt": "2f0ec1e1e136c552a1ab4a4a8002e1af3293489d5fb649d95b30c2ca5e4a9533",
        "best_phase6_intensity_wind.pt": "a014e06dbf6fc17f37d04bb545a58746e57be6cb309808735660989009a460ef",
    }

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        feature_cache_path: Optional[str] = None,
        sample_index_path: Optional[str] = None,
        norm_stats_path: Optional[str] = None,
        device: Optional[str] = None,
        verify_checksums: Optional[bool] = None,
    ):
        base_ckpt = os.getenv("VAYU_CHECKPOINT_DIR", "data/interim/ml/checkpoints")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else (PROJECT_ROOT / base_ckpt)

        base_cache = os.getenv("VAYU_FEATURE_CACHE", "data/interim/ml/cache/phase5b_hybrid_features.pt")
        self.feature_cache_path = Path(feature_cache_path) if feature_cache_path else (PROJECT_ROOT / base_cache)

        base_sample_idx = os.getenv("VAYU_SAMPLE_INDEX", "data/manifests/vayu_net_sample_index.csv")
        self.sample_index_path = Path(sample_index_path) if sample_index_path else (PROJECT_ROOT / base_sample_idx)

        base_norm_stats = os.getenv("VAYU_NORM_STATS", "data/interim/ml/train_normalization_stats.json")
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else (PROJECT_ROOT / base_norm_stats)

        configured_device = device or os.getenv("VAYU_DEVICE", "cpu")
        self.device = torch.device(configured_device if torch.cuda.is_available() or configured_device == "cpu" else "cpu")

        if verify_checksums is None:
            self.verify_checksums = os.getenv("VAYU_VERIFY_CHECKSUMS", "0").lower() in ("1", "true", "yes")
        else:
            self.verify_checksums = verify_checksums

        # Lazy-loaded model instances and metadata
        self._model_center: Optional[DedicatedCenterLocalizationResNet] = None
        self._model_5b_a: Optional[Phase5BHybridResidualModel] = None
        self._model_5b_b: Optional[Phase5BHybridResidualModel] = None
        self._model_p6: Optional[Phase6IntensityWindModel] = None
        self._feature_lookup: Optional[Dict[str, Dict[str, Any]]] = None
        self._sample_index_df: Optional[pd.DataFrame] = None
        self._norm_stats: Optional[Dict[str, Any]] = None

        logger.info(
            f"Initialized ModelService (device={self.device}, ckpt_dir={self.checkpoint_dir}, verify_checksums={self.verify_checksums})"
        )

    @staticmethod
    def compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def _verify_file_checksum(self, filepath: Path, filename: str) -> None:
        if not self.verify_checksums:
            return
        expected = self.EXPECTED_SHA256.get(filename)
        if not expected:
            return
        actual = self.compute_sha256(filepath)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Checksum mismatch for {filename}! Expected {expected}, got {actual}. Checkpoint may be corrupted."
            )
        logger.info(f"Verified SHA-256 integrity for {filename}: {actual[:12]}...")

    def get_center_model(self) -> DedicatedCenterLocalizationResNet:
        """Returns loaded and evaluated Phase 3C center localization model instance."""
        if self._model_center is None:
            filename = "best_center_localization_cnn.pt"
            ckpt_path = self.checkpoint_dir / filename
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Phase 3C center localization checkpoint not found at {ckpt_path}. Verify runtime artifacts."
                )
            self._verify_file_checksum(ckpt_path, filename)

            logger.info(f"Loading Phase 3C Center Localization Model from {ckpt_path} on {self.device}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model = DedicatedCenterLocalizationResNet().to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model_center = model
        return self._model_center

    def get_phase5b_model(self, variant: str = "A") -> Phase5BHybridResidualModel:
        """Returns loaded and evaluated Phase 5B model instance."""
        v = variant.upper()
        if v not in ("A", "B"):
            raise ValueError(f"Unknown Phase 5B variant '{variant}'. Expected 'A' or 'B'.")

        if v == "A":
            if self._model_5b_a is None:
                filename = "best_phase5b_variant_a.pt"
                ckpt_path = self.checkpoint_dir / filename
                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Phase 5B Variant A checkpoint not found at {ckpt_path}. Verify runtime artifacts."
                    )
                self._verify_file_checksum(ckpt_path, filename)

                logger.info(f"Loading Phase 5B Variant A from {ckpt_path} on {self.device}")
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = Phase5BHybridResidualModel().to(self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self._model_5b_a = model
            return self._model_5b_a

        else:
            if self._model_5b_b is None:
                filename = "best_phase5b_variant_b.pt"
                ckpt_path = self.checkpoint_dir / filename
                if not ckpt_path.exists():
                    raise FileNotFoundError(
                        f"Phase 5B Variant B checkpoint not found at {ckpt_path}. Verify runtime artifacts."
                    )
                self._verify_file_checksum(ckpt_path, filename)

                logger.info(f"Loading Phase 5B Variant B from {ckpt_path} on {self.device}")
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = Phase5BHybridResidualModel().to(self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self._model_5b_b = model
            return self._model_5b_b

    def get_phase6_model(self) -> Phase6IntensityWindModel:
        """Returns loaded and evaluated Phase 6 multi-task model instance."""
        if self._model_p6 is None:
            filename = "best_phase6_intensity_wind.pt"
            ckpt_path = self.checkpoint_dir / filename
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Phase 6 checkpoint not found at {ckpt_path}. Verify runtime artifacts."
                )
            self._verify_file_checksum(ckpt_path, filename)

            logger.info(f"Loading Phase 6 Multi-Task Model from {ckpt_path} on {self.device}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model = Phase6IntensityWindModel(mode="multimodal_era5").to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model_p6 = model
        return self._model_p6

    def _ensure_features_loaded(self) -> None:
        """Loads and indexes the feature cache once in memory."""
        if self._feature_lookup is not None:
            return
        if not self.feature_cache_path.exists():
            raise FileNotFoundError(
                f"Feature cache not found at {self.feature_cache_path}. Runtime model inference requires cached feature sequence."
            )
        logger.info(f"Loading feature cache from {self.feature_cache_path}...")
        cache_data = torch.load(self.feature_cache_path, map_location="cpu", weights_only=False)
        self._feature_lookup = {s["sample_id"]: s for s in cache_data.get("samples", [])}
        logger.info(f"Indexed {len(self._feature_lookup)} samples from feature cache.")

    def get_sample_features(self, sample_id: str) -> Dict[str, Any]:
        """Retrieves raw input tensors for a sample without exposing future targets."""
        self._ensure_features_loaded()
        assert self._feature_lookup is not None
        if sample_id not in self._feature_lookup:
            raise KeyError(
                f"Features for sample '{sample_id}' not found in runtime feature store. Available samples: {len(self._feature_lookup)}."
            )
        return self._feature_lookup[sample_id]

    def _get_sample_row(self, sample_id: str) -> pd.Series:
        """Looks up the sample manifest entry from sample index."""
        if self._sample_index_df is None:
            if not self.sample_index_path.exists():
                raise FileNotFoundError(f"Sample index not found at {self.sample_index_path}.")
            self._sample_index_df = pd.read_csv(self.sample_index_path)
        matches = self._sample_index_df[self._sample_index_df["sample_id"] == sample_id]
        if matches.empty:
            raise KeyError(f"Sample '{sample_id}' not found in sample index {self.sample_index_path}.")
        return matches.iloc[0]

    def load_t0_satellite_frame(self, sample_id: str) -> torch.Tensor:
        """
        Loads, sanitizes, and normalizes the raw t0 satellite infrared frame [1, 1, 572, 929].

        Input Provenance:
          - Source: Frame path specified by 'frame_t0' in sample index (GridSat-B1 IRWIN CDR).
          - Preprocessing: Missing/NaN pixel imputation with mean_kelvin, standardized by (arr - mean_k) / std_k.
          - Zero ground-truth usage: Does NOT read imd_lat_t0, imd_lon_t0, or any future truth.
        """
        row = self._get_sample_row(sample_id)
        frame_rel = row["frame_t0"]
        frame_path = PROJECT_ROOT / frame_rel if not Path(frame_rel).is_absolute() else Path(frame_rel)
        if not frame_path.exists():
            raise FileNotFoundError(f"Raw t0 satellite frame not found at {frame_path} for sample '{sample_id}'.")

        if self._norm_stats is None:
            if not self.norm_stats_path.exists():
                raise FileNotFoundError(f"Normalization stats not found at {self.norm_stats_path}.")
            with open(self.norm_stats_path, "r") as f:
                self._norm_stats = json.load(f)

        mean_k = float(self._norm_stats["mean_kelvin"])
        std_k = float(self._norm_stats["std_kelvin"])

        with np.load(frame_path) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)

        # Handle invalid / missing pixels identically to Phase 3C training
        invalid_mask = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(invalid_mask):
            arr = arr.copy()
            arr[invalid_mask] = mean_k

        # Normalize with train-split stats
        arr = (arr - mean_k) / std_k
        # Add batch and channel dimensions: [1, 1, 572, 929]
        return torch.from_numpy(arr[np.newaxis, np.newaxis, :, :]).float()

    def predict_center(
        self,
        sample_id: str,
        t0_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Executes live PyTorch forward pass for Phase 3C cyclone center localization (F03).

        Input: Single-frame infrared satellite observation [1, 1, 572, 929].
        Architecture: DedicatedCenterLocalizationResNet (best_center_localization_cnn.pt).
        Output: AI-detected center coordinates (latitude, longitude) derived via 2D soft-argmax.
        Zero Leakage: Strictly does NOT read imd_lat_t0, imd_lon_t0, or any future ground-truth targets.
        """
        model = self.get_center_model()
        if t0_tensor is None:
            t0_tensor = self.load_t0_satellite_frame(sample_id)

        if t0_tensor.dim() == 2:
            t0_tensor = t0_tensor.unsqueeze(0).unsqueeze(0)
        elif t0_tensor.dim() == 3:
            t0_tensor = t0_tensor.unsqueeze(0)

        t0_tensor = t0_tensor.to(self.device).float()

        with torch.no_grad():
            out = model(t0_tensor)
            norm_center = out["norm_center"]  # [B, 2] in [0, 1]
            peak_center = out["peak_center"]  # [B, 2] in [0, 1]

            center_deg = DedicatedCenterLocalizationResNet.denormalize_center(norm_center.cpu())[0]
            peak_deg = DedicatedCenterLocalizationResNet.denormalize_center(peak_center.cpu())[0]

            lat = round(float(center_deg[0].item()), 4)
            lon = round(float(center_deg[1].item()), 4)
            lat_peak = round(float(peak_deg[0].item()), 4)
            lon_peak = round(float(peak_deg[1].item()), 4)

        return {
            "sample_id": sample_id,
            "center_detection_source": "MODEL_INFERENCE",
            "prediction_source": "MODEL_INFERENCE",
            "model_version": "phase3c-v1.0-center-localization",
            "checkpoint_identity": "best_center_localization_cnn.pt",
            "ai_detected_center": {
                "latitude": lat,
                "longitude": lon,
            },
            "discrete_peak_center": {
                "latitude": lat_peak,
                "longitude": lon_peak,
            },
        }

    def predict_track(self, sample_id: str, variant: str = "A") -> Dict[str, Any]:
        """
        Executes live PyTorch forward pass for multi-horizon track forecasting.
        Inputs: satellite sequence [1, 6, 132], ERA5 sequence [1, 6, 8, 41, 66], motion context [1, 5], kinematic anchors [1, 2].
        Outputs: predicted multi-horizon coordinates (+12h, +24h, +48h).
        Does NOT access or use any future ground-truth target coordinates.
        """
        model = self.get_phase5b_model(variant=variant)
        s = self.get_sample_features(sample_id)

        v = variant.upper()
        with torch.no_grad():
            sat = s["sat_seq"].unsqueeze(0).to(self.device)
            env = s["env_seq"].unsqueeze(0).to(self.device)
            if v == "A":
                mot = s["kin_ctx_a"].unsqueeze(0).to(self.device)
                k12 = s["kin_a_12"].unsqueeze(0).to(self.device)
                k24 = s["kin_a_24"].unsqueeze(0).to(self.device)
                k48 = s["kin_a_48"].unsqueeze(0).to(self.device)
            else:
                mot = s["kin_ctx_b"].unsqueeze(0).to(self.device)
                k12 = s["kin_b_12"].unsqueeze(0).to(self.device)
                k24 = s["kin_b_24"].unsqueeze(0).to(self.device)
                k48 = s["kin_b_48"].unsqueeze(0).to(self.device)

            out = model(sat, env, mot, k12, k24, k48)

            p12 = out["pred_12"][0].cpu().tolist()
            p24 = out["pred_24"][0].cpu().tolist()
            p48 = out["pred_48"][0].cpu().tolist()

            r12 = out["r12_km"][0].cpu().tolist()
            r24 = out["r24_km"][0].cpu().tolist()
            r48 = out["r48_km"][0].cpu().tolist()

        ckpt_file = "best_phase5b_variant_a.pt" if v == "A" else "best_phase5b_variant_b.pt"
        anchor_desc = "observed_center" if v == "A" else "satellite_derived_center"

        return {
            "sample_id": sample_id,
            "prediction_source": "MODEL_INFERENCE",
            "model_version": f"phase5b-v1.0-variant-{v.lower()}",
            "checkpoint_identity": ckpt_file,
            "anchor_type": anchor_desc,
            "predicted_coordinates": {
                "12h": {"lat": round(float(p12[0]), 4), "lon": round(float(p12[1]), 4)},
                "24h": {"lat": round(float(p24[0]), 4), "lon": round(float(p24[1]), 4)},
                "48h": {"lat": round(float(p48[0]), 4), "lon": round(float(p48[1]), 4)},
            },
            "predicted_residuals_km": {
                "12h": {"north_km": round(float(r12[0]), 2), "east_km": round(float(r12[1]), 2)},
                "24h": {"north_km": round(float(r24[0]), 2), "east_km": round(float(r24[1]), 2)},
                "48h": {"north_km": round(float(r48[0]), 2), "east_km": round(float(r48[1]), 2)},
            },
        }

    def predict_intensity_and_wind(self, sample_id: str) -> Dict[str, Any]:
        """
        Executes live PyTorch forward pass for Phase 6 intensity classification and wind regression.
        Inputs: satellite sequence [1, 6, 132], ERA5 sequence [1, 6, 8, 41, 66].
        Outputs: predicted IMD intensity category, calibrated probabilities, maximum sustained wind in knots.
        """
        model = self.get_phase6_model()
        s = self.get_sample_features(sample_id)

        with torch.no_grad():
            sat = s["sat_seq"].unsqueeze(0).to(self.device)
            env = s["env_seq"].unsqueeze(0).to(self.device)
            out = model(sat_seq=sat, env_seq=env)

            pred_idx = int(out["category_pred"][0].item())
            pred_cat = IDX_TO_CATEGORY.get(pred_idx, "D")
            probs = out["category_probs"][0].cpu().tolist()
            pred_wind_kt = round(float(out["pred_wind_kt"][0].item()), 1)
            wind_p80_kt = round(float(out["wind_uncertainty_p80_kt"][0].item()), 1)
            pred_pres_hpa = round(float(out["pred_pressure_hpa"][0].item()), 1)

        prob_dict = {IDX_TO_CATEGORY[i]: round(float(probs[i]), 4) for i in range(len(probs))}

        return {
            "prediction_source": "MODEL_INFERENCE",
            "model_version": "phase6-v1.0-multimodal-era5",
            "checkpoint_identity": "best_phase6_intensity_wind.pt",
            "predicted_category": pred_cat,
            "predicted_category_confidence": round(float(probs[pred_idx]), 4),
            "category_probabilities": prob_dict,
            "predicted_wind_kt": pred_wind_kt,
            "wind_uncertainty_p80_kt": wind_p80_kt,
            "predicted_central_pressure_hpa": pred_pres_hpa,
            "scientific_disclaimer": (
                "Phase 6 multi-task intensity/wind neural network is an experimental research model "
                "(validation macro F1: 0.2006, accuracy: 25.10%, wind MAE: 25.44 kt). "
                "It is NOT a certified operational replacement for the IMD Advanced Dvorak Technique (ADT)."
            ),
        }


# Singleton service instance
_global_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    global _global_model_service
    if _global_model_service is None:
        _global_model_service = ModelService()
    return _global_model_service
