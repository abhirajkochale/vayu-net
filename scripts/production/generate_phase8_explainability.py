"""VAYU-NET WP-08 — PRODUCTION EXPLAINABILITY GENERATION PIPELINE
==============================================================
Generates reproducible Grad-CAM saliency heatmaps, original satellite frames,
and multi-layer overlays for the locked reference cyclone shortlist:
  - FANI 2019 (Bay of Bengal, Extremely Severe Cyclonic Storm)
  - AMPHAN 2020 (Bay of Bengal, Super Cyclonic Storm)
  - TAUKTAE 2021 (Arabian Sea, Extremely Severe Cyclonic Storm)
  - BIPARJOY 2023 (Arabian Sea, Extremely Severe Cyclonic Storm)
  - REMAL 2024 (Bay of Bengal, Severe Cyclonic Storm)

Outputs saved to:
  data/interim/ml/explainability/
    - original_{storm}_{t0}.png
    - heatmap_{storm}_{t0}.png
    - overlay_{storm}_{t0}.png
    - metadata_{storm}_{t0}.json
    - explainability_manifest.json

IMPORTANT SCIENTIFIC INTERPRETATION NOTICE:
The saliency/Grad-CAM output is an interpretation aid showing spatial regions
that contributed most strongly to the model's IMD intensity classification.
It must NOT be described or presented as a causal meteorological explanation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.explainability.gradcam import GradCAMExplainer

SAMPLE_INDEX_PATH = PROJECT_ROOT / "data/manifests/vayu_net_sample_index.csv"
NORM_STATS_PATH = PROJECT_ROOT / "data/interim/ml/train_normalization_stats.json"
CACHE_PATH = PROJECT_ROOT / "data/interim/ml/cache/phase6_intensity_features.pt"
OUTPUT_DIR = PROJECT_ROOT / "data/interim/ml/explainability"

REFERENCE_STORMS = ["FANI", "AMPHAN", "TAUKTAE", "BIPARJOY", "REMAL"]


def load_raw_satellite_frame(
    npz_path: Path, mean_kelvin: float, std_kelvin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Loads raw GridSat frame, applies invalid pixel masking, and returns both

    visual grayscale [0, 1] (cold clouds bright) and model standardized tensor [572, 929].
    """
    with np.load(npz_path) as npz:
        arr = npz["irwin_cdr"].astype(np.float32)

    # Clean invalid pixels
    invalid_mask = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
    clean_arr = arr.copy()
    if np.any(invalid_mask):
        clean_arr[invalid_mask] = mean_kelvin

    # Model input standardization: z = (x - mean) / std
    model_norm = (clean_arr - mean_kelvin) / std_kelvin

    # Visual representation: Inverted IR (cold convective tops = bright white, warm ocean = dark)
    # Range 190 K (cold high clouds) to 305 K (sea surface)
    vis_norm = np.clip((305.0 - clean_arr) / (305.0 - 190.0), 0.0, 1.0)

    return vis_norm, model_norm


def run_explainability_pipeline() -> List[Dict[str, Any]]:
    """Runs Grad-CAM explainability pipeline for all reference shortlist storms."""
    print("=" * 65)
    print("VAYU-NET WP-08 — PRODUCTION EXPLAINABILITY GENERATION PIPELINE")
    print("=" * 65)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load normalization stats
    with open(NORM_STATS_PATH, "r") as f:
        norm_stats = json.load(f)
    mean_kelvin = float(norm_stats["mean_kelvin"])
    std_kelvin = float(norm_stats["std_kelvin"])
    print(f"Loaded TRAIN normalization: mean={mean_kelvin:.2f} K, std={std_kelvin:.2f} K")

    # 2. Load manifests & feature cache
    sample_df = pd.read_csv(SAMPLE_INDEX_PATH)
    print(f"Loaded sample index: {len(sample_df)} total records")

    print(f"Loading cached multimodal sequence features from: {CACHE_PATH} ...")
    cache = torch.load(CACHE_PATH, map_location="cpu")
    cached_samples = {s["sample_id"]: s for s in cache["samples"]}
    print(f"Loaded {len(cached_samples)} cached feature records.")

    # 3. Initialize GradCAMExplainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing GradCAMExplainer on device: {device} ...")
    explainer = GradCAMExplainer(device=device)

    manifest_records: List[Dict[str, Any]] = []

    for storm_name in REFERENCE_STORMS:
        print(f"\nProcessing Reference Cyclone: {storm_name} ...")
        sub_df = sample_df[sample_df["storm_id"].str.contains(storm_name, case=False, na=False)]
        if len(sub_df) == 0:
            print(f"  Warning: No samples found for {storm_name}!")
            continue

        # Select peak intensity sample
        peak_row = sub_df.sort_values("imd_wind_t0", ascending=False).iloc[0]
        sample_id = str(peak_row["sample_id"])
        storm_id = str(peak_row["storm_id"])
        t0_str = str(peak_row["t0"])
        frame_t0_path = PROJECT_ROOT / str(peak_row["frame_t0"])
        split = str(peak_row["split"])
        truth_cat = str(peak_row["imd_category_t0"])
        truth_wind = float(peak_row["imd_wind_t0"])

        t0_tag = t0_str.replace(":", "").replace("-", "")

        print(f"  Sample ID: {sample_id}")
        print(f"  Timestamp t0: {t0_str} (Split: {split})")
        print(f"  IMD Truth: Category={truth_cat}, Wind={truth_wind} kt")
        print(f"  Satellite Frame: {frame_t0_path}")

        if not frame_t0_path.exists():
            raise FileNotFoundError(f"Missing satellite file: {frame_t0_path}")

        if sample_id not in cached_samples:
            raise KeyError(f"Sample {sample_id} missing from feature cache!")

        cached_rec = cached_samples[sample_id]
        sat_seq = cached_rec["sat_seq"].unsqueeze(0)  # [1, 6, 132]
        env_seq = cached_rec["env_seq"].unsqueeze(0)  # [1, 6, 8, 41, 66]

        # Load satellite frames
        vis_norm, model_norm = load_raw_satellite_frame(frame_t0_path, mean_kelvin, std_kelvin)
        t0_tensor = torch.from_numpy(model_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, 572, 929]

        # Generate Grad-CAM Saliency
        saliency_out = explainer.generate_saliency(
            t0_img_tensor=t0_tensor,
            cached_seq_132=sat_seq,
            env_seq_tensor=env_seq,
        )

        pred_class = saliency_out["predicted_class"]
        pred_idx = saliency_out["predicted_class_idx"]
        confidence = float(saliency_out["confidence"])
        cam_resized = saliency_out["cam_resized"]  # [572, 929] in [0, 1]
        max_loc = saliency_out["max_activation_loc"]
        centroid = saliency_out["activation_centroid"]

        print(f"  Predicted Class: {pred_class} (Conf: {confidence:.2%})")
        print(f"  Target Layer: {saliency_out['target_layer']}")
        print(f"  Max Activation Loc: {max_loc}, Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")

        # Render overlay
        overlay_rgb = explainer.render_overlay(
            original_img=vis_norm,
            heatmap_norm=cam_resized,
            alpha=0.45,
            colormap_name="turbo",
        )

        # File names
        base_name = f"{storm_name.lower()}_{t0_tag}"
        orig_file = f"original_{base_name}.png"
        heat_file = f"heatmap_{base_name}.png"
        over_file = f"overlay_{base_name}.png"
        meta_file = f"metadata_{base_name}.json"

        orig_path = OUTPUT_DIR / orig_file
        heat_path = OUTPUT_DIR / heat_file
        over_path = OUTPUT_DIR / over_file
        meta_path = OUTPUT_DIR / meta_file

        # Save Original Grayscale Image
        orig_img_uint8 = (vis_norm * 255.0).astype(np.uint8)
        Image.fromarray(orig_img_uint8).save(orig_path)

        # Save Colormapped Heatmap
        cmap = plt.get_cmap("turbo")
        heatmap_rgb = (cmap(cam_resized)[:, :, :3] * 255.0).astype(np.uint8)
        Image.fromarray(heatmap_rgb).save(heat_path)

        # Save Overlay Image
        Image.fromarray(overlay_rgb).save(over_path)

        # Save Structured Metadata
        metadata: Dict[str, Any] = {
            "sample_id": sample_id,
            "storm_id": storm_id,
            "storm_name": storm_name,
            "t0": t0_str,
            "source": "GridSat-B1 IRWIN CDR",
            "split": split,
            "frame_index": 5,
            "ground_truth_category": truth_cat,
            "ground_truth_wind_kt": truth_wind,
            "predicted_class": pred_class,
            "predicted_class_idx": pred_idx,
            "confidence": confidence,
            "method": "Grad-CAM",
            "target_layer": saliency_out["target_layer"],
            "image_height": int(cam_resized.shape[0]),
            "image_width": int(cam_resized.shape[1]),
            "max_activation_loc": [int(max_loc[0]), int(max_loc[1])],
            "activation_centroid": [float(centroid[0]), float(centroid[1])],
            "heatmap_path": f"data/interim/ml/explainability/{heat_file}",
            "overlay_path": f"data/interim/ml/explainability/{over_file}",
            "original_path": f"data/interim/ml/explainability/{orig_file}",
            "model_checkpoint": "data/interim/ml/checkpoints/best_phase6_intensity_wind.pt",
            "spatial_checkpoint": "data/interim/ml/checkpoints/best_center_localization_cnn.pt",
            "temporal_checkpoint": "data/interim/ml/checkpoints/best_temporal_track_gru.pt",
            "dataset_version": "v1.0-locked",
            "preprocessing_version": "TRAIN_zscore_standardization",
            "interpretation_note": (
                "Interpretation aid showing regions that contributed most strongly "
                "to the model's IMD intensity classification. "
                "Must not be presented or interpreted as a causal meteorological explanation."
            ),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        manifest_records.append(metadata)
        print(f"  Artifacts saved for {storm_name}:")
        print(f"    - Original: {orig_file}")
        print(f"    - Heatmap:  {heat_file}")
        print(f"    - Overlay:  {over_file}")
        print(f"    - Metadata: {meta_file}")

    # Save summary manifest
    manifest_path = OUTPUT_DIR / "explainability_manifest.json"
    summary_doc = {
        "status": "COMPLETED",
        "num_reference_cases": len(manifest_records),
        "cases": manifest_records,
        "method": "Grad-CAM",
        "target_layer": "spatial_encoder.layer2",
        "spatial_resolution": [572, 929],
        "interpretation_disclaimer": (
            "The saliency/Grad-CAM output is an interpretation aid and must not "
            "be presented as a causal meteorological explanation."
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(summary_doc, f, indent=2)

    explainer.close()
    print("\n" + "=" * 65)
    print(f"EXPLAINABILITY PIPELINE COMPLETE: {len(manifest_records)} reference cases generated.")
    print(f"Manifest written to: {manifest_path}")
    print("=" * 65)
    return manifest_records


if __name__ == "__main__":
    run_explainability_pipeline()
