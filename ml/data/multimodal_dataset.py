"""
VAYU-NET: Multimodal Satellite Dataset (GridSat-B1 + NASA GPM IMERG Final Run V07B).
Integrates calibrated IR brightness temperature and satellite precipitation rate.

Modality Contract:
- gridsat: [6, 72, 116] (standardized via TRAIN-only Kelvin stats)
- imerg:   [6, 72, 116] (standardized via TRAIN-only mm/hr stats)
- targets: Center coordinates, maximum sustained wind, central pressure, and 7-class category
           at t0, +12h, +24h, +48h.

Guarantees:
- Strict storm-level split preservation
- Zero temporal leakage (delta-t = 0, no future frames relative to t0)
- Zero synthetic data fallback
- Zero ground-truth IMD center cropping (fixed NIO synoptic basin [-5, 35]N, [40, 105]E)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.data.process_imerg import IMERGProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultimodalDataset")

DEFAULT_MULTIMODAL_MANIFEST = "data/manifests/gridsat_imerg_multimodal_dataset_manifest.csv"
DEFAULT_SAMPLE_INDEX = "data/manifests/vayu_net_sample_index.csv"
DEFAULT_GRIDSAT_NORM = "data/interim/ml/train_normalization_stats.json"
DEFAULT_IMERG_NORM = "data/interim/ml/imerg_train_normalization_stats.json"
DEFAULT_IMERG_RAW_DIR = "data/raw/imerg"

TARGET_H = 72
TARGET_W = 116

CATEGORY_TO_IDX = {
    "D": 0, "DD": 1, "CS": 2, "SCS": 3, "VSCS": 4, "ESCS": 5, "SuCS": 6
}
IDX_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_IDX.items()}


class MultimodalVayuDataset(Dataset):
    """
    PyTorch Dataset for paired GridSat-B1 (IR) and NASA GPM IMERG Final Run V07B (Precipitation)
    6-frame temporal sequences.
    """

    def __init__(self,
                 manifest_csv: str = DEFAULT_MULTIMODAL_MANIFEST,
                 sample_index_csv: str = DEFAULT_SAMPLE_INDEX,
                 split: Optional[str] = "TRAIN",
                 gridsat_norm_path: str = DEFAULT_GRIDSAT_NORM,
                 imerg_norm_path: str = DEFAULT_IMERG_NORM,
                 imerg_raw_dir: str = DEFAULT_IMERG_RAW_DIR,
                 normalize: bool = True,
                 use_log1p_imerg: bool = False,
                 transform=None):
        super().__init__()
        
        self.manifest_path = Path(manifest_csv)
        self.index_path = Path(sample_index_csv)
        self.gridsat_norm_path = Path(gridsat_norm_path)
        self.imerg_norm_path = Path(imerg_norm_path)
        self.imerg_raw_dir = Path(imerg_raw_dir)
        self.normalize = normalize
        self.use_log1p_imerg = use_log1p_imerg
        self.transform = transform
        
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Multimodal manifest not found: {self.manifest_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Sample index not found: {self.index_path}")
            
        manifest_df = pd.read_csv(self.manifest_path)
        idx_df = pd.read_csv(self.index_path)
        
        # Merge target columns from sample index
        target_cols = [
            "sample_id",
            "imd_lat_t0", "imd_lon_t0", "imd_wind_t0", "imd_pressure_t0", "imd_category_t0",
            "imd_lat_12h", "imd_lon_12h", "wind_12h", "pressure_12h", "category_12h",
            "imd_lat_24h", "imd_lon_24h", "wind_24h", "pressure_24h", "category_24h",
            "imd_lat_48h", "imd_lon_48h", "wind_48h", "pressure_48h", "category_48h"
        ]
        merged = pd.merge(manifest_df, idx_df[target_cols], on="sample_id", how="inner")
        assert len(merged) == len(manifest_df), "Mismatch merging manifest with target sample index"
        
        if split is not None:
            split_upper = split.upper()
            if split_upper not in ["TRAIN", "VALIDATION", "TEST"]:
                raise ValueError(f"Invalid split: {split}. Expected one of ['TRAIN', 'VALIDATION', 'TEST']")
            self.df = merged[merged["split"] == split_upper].reset_index(drop=True)
            self.split = split_upper
        else:
            self.df = merged.reset_index(drop=True)
            self.split = "ALL"
            
        # Load normalization parameters (strictly TRAIN statistics)
        self._load_norm_stats()
        
        # Initialize IMERG processor with model grid resolution
        self.imerg_processor = IMERGProcessor(target_resolution="model")
        self._gridsat_cache = {}
        self._imerg_cache = {}
        
    def _load_norm_stats(self):
        """Loads normalization statistics derived strictly from TRAIN storms."""
        with open(self.gridsat_norm_path, "r") as f:
            g_stats = json.load(f)
        self.gridsat_mean_k = float(g_stats["mean_kelvin"])
        self.gridsat_std_k = float(g_stats["std_kelvin"])
        
        with open(self.imerg_norm_path, "r") as f:
            i_stats = json.load(f)
        self.imerg_raw_mean = float(i_stats["raw_stats"]["mean"])
        self.imerg_raw_std = float(i_stats["raw_stats"]["std"])
        self.imerg_log1p_mean = float(i_stats["log1p_stats"]["mean"])
        self.imerg_log1p_std = float(i_stats["log1p_stats"]["std"])
        self.imerg_nan_fill = self.imerg_raw_mean

    def __len__(self) -> int:
        return len(self.df)

    def _load_gridsat_frame(self, fpath: str) -> np.ndarray:
        """Loads and resamples a single GridSat frame to [72, 116]."""
        if fpath in self._gridsat_cache:
            return self._gridsat_cache[fpath]

        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)  # Raw (572, 929)
            
        # Impute invalid/missing pixels
        invalid = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(invalid):
            arr = arr.copy()
            arr[invalid] = self.gridsat_mean_k
            
        # Downsample to [72, 116]
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t_down = F.interpolate(t, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        arr_down = t_down.numpy()
        
        if self.normalize:
            arr_down = (arr_down - self.gridsat_mean_k) / self.gridsat_std_k
            
        self._gridsat_cache[fpath] = arr_down
        return arr_down

    def _load_imerg_frame(self, granule_filename: str) -> np.ndarray:
        """Loads, cleans, and resamples a single IMERG granule to [72, 116]."""
        if granule_filename in self._imerg_cache:
            return self._imerg_cache[granule_filename]

        h5_path = self.imerg_raw_dir / granule_filename
        res = self.imerg_processor.process_granule(h5_path, verify_sha=False)
        arr = res["precipitation"]  # [72, 116], physical mm/hr with NaNs
        
        # Fill missing values with train mean
        nan_mask = np.isnan(arr)
        if np.any(nan_mask):
            arr = arr.copy()
            arr[nan_mask] = self.imerg_nan_fill
            
        if self.normalize:
            if self.use_log1p_imerg:
                arr = (np.log1p(np.maximum(0.0, arr)) - self.imerg_log1p_mean) / self.imerg_log1p_std
            else:
                arr = (arr - self.imerg_raw_mean) / self.imerg_raw_std
                
        arr = arr.astype(np.float32)
        self._imerg_cache[granule_filename] = arr
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 1. Load 6 GridSat frames -> [6, 72, 116]
        gs_paths = row["gridsat_sequence_reference"].split("|")
        gs_frames = [self._load_gridsat_frame(p) for p in gs_paths]
        gridsat_tensor = torch.from_numpy(np.stack(gs_frames, axis=0)).float()
        
        # 2. Load 6 IMERG frames -> [6, 72, 116]
        im_granules = row["imerg_sequence_reference"].split("|")
        im_frames = [self._load_imerg_frame(g) for g in im_granules]
        imerg_tensor = torch.from_numpy(np.stack(im_frames, axis=0)).float()
        
        if self.transform is not None:
            gridsat_tensor, imerg_tensor = self.transform(gridsat_tensor, imerg_tensor)
            
        # 3. Targets and validity masks
        def parse_target(lat_val, lon_val, wind_val, pres_val, cat_val):
            c_lat = float(lat_val)
            c_lon = float(lon_val)
            center_t = torch.tensor([c_lat, c_lon], dtype=torch.float32)
            
            w_val = float(wind_val) if not pd.isna(wind_val) else 0.0
            w_mask = 1.0 if not pd.isna(wind_val) else 0.0
            wind_t = torch.tensor(w_val, dtype=torch.float32)
            wind_mask_t = torch.tensor(w_mask, dtype=torch.float32)
            
            p_val = float(pres_val) if not pd.isna(pres_val) else 0.0
            p_mask = 1.0 if not pd.isna(pres_val) else 0.0
            pres_t = torch.tensor(p_val, dtype=torch.float32)
            pres_mask_t = torch.tensor(p_mask, dtype=torch.float32)
            
            if pd.isna(cat_val) or cat_val not in CATEGORY_TO_IDX:
                c_idx = -1
                c_mask = 0.0
            else:
                c_idx = CATEGORY_TO_IDX[cat_val]
                c_mask = 1.0
            cat_t = torch.tensor(c_idx, dtype=torch.long)
            cat_mask_t = torch.tensor(c_mask, dtype=torch.float32)
            
            return center_t, wind_t, wind_mask_t, pres_t, pres_mask_t, cat_t, cat_mask_t
            
        c0, w0, wm0, p0, pm0, cat0, catm0 = parse_target(
            row["imd_lat_t0"], row["imd_lon_t0"], row["imd_wind_t0"], row["imd_pressure_t0"], row["imd_category_t0"]
        )
        c12, w12, wm12, p12, pm12, cat12, catm12 = parse_target(
            row["imd_lat_12h"], row["imd_lon_12h"], row["wind_12h"], row["pressure_12h"], row["category_12h"]
        )
        c24, w24, wm24, p24, pm24, cat24, catm24 = parse_target(
            row["imd_lat_24h"], row["imd_lon_24h"], row["wind_24h"], row["pressure_24h"], row["category_24h"]
        )
        c48, w48, wm48, p48, pm48, cat48, catm48 = parse_target(
            row["imd_lat_48h"], row["imd_lon_48h"], row["wind_48h"], row["pressure_48h"], row["category_48h"]
        )
        
        return {
            "gridsat": gridsat_tensor,      # [6, 72, 116]
            "imerg": imerg_tensor,          # [6, 72, 116]
            
            # Forecast horizon targets
            "center_t0": c0,
            "wind_t0": w0,
            "wind_t0_mask": wm0,
            "pressure_t0": p0,
            "pressure_t0_mask": pm0,
            "category_t0": cat0,
            "category_t0_mask": catm0,
            
            "center_12h": c12,
            "wind_12h": w12,
            "wind_12h_mask": wm12,
            "pressure_12h": p12,
            "pressure_12h_mask": pm12,
            "category_12h": cat12,
            "category_12h_mask": catm12,
            
            "center_24h": c24,
            "wind_24h": w24,
            "wind_24h_mask": wm24,
            "pressure_24h": p24,
            "pressure_24h_mask": pm24,
            "category_24h": cat24,
            "category_24h_mask": catm24,
            
            "center_48h": c48,
            "wind_48h": w48,
            "wind_48h_mask": wm48,
            "pressure_48h": p48,
            "pressure_48h_mask": pm48,
            "category_48h": cat48,
            "category_48h_mask": catm48,
            
            # Sequence metadata
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "storm_name": row["storm_name"],
            "split": row["split"],
            "t0_utc": row["t0_utc"],
            "gridsat_paths": gs_paths,
            "imerg_granules": im_granules
        }