"""
VAYU-NET — PyTorch Satellite Sequence Dataset
Lazy-loading dataset for 6-frame GridSat sequences and multi-horizon IMD targets.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Canonical 7-class IMD cyclonic category mapping
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

DEFAULT_SAMPLE_INDEX = "data/manifests/vayu_net_sample_index.csv"
DEFAULT_NORM_STATS = "data/interim/ml/train_normalization_stats.json"

FRAME_COLUMNS = [
    "frame_t_minus_15h",
    "frame_t_minus_12h",
    "frame_t_minus_9h",
    "frame_t_minus_6h",
    "frame_t_minus_3h",
    "frame_t0"
]

class VayuSatelliteDataset(Dataset):
    """
    PyTorch Dataset representing 6-step temporal satellite sequences aligned to
    multi-horizon IMD ground-truth track and intensity targets.
    """

    def __init__(self,
                 sample_index_csv=DEFAULT_SAMPLE_INDEX,
                 split="TRAIN",
                 norm_stats_path=DEFAULT_NORM_STATS,
                 normalize=True,
                 transform=None):
        super().__init__()
        
        if not os.path.exists(sample_index_csv):
            raise FileNotFoundError(f"Sample index not found at: {sample_index_csv}")
            
        full_df = pd.read_csv(sample_index_csv)
        
        if split is not None:
            split_upper = split.upper()
            if split_upper not in ["TRAIN", "VALIDATION", "TEST"]:
                raise ValueError(f"Invalid split: {split}. Expected one of ['TRAIN', 'VALIDATION', 'TEST']")
            self.df = full_df[full_df["split"] == split_upper].reset_index(drop=True)
            self.split = split_upper
        else:
            self.df = full_df.reset_index(drop=True)
            self.split = "ALL"

        self.normalize = normalize
        self.transform = transform
        self.norm_stats_path = norm_stats_path

        # Load or compute training-only normalization statistics
        self._load_or_compute_norm_stats(full_df)

    def _load_or_compute_norm_stats(self, full_df):
        """Loads normalization statistics strictly from TRAIN samples"""
        if os.path.exists(self.norm_stats_path):
            with open(self.norm_stats_path, "r") as f:
                stats = json.load(f)
            self.mean_kelvin = float(stats["mean_kelvin"])
            self.std_kelvin = float(stats["std_kelvin"])
        else:
            # Compute strictly on TRAIN split
            train_rows = full_df[full_df["split"] == "TRAIN"]
            train_frames = sorted(list(set(train_rows[FRAME_COLUMNS].values.flatten())))
            
            total_count = 0
            sum_val = 0.0
            sum_sq_val = 0.0
            nan_count = 0
            
            for p in train_frames:
                arr = np.load(p)["irwin_cdr"]
                valid = ~np.isnan(arr)
                nan_count += int((~valid).sum())
                vdata = arr[valid].astype(np.float64)
                total_count += len(vdata)
                sum_val += float(np.sum(vdata))
                sum_sq_val += float(np.sum(vdata ** 2))
                
            self.mean_kelvin = sum_val / total_count
            variance = (sum_sq_val / total_count) - (self.mean_kelvin ** 2)
            self.std_kelvin = float(np.sqrt(variance))
            
            stats = {
                "split": "TRAIN",
                "num_train_samples": len(train_rows),
                "num_unique_train_frames": len(train_frames),
                "total_valid_pixels": int(total_count),
                "total_nan_pixels": int(nan_count),
                "mean_kelvin": self.mean_kelvin,
                "std_kelvin": self.std_kelvin,
                "units": "Kelvin",
                "missing_value_fill": "mean_kelvin"
            }
            os.makedirs(os.path.dirname(self.norm_stats_path), exist_ok=True)
            with open(self.norm_stats_path, "w") as f:
                json.dump(stats, f, indent=2)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Lazy load 6-frame satellite sequence
        frame_arrays = []
        for col in FRAME_COLUMNS:
            fpath = row[col]
            # Load from NumPy archive
            with np.load(fpath) as npz:
                arr = npz["irwin_cdr"].astype(np.float32)  # shape: (572, 929)
                
            # Handle invalid / missing pixels explicitly before normalization
            invalid_mask = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
            if np.any(invalid_mask):
                arr = arr.copy()
                arr[invalid_mask] = self.mean_kelvin
                
            # Apply training-only standardization: z = (x - mean) / std
            if self.normalize:
                arr = (arr - self.mean_kelvin) / self.std_kelvin
                
            frame_arrays.append(arr)
            
        # Shape: (6, 572, 929)
        sequence_np = np.stack(frame_arrays, axis=0)
        satellite_tensor = torch.from_numpy(sequence_np).float()
        
        if self.transform is not None:
            satellite_tensor = self.transform(satellite_tensor)
            
        # 2. Extract targets and validity masks
        def parse_target(lat_val, lon_val, wind_val, pres_val, cat_val):
            # Center coordinates
            c_lat = float(lat_val)
            c_lon = float(lon_val)
            center_t = torch.tensor([c_lat, c_lon], dtype=torch.float32)
            
            # Wind speed (kt)
            if pd.isna(wind_val):
                w_val = 0.0
                w_mask = 0.0
            else:
                w_val = float(wind_val)
                w_mask = 1.0
            wind_t = torch.tensor(w_val, dtype=torch.float32)
            wind_mask_t = torch.tensor(w_mask, dtype=torch.float32)
            
            # Central pressure (hPa)
            if pd.isna(pres_val):
                p_val = 0.0
                p_mask = 0.0
            else:
                p_val = float(pres_val)
                p_mask = 1.0
            pres_t = torch.tensor(p_val, dtype=torch.float32)
            pres_mask_t = torch.tensor(p_mask, dtype=torch.float32)
            
            # Category class (0-6)
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
            "satellite_sequence": satellite_tensor,
            # Current state (t0)
            "center_t0": c0,
            "wind_t0": w0,
            "wind_t0_mask": wm0,
            "pressure_t0": p0,
            "pressure_t0_mask": pm0,
            "category_t0": cat0,
            "category_t0_mask": catm0,
            # Forecast horizon +12h
            "center_12h": c12,
            "wind_12h": w12,
            "wind_12h_mask": wm12,
            "pressure_12h": p12,
            "pressure_12h_mask": pm12,
            "category_12h": cat12,
            "category_12h_mask": catm12,
            # Forecast horizon +24h
            "center_24h": c24,
            "wind_24h": w24,
            "wind_24h_mask": wm24,
            "pressure_24h": p24,
            "pressure_24h_mask": pm24,
            "category_24h": cat24,
            "category_24h_mask": catm24,
            # Forecast horizon +48h
            "center_48h": c48,
            "wind_48h": w48,
            "wind_48h_mask": wm48,
            "pressure_48h": p48,
            "pressure_48h_mask": pm48,
            "category_48h": cat48,
            "category_48h_mask": catm48,
            # Metadata
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "t0": row["t0"],
            "split": row["split"]
        }


class SingleFrameVayuDataset(Dataset):
    """
    PyTorch Dataset strictly loading ONLY frame_t0 for single-frame CNN ablation baselines.
    Bypasses the 5 historical frames to enforce causality and optimize loading throughput.
    """
    def __init__(self,
                 sample_index_csv=DEFAULT_SAMPLE_INDEX,
                 split="TRAIN",
                 norm_stats_path=DEFAULT_NORM_STATS,
                 normalize=True):
        super().__init__()
        
        if not os.path.exists(sample_index_csv):
            raise FileNotFoundError(f"Sample index not found at: {sample_index_csv}")
            
        full_df = pd.read_csv(sample_index_csv)
        
        if split is not None:
            split_upper = split.upper()
            if split_upper not in ["TRAIN", "VALIDATION", "TEST"]:
                raise ValueError(f"Invalid split: {split}. Expected one of ['TRAIN', 'VALIDATION', 'TEST']")
            self.df = full_df[full_df["split"] == split_upper].reset_index(drop=True)
            self.split = split_upper
        else:
            self.df = full_df.reset_index(drop=True)
            self.split = "ALL"

        self.normalize = normalize
        self.norm_stats_path = norm_stats_path

        # Load normalization stats strictly from JSON (TRAIN-only baseline)
        if not os.path.exists(self.norm_stats_path):
            raise FileNotFoundError(f"Normalization stats file missing: {self.norm_stats_path}")
            
        with open(self.norm_stats_path, "r") as f:
            stats = json.load(f)
            
        self.mean_kelvin = float(stats["mean_kelvin"])
        self.std_kelvin = float(stats["std_kelvin"])
        self.train_wind_mean = float(stats.get("train_wind_mean_kt", 36.88362))
        self.train_wind_std = float(stats.get("train_wind_std_kt", 17.97539))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load strictly frame_t0
        fpath = row["frame_t0"]
        with np.load(fpath) as npz:
            arr = npz["irwin_cdr"].astype(np.float32)  # shape: (572, 929)
            
        # Handle invalid / missing pixels
        invalid_mask = np.isnan(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(invalid_mask):
            arr = arr.copy()
            arr[invalid_mask] = self.mean_kelvin
            
        if self.normalize:
            arr = (arr - self.mean_kelvin) / self.std_kelvin
            
        # Add channel dimension: shape [1, 572, 929]
        image_tensor = torch.from_numpy(arr[np.newaxis, :, :]).float()
        
        # 2. Ground-truth Targets
        # Center (lat, lon)
        lat = float(row["imd_lat_t0"])
        lon = float(row["imd_lon_t0"])
        center_deg = torch.tensor([lat, lon], dtype=torch.float32)
        
        # Normalized center: (lat + 5)/40, (lon - 40)/65
        u_lat = (lat - (-5.0)) / 40.0
        u_lon = (lon - 40.0) / 65.0
        norm_center = torch.tensor([u_lat, u_lon], dtype=torch.float32)
        
        # Wind (kt)
        w_raw = row["imd_wind_t0"]
        if pd.isna(w_raw):
            w_val = 0.0
            w_mask = 0.0
            u_wind = 0.0
        else:
            w_val = float(w_raw)
            w_mask = 1.0
            u_wind = (w_val - self.train_wind_mean) / self.train_wind_std
            
        wind_kt = torch.tensor(w_val, dtype=torch.float32)
        norm_wind = torch.tensor(u_wind, dtype=torch.float32)
        wind_mask = torch.tensor(w_mask, dtype=torch.float32)
        
        # Category (0-6)
        cat_raw = row["imd_category_t0"]
        if pd.isna(cat_raw) or cat_raw not in CATEGORY_TO_IDX:
            cat_idx = -1
            cat_mask = 0.0
        else:
            cat_idx = CATEGORY_TO_IDX[cat_raw]
            cat_mask = 1.0
            
        category = torch.tensor(cat_idx, dtype=torch.long)
        category_mask = torch.tensor(cat_mask, dtype=torch.float32)
        
        return {
            "satellite_image": image_tensor,
            "norm_center": norm_center,
            "center_deg": center_deg,
            "norm_wind": norm_wind,
            "wind_kt": wind_kt,
            "wind_mask": wind_mask,
            "category": category,
            "category_mask": category_mask,
            "sample_id": row["sample_id"],
            "storm_id": row["storm_id"],
            "t0": row["t0"],
            "split": row["split"]
        }

