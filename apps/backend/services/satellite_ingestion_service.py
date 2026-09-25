"""
VAYU-NET — Satellite Observation Ingestion Service
===================================================
Responsible for ingesting, validating, and preprocessing geostationary
infrared satellite observations (NOAA NCEI GridSat-B1 IRWIN CDR) for current
and live cyclone events.

Key Invariants:
1. Causal 6-frame observation sequence: [t-15h, t-12h, t-9h, t-6h, t-3h, t0].
2. Zero future observation leakage: never ingests or exposes frames at t > t0.
3. No artificial interpolation: if any of the 6 causal frames is missing,
   declares WAITING_FOR_FRAMES instead of fabricating synthetic pixels.
4. Identical production normalization: applies train-split mean (265.41 K)
   and standard deviation (24.89 K) with invalid pixel imputation.
5. Domain bounds: North Indian Ocean [-5.0° to 35.0°N, 40.0° to 105.0°E].
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("vayu.backend.satellite_ingestion")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GRIDSAT_BASE_DIR = PROJECT_ROOT / "data" / "interim" / "gridsat"
NORM_STATS_PATH = PROJECT_ROOT / "data" / "interim" / "ml" / "train_normalization_stats.json"

# Canonical North Indian Ocean Grid Parameters
LAT_MIN, LAT_MAX = -5.0, 35.0
LON_MIN, LON_MAX = 40.0, 105.0
RAW_H, RAW_W = 572, 929
SEQ_TARGET_H, SEQ_TARGET_W = 72, 116

DEFAULT_MEAN_K = 265.41
DEFAULT_STD_K = 24.89

REQUIRED_OFFSETS_HOURS = [-15, -12, -9, -6, -3, 0]
STEP_NAMES = ["t_minus_15h", "t_minus_12h", "t_minus_9h", "t_minus_6h", "t_minus_3h", "t0"]


class SatelliteIngestionError(Exception):
    """Exception raised for satellite ingestion failures."""
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class SatelliteIngestionService:
    """
    Ingests and validates geostationary satellite frames for operational cyclone inference.
    """

    def __init__(
        self,
        gridsat_dir: Optional[Path] = None,
        norm_stats_path: Optional[Path] = None,
    ):
        self.gridsat_dir = Path(gridsat_dir) if gridsat_dir else GRIDSAT_BASE_DIR
        self.norm_stats_path = Path(norm_stats_path) if norm_stats_path else NORM_STATS_PATH
        self._mean_k = DEFAULT_MEAN_K
        self._std_k = DEFAULT_STD_K
        self._load_norm_stats()

    def _load_norm_stats(self) -> None:
        """Loads train-split normalization constants."""
        if self.norm_stats_path.exists():
            try:
                with open(self.norm_stats_path, "r") as f:
                    stats = json.load(f)
                self._mean_k = float(stats.get("mean_kelvin", DEFAULT_MEAN_K))
                self._std_k = float(stats.get("std_kelvin", DEFAULT_STD_K))
                logger.info(f"Loaded satellite norm stats: mean={self._mean_k:.2f}K, std={self._std_k:.2f}K")
            except Exception as e:
                logger.warning(f"Could not load norm stats from {self.norm_stats_path}: {e}. Using defaults.")

    @staticmethod
    def parse_utc_timestamp(ts_str: str) -> datetime:
        """Parses ISO-8601 UTC timestamp string."""
        clean = ts_str.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(clean)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except ValueError as err:
            raise SatelliteIngestionError("INVALID_TIMESTAMP", f"Invalid ISO-8601 timestamp '{ts_str}': {err}")

    def get_expected_frame_timestamps(self, t0_dt: datetime) -> List[Tuple[str, int, datetime]]:
        """Returns the 6 required causal frame timestamps."""
        frames = []
        for name, offset in zip(STEP_NAMES, REQUIRED_OFFSETS_HOURS):
            f_dt = t0_dt + timedelta(hours=offset)
            frames.append((name, offset, f_dt))
        return frames

    def find_gridsat_file_for_timestamp(self, dt: datetime) -> Optional[Path]:
        """
        Locates a GridSat .npz or .nc file on disk matching year/month/day/hour.
        Expected format: data/interim/gridsat/{YYYY}/gridsat_{YYYY}.{MM}.{DD}.{HH}.npz
        """
        year_str = str(dt.year)
        fname_base = f"gridsat_{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}"

        # 1. Search year subdirectory in interim
        candidate_dir = self.gridsat_dir / year_str
        if candidate_dir.exists():
            npz_candidate = candidate_dir / f"{fname_base}.npz"
            if npz_candidate.exists():
                return npz_candidate
            nc_candidate = candidate_dir / f"{fname_base}.nc"
            if nc_candidate.exists():
                return nc_candidate

        # 2. Search root gridsat dir directly
        npz_root = self.gridsat_dir / f"{fname_base}.npz"
        if npz_root.exists():
            return npz_root

        return None

    def check_sequence_readiness(
        self,
        t0_dt: datetime,
    ) -> Dict[str, Any]:
        """
        Checks whether the 6 causal frames exist on disk for the given t0.
        Does NOT allow future frames. Does NOT interpolate.
        """
        expected = self.get_expected_frame_timestamps(t0_dt)
        available_frames = []
        missing_frames = []

        now_utc = datetime.now(timezone.utc)

        for step_name, offset_h, f_dt in expected:
            iso_str = f_dt.isoformat()
            # Strictly verify no future frame relative to t0
            if f_dt > t0_dt:
                raise SatelliteIngestionError(
                    "FUTURE_FRAME_VIOLATION",
                    f"Frame {step_name} ({iso_str}) exceeds issue timestamp t0 ({t0_dt.isoformat()})."
                )

            f_path = self.find_gridsat_file_for_timestamp(f_dt)
            if f_path is not None and f_path.exists():
                available_frames.append({
                    "step": step_name,
                    "offset_hours": offset_h,
                    "timestamp_utc": iso_str,
                    "file_path": str(f_path.relative_to(PROJECT_ROOT) if f_path.is_relative_to(PROJECT_ROOT) else f_path),
                    "file_name": f_path.name,
                    "status": "AVAILABLE"
                })
            else:
                missing_frames.append({
                    "step": step_name,
                    "offset_hours": offset_h,
                    "timestamp_utc": iso_str,
                    "status": "MISSING"
                })

        count_avail = len(available_frames)
        if count_avail == 6:
            readiness_status = "READY"
            message = "All 6 causal GridSat observation frames validated and available."
        elif count_avail == 0:
            readiness_status = "SOURCE_UNAVAILABLE"
            message = f"Zero GridSat observation frames found for observation window ending {t0_dt.isoformat()}."
        else:
            readiness_status = "WAITING_FOR_FRAMES"
            missing_offsets = [f"{m['step']} ({m['offset_hours']}h)" for m in missing_frames]
            message = f"Incomplete satellite sequence: {count_avail}/6 frames available. Missing: {', '.join(missing_offsets)}."

        return {
            "t0_utc": t0_dt.isoformat(),
            "status": readiness_status,
            "message": message,
            "available_count": count_avail,
            "required_count": 6,
            "is_complete": (count_avail == 6),
            "available_frames": available_frames,
            "missing_frames": missing_frames,
        }

    def load_and_preprocess_single_frame(
        self,
        file_path: Path,
        downsample_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Loads a single GridSat frame (.npz or .nc), imputes invalid pixels,
        and returns normalized brightness temperature.
        """
        p = Path(file_path)
        if not p.exists():
            raise SatelliteIngestionError("FILE_NOT_FOUND", f"GridSat frame not found at {p}")

        if p.suffix == ".npz":
            with np.load(p) as npz:
                if "irwin_cdr" not in npz:
                    raise SatelliteIngestionError("CORRUPT_FILE", f"'irwin_cdr' array not found in {p.name}")
                arr = npz["irwin_cdr"].astype(np.float32)
        elif p.suffix == ".nc":
            try:
                import netCDF4 as nc
                with nc.Dataset(p, "r") as ds:
                    if "irwin_cdr" in ds.variables:
                        raw = ds.variables["irwin_cdr"][:]
                        arr = np.squeeze(raw).astype(np.float32)
                    else:
                        raise SatelliteIngestionError("CORRUPT_FILE", f"'irwin_cdr' variable not found in NetCDF {p.name}")
            except Exception as e:
                raise SatelliteIngestionError("NETCDF_READ_ERROR", f"Failed to read NetCDF {p.name}: {e}")
        else:
            raise SatelliteIngestionError("UNSUPPORTED_FORMAT", f"Unsupported file extension {p.suffix}")

        # Check raw spatial shape
        if arr.shape != (RAW_H, RAW_W):
            if arr.ndim == 2:
                # Interpolate if needed to canonical RAW_H x RAW_W
                t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                arr = F.interpolate(t, size=(RAW_H, RAW_W), mode="bilinear", align_corners=False).squeeze().numpy()
            else:
                raise SatelliteIngestionError("INVALID_SHAPE", f"Unexpected frame shape {arr.shape} in {p.name}")

        # Impute invalid pixels
        invalid = np.isnan(arr) | np.isinf(arr) | (arr < 100.0) | (arr > 380.0)
        if np.any(invalid):
            arr = arr.copy()
            arr[invalid] = self._mean_k

        # Downsample if requested (e.g. for sequence GRU [72, 116])
        if downsample_shape is not None:
            t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            t_down = F.interpolate(t, size=downsample_shape, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            arr = t_down.numpy()

        # Normalize using train-split constants
        arr_norm = (arr - self._mean_k) / self._std_k
        return arr_norm.astype(np.float32)

    def verify_frame_integrity(self, file_path: Path) -> Tuple[bool, str]:
        """Verifies that a satellite file exists, is non-empty, and loadable."""
        p = Path(file_path)
        if not p.exists():
            return False, "FILE_NOT_FOUND"
        if p.stat().st_size < 1000:
            return False, "FILE_UNDERSIZED"
        try:
            _ = self.load_and_preprocess_single_frame(p)
            return True, "VALID"
        except Exception as e:
            return False, str(e)

    def ingest_causal_sequence(
        self,
        t0_dt: datetime,
    ) -> Dict[str, Any]:
        """
        Executes strict ingestion of the 6 causal frames [t-15h ... t0].
        Raises SatelliteIngestionError if sequence is incomplete.
        Returns:
            - t0_tensor: [1, 1, 572, 929] torch float32 tensor
            - seq_tensor_72_116: [1, 6, 72, 116] torch float32 tensor
            - frames_meta: list of frame descriptors
        """
        readiness = self.check_sequence_readiness(t0_dt)
        if not readiness["is_complete"]:
            raise SatelliteIngestionError(
                "INCOMPLETE_OBSERVATION_SEQUENCE",
                f"Cannot ingest causal sequence for t0={t0_dt.isoformat()}: {readiness['message']}",
                status_code=422
            )

        frames_meta = readiness["available_frames"]
        # Ensure chronological ordering: t-15h -> t0
        frames_meta.sort(key=lambda x: x["offset_hours"])

        seq_frames_72 = []
        t0_arr_raw = None

        for idx, f_info in enumerate(frames_meta):
            f_path = PROJECT_ROOT / f_info["file_path"] if not Path(f_info["file_path"]).is_absolute() else Path(f_info["file_path"])
            arr_72 = self.load_and_preprocess_single_frame(f_path, downsample_shape=(SEQ_TARGET_H, SEQ_TARGET_W))
            seq_frames_72.append(arr_72)

            if f_info["offset_hours"] == 0:
                # Load full-resolution frame for Phase 3C center localization
                t0_arr_raw = self.load_and_preprocess_single_frame(f_path, downsample_shape=None)

        if t0_arr_raw is None:
            raise SatelliteIngestionError("MISSING_T0_FRAME", "t0 frame missing during tensor assembly")

        # Construct tensors
        t0_tensor = torch.from_numpy(t0_arr_raw).unsqueeze(0).unsqueeze(0).float() # [1, 1, 572, 929]
        seq_arr = np.stack(seq_frames_72, axis=0) # [6, 72, 116]
        seq_tensor = torch.from_numpy(seq_arr).unsqueeze(0).float() # [1, 6, 72, 116]

        # Verify no NaN or Inf
        if torch.isnan(t0_tensor).any() or torch.isinf(t0_tensor).any():
            raise SatelliteIngestionError("NUMERICAL_INSTABILITY", "NaN or Inf detected in t0 satellite tensor")
        if torch.isnan(seq_tensor).any() or torch.isinf(seq_tensor).any():
            raise SatelliteIngestionError("NUMERICAL_INSTABILITY", "NaN or Inf detected in sequence satellite tensor")

        return {
            "t0_utc": t0_dt.isoformat(),
            "status": "READY",
            "t0_tensor": t0_tensor,
            "seq_tensor": seq_tensor,
            "frames": frames_meta,
            "shape_t0": list(t0_tensor.shape),
            "shape_seq": list(seq_tensor.shape),
        }

    def _process_and_cache_raw_netcdf(self, raw_path: Path, dt: datetime) -> Optional[Path]:
        """
        Processes a raw global NetCDF-4 GridSat file, decodes physical Kelvin,
        crops to the North Indian Ocean basin (572 x 929), and caches as .npz and .nc.
        """
        try:
            import netCDF4
            year_str = str(dt.year)
            out_dir_year = self.gridsat_dir / year_str
            out_dir_year.mkdir(parents=True, exist_ok=True)

            base_name = f"gridsat_{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}"
            out_nc = out_dir_year / f"{base_name}.nc"
            out_npz = out_dir_year / f"{base_name}.npz"

            if out_npz.exists():
                return out_npz

            nc = netCDF4.Dataset(raw_path, "r")
            nc.set_auto_maskandscale(False)

            if "irwin_cdr" not in nc.variables:
                nc.close()
                return None

            lat = nc.variables["lat"][:]
            lon = nc.variables["lon"][:]

            lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
            lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

            cropped_lat = lat[lat_mask]
            cropped_lon = lon[lon_mask]

            var = nc.variables["irwin_cdr"]
            scale_factor = getattr(var, "scale_factor", 0.01)
            add_offset = getattr(var, "add_offset", 200.0)
            fill_val = getattr(var, "_FillValue", -31999)
            miss_val = getattr(var, "missing_value", -31999)

            raw_slice = var[0, lat_mask, :][:, lon_mask]
            nc.close()

            is_missing = (raw_slice == fill_val) | (raw_slice == miss_val) | (raw_slice <= -30000)
            decoded = raw_slice.astype(np.float32) * float(scale_factor) + float(add_offset)
            decoded[is_missing] = np.nan

            # Save cropped .npz
            np.savez_compressed(out_npz, irwin_cdr=decoded, lat=cropped_lat, lon=cropped_lon)
            return out_npz

        except Exception as e:
            logger.warning(f"Failed to process and crop raw NetCDF {raw_path}: {e}")
            return None

    def fetch_required_causal_sequence(
        self,
        t0_dt: datetime,
    ) -> Dict[str, Any]:
        """
        Operational acquisition workflow:
        1. Checks current availability of 6 causal frames [t-15h ... t0].
        2. For any missing frames, invokes LiveSourceService to attempt download from configured provider (e.g. NOAA AWS S3).
        3. If raw NetCDF is downloaded, validates integrity, decodes Kelvin, crops to NIO domain, and saves to interim cache.
        4. Re-evaluates readiness; if all 6 frames present, ingests sequence and returns normalized tensors.
        5. If still incomplete, returns readiness dictionary with WAITING_FOR_FRAMES / SOURCE_UNAVAILABLE without crashing.
        6. Strictly NEVER downloads future frames (t > t0) and NEVER interpolates missing frames.
        """
        expected = self.get_expected_frame_timestamps(t0_dt)
        for step_name, offset_h, f_dt in expected:
            if f_dt > t0_dt:
                raise SatelliteIngestionError(
                    "FUTURE_FRAME_VIOLATION",
                    f"Frame {step_name} ({f_dt.isoformat()}) exceeds issue timestamp t0 ({t0_dt.isoformat()})."
                )

            # Check if local file exists
            local_file = self.find_gridsat_file_for_timestamp(f_dt)
            if local_file is None:
                # Attempt acquisition from LiveSourceService
                try:
                    from apps.backend.services.live_source_service import get_live_source_service
                    live_source = get_live_source_service()
                    success, fetched_path, msg = live_source.get_satellite_observation(f_dt.isoformat())
                    if success and fetched_path and fetched_path.exists():
                        # If fetched path is raw NetCDF, process and crop to interim
                        if fetched_path.suffix == ".nc" and "data/raw" in str(fetched_path).replace("\\", "/"):
                            self._process_and_cache_raw_netcdf(fetched_path, f_dt)
                except Exception as e:
                    logger.warning(f"Could not acquire satellite frame for {f_dt.isoformat()} via live provider: {e}")

        # Check sequence readiness
        readiness = self.check_sequence_readiness(t0_dt)
        if not readiness["is_complete"]:
            return {
                "t0_utc": t0_dt.isoformat(),
                "status": readiness["status"],
                "readiness": readiness,
                "t0_tensor": None,
                "seq_tensor": None,
                "frames": readiness["available_frames"],
            }

        # Ingest and return normalized tensors
        ingested = self.ingest_causal_sequence(t0_dt)
        ingested["readiness"] = readiness
        return ingested
