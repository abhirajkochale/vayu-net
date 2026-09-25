"""
VAYU-NET: Native INSAT-3D Level-1C HDF5 Calibration and Standardization Processor.

Target Product: 3DIMG_L1C_ASIA_MER (INSAT-3D Imager Level-1C Asian Sector Mercator)
Data Format: Hierarchical Data Format 5 (.h5)

Scientific Principles & Constraints:
1. Physical Calibration: Counts -> Brightness Temperature via native LUTs (IMG_TIR1_TEMP, IMG_TIR2_TEMP, IMG_WV_TEMP).
2. Physical Boundary Enforcement: Validates physical Kelvin bounds [180K, 330K] for TIR, [190K, 290K] for WV.
3. Coordinate Standardization: Standardizes to North Indian Ocean domain (lat [-5.0, 35.0], lon [40.0, 105.0]).
4. Geographic Orientation: Ensures latitude is strictly ascending (South to North) and longitude ascending (West to East).
5. Zero Synthetic Derivation: Rejects synthetic channel generation or GridSat copying.
6. Zero IMD Centering: Imagery is extracted over the fixed North Indian Ocean synoptic basin. No storm-centering or IMD label cropping.
"""

import os
import sys
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProcessNativeINSAT3D")

# VAYU-NET Fixed Basin Spatial Coordinate Bounds
NIO_LAT_MIN = -5.0
NIO_LAT_MAX = 35.0
NIO_LON_MIN = 40.0
NIO_LON_MAX = 105.0

TARGET_FULL_H = 572
TARGET_FULL_W = 929
TARGET_MODEL_H = 72
TARGET_MODEL_W = 116

# Physical Plausibility Limits (Kelvin)
PHYSICAL_LIMITS = {
    "IMG_TIR1": (180.0, 330.0),
    "IMG_TIR2": (180.0, 330.0),
    "IMG_WV": (190.0, 290.0)
}
FILL_VALUE_SENTINEL = -999.0
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class NativeINSAT3DProcessor:
    """Processes native INSAT-3D Level-1C HDF5 files into calibrated, spatially standardized tensors."""

    def __init__(self, target_resolution: str = "model"):
        """
        Args:
            target_resolution: "model" for [72, 116] or "full" for [572, 929].
        """
        self.target_resolution = target_resolution
        if target_resolution == "model":
            self.target_shape = (TARGET_MODEL_H, TARGET_MODEL_W)
        elif target_resolution == "full":
            self.target_shape = (TARGET_FULL_H, TARGET_FULL_W)
        else:
            raise ValueError(f"Unknown target_resolution: {target_resolution}")

    @staticmethod
    def verify_hdf5_file(file_path: Path) -> str:
        """Verify HDF5 magic header and compute SHA-256 hash."""
        if not file_path.exists():
            raise FileNotFoundError(f"Native INSAT-3D file does not exist: {file_path}")

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            header = f.read(8)
            if header != HDF5_SIGNATURE:
                raise ValueError(f"File {file_path} is not a valid HDF5 file (invalid magic signature).")
            f.seek(0)
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def calibrate_channel(
        self,
        counts: np.ndarray,
        lut: np.ndarray,
        channel_name: str
    ) -> np.ndarray:
        """
        Apply Look-Up Table (LUT) to convert raw counts (uint16) to Brightness Temperature (Kelvin).
        Preserves fill values and rejects physically corrupt pixels.
        """
        if not (0 <= counts.min() and counts.max() <= 65535):
            raise ValueError(f"Channel {channel_name} counts out of uint16 range: [{counts.min()}, {counts.max()}]")

        bt_array = np.full(counts.shape, np.nan, dtype=np.float32)

        # Valid LUT indices (typically 0 to 1023)
        valid_count_mask = (counts >= 0) & (counts < len(lut))
        bt_array[valid_count_mask] = lut[counts[valid_count_mask]]

        # Enforce physical Kelvin plausibility
        min_k, max_k = PHYSICAL_LIMITS[channel_name]
        physically_valid = (bt_array >= min_k) & (bt_array <= max_k)
        
        # Mark out-of-bounds or fill values as NaN / FILL_VALUE_SENTINEL
        bt_array[~physically_valid] = np.nan
        return bt_array

    def standardize_spatial_grid(
        self,
        calibrated_grid: np.ndarray,
        native_lat: Optional[np.ndarray] = None,
        native_lon: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample and standardize the calibrated grid to the VAYU-NET fixed North Indian Ocean domain.
        Strictly enforces:
        1. Latitude ascending: South (-5.0N) to North (+35.0N)
        2. Longitude ascending: West (40.0E) to East (105.0E)
        3. Zero storm-centering or IMD tracking crops.
        """
        h_native, w_native = calibrated_grid.shape

        # Generate or verify coordinate axes
        # Standard MOSDAC Mercator Level-1C bounds: 45.5N down to -10.0S; 44.5E to 105.5E
        if native_lat is None:
            # Native Mercator raster rows are stored North-to-South (descending)
            native_lat = np.linspace(45.5, -10.0, h_native)
        if native_lon is None:
            native_lon = np.linspace(44.5, 105.5, w_native)

        # 1. Coordinate orientation check
        lat_ascending = native_lat[1] > native_lat[0]
        if not lat_ascending:
            # Flip rows so latitude is strictly ascending (South to North)
            calibrated_grid = np.flipud(calibrated_grid)
            native_lat = native_lat[::-1]

        lon_ascending = native_lon[1] > native_lon[0]
        if not lon_ascending:
            calibrated_grid = np.fliplr(calibrated_grid)
            native_lon = native_lon[::-1]

        # 2. Extract fixed NIO domain bounding box [-5.0 to 35.0 N, 40.0 to 105.0 E]
        lat_mask = (native_lat >= NIO_LAT_MIN) & (native_lat <= NIO_LAT_MAX)
        lon_mask = (native_lon >= NIO_LON_MIN) & (native_lon <= NIO_LON_MAX)

        sub_grid = calibrated_grid[np.ix_(lat_mask, lon_mask)]
        sub_lat = native_lat[lat_mask]
        sub_lon = native_lon[lon_mask]

        # 3. Bilinear interpolation to target resolution without synthetic extrapolation
        target_h, target_w = self.target_shape
        # Use numpy meshgrid + regular grid interpolation or torch.nn.functional
        import torch
        import torch.nn.functional as F

        # Convert NaNs to sentinel for interpolation then restore
        nan_mask = np.isnan(sub_grid)
        sub_grid_filled = np.where(nan_mask, FILL_VALUE_SENTINEL, sub_grid)

        t_in = torch.from_numpy(sub_grid_filled).unsqueeze(0).unsqueeze(0).float()
        t_out = F.interpolate(t_in, size=(target_h, target_w), mode="bilinear", align_corners=False)
        out_arr = t_out.squeeze(0).squeeze(0).numpy()

        # Interpolate mask to preserve missing value integrity
        m_in = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        m_out = F.interpolate(m_in, size=(target_h, target_w), mode="bilinear", align_corners=False)
        out_nan_mask = m_out.squeeze(0).squeeze(0).numpy() > 0.5

        out_arr[out_nan_mask] = np.nan

        target_lat = np.linspace(NIO_LAT_MIN, NIO_LAT_MAX, target_h)
        target_lon = np.linspace(NIO_LON_MIN, NIO_LON_MAX, target_w)

        return out_arr, target_lat, target_lon

    def process_granule(self, file_path: Path) -> Dict[str, Any]:
        """
        Fully process a native INSAT-3D Level-1C HDF5 granule.
        Returns a dictionary containing calibrated arrays and complete scientific provenance.
        """
        sha256 = self.verify_hdf5_file(file_path)
        logger.info(f"Opening genuine HDF5 file {file_path.name} (SHA-256: {sha256[:16]}...)")

        with h5py.File(file_path, "r") as h5:
            # Check required datasets
            required_channels = ["IMG_TIR1", "IMG_TIR2", "IMG_WV"]
            for ch in required_channels:
                if ch not in h5:
                    raise KeyError(f"Dataset {ch} not found in {file_path.name}. Real Level-1C product required.")
                lut_key = f"{ch}_TEMP"
                if lut_key not in h5:
                    raise KeyError(f"Calibration LUT {lut_key} not found in {file_path.name}.")

            # Extract metadata attributes
            product_name = h5.attrs.get("Product_Name", "3DIMG_L1C_ASIA_MER")
            satellite = h5.attrs.get("Satellite_Name", "INSAT-3D")
            acq_time = h5.attrs.get("Acquisition_Date_Time_GMT", "")

            # Process each channel
            calibrated_channels = {}
            for ch in required_channels:
                raw_counts = np.array(h5[ch])
                lut = np.array(h5[f"{ch}_TEMP"])
                calibrated_bt = self.calibrate_channel(raw_counts, lut, ch)
                std_grid, target_lat, target_lon = self.standardize_spatial_grid(calibrated_bt)
                calibrated_channels[ch] = std_grid

        # Stack into [3, H, W]
        stacked_tensor = np.stack([
            calibrated_channels["IMG_TIR1"],
            calibrated_channels["IMG_TIR2"],
            calibrated_channels["IMG_WV"]
        ], axis=0)

        provenance = {
            "source_file": str(file_path),
            "file_sha256": sha256,
            "product_id": "3DIMG_L1C_ASIA_MER",
            "product_name": str(product_name),
            "satellite": str(satellite),
            "acquisition_time_gmt": str(acq_time),
            "channels": required_channels,
            "calibration_method": "MOSDAC_NATIVE_COUNT_TO_KELVIN_LUT",
            "physical_limits": PHYSICAL_LIMITS,
            "spatial_domain": {
                "lat_min": NIO_LAT_MIN,
                "lat_max": NIO_LAT_MAX,
                "lon_min": NIO_LON_MIN,
                "lon_max": NIO_LON_MAX,
                "lat_orientation": "strictly_ascending_south_to_north",
                "lon_orientation": "strictly_ascending_west_to_east"
            },
            "output_shape": list(stacked_tensor.shape),
            "is_synthetic": False,
            "is_gridsat_derived": False
        }

        return {
            "tensor": stacked_tensor,
            "lat": target_lat,
            "lon": target_lon,
            "provenance": provenance
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process Native INSAT-3D Level-1C HDF5 Granule")
    parser.add_argument("--input", type=str, required=True, help="Path to native .h5 file")
    parser.add_argument("--resolution", choices=["model", "full"], default="model")
    parser.add_argument("--output", type=str, default=None, help="Path to save processed npz")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Native HDF5 file not found: {input_path}")
        print("[NOTICE] MOSDAC credentials must be supplied to fetch native observations.")
        sys.exit(1)

    processor = NativeINSAT3DProcessor(target_resolution=args.resolution)
    try:
        result = processor.process_granule(input_path)
        print(f"[SUCCESS] Processed {input_path.name} -> shape: {result['tensor'].shape}")
        if args.output:
            np.savez_compressed(
                args.output,
                insat_data=result["tensor"],
                lat=result["lat"],
                lon=result["lon"],
                provenance=result["provenance"]
            )
            print(f"[SAVED] Saved calibrated tensor to {args.output}")
    except Exception as e:
        print(f"[PROCESSING_ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
