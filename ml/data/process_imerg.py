"""
VAYU-NET: NASA GPM IMERG Final Run V07B Processing & Standardization Pipeline.

Target Product: NASA GPM IMERG Final Run V07B (Half-Hourly, 0.1 degree)
Primary Modality: Calibrated Surface Precipitation Rate (mm/hr)
HDF5 Path: /Grid/precipitation (formerly precipitationCal)

Scientific Constraints & Guardrails:
1. Observational Integrity: Strict read of native radiometer-gauge calibrated precipitation.
2. Fill Value Policy: Fill values (-9999.9) are strictly converted to NaN.
3. No Interpolation of Missing Data: Native missing data is preserved as NaN.
4. No IMD Storm Centering: Standardized strictly to the fixed North Indian Ocean synoptic basin
   (lat [-5.0, 35.0], lon [40.0, 105.0]).
5. Provenance Retention: Every processed output carries SHA-256, granule ID, units, and spatial metadata.
6. Zero Synthetic Generation: Rejects any synthetic fallback.
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
logger = logging.getLogger("ProcessIMERG")

# Fixed VAYU-NET North Indian Ocean Domain Coordinates
NIO_LAT_MIN = -5.0
NIO_LAT_MAX = 35.0
NIO_LON_MIN = 40.0
NIO_LON_MAX = 105.0

# Target Model Grid Dimensions
TARGET_MODEL_H = 72
TARGET_MODEL_W = 116
TARGET_FULL_H = 572
TARGET_FULL_W = 929

# IMERG Product Metadata Constants
IMERG_FILL_VALUE = -9999.9
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class IMERGProcessor:
    """Processes genuine NASA GPM IMERG Final Run V07B HDF5 granules into standardized analysis arrays."""

    def __init__(self, target_resolution: str = "native_subgrid"):
        """
        Args:
            target_resolution: "native_subgrid" for (400, 650), "model" for (72, 116), or "full" for (572, 929).
        """
        self.target_resolution = target_resolution

    @staticmethod
    def verify_hdf5_file(file_path: Path) -> str:
        """Verify HDF5 magic header and compute SHA-256 hash."""
        if not file_path.exists():
            raise FileNotFoundError(f"IMERG file not found: {file_path}")

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            header = f.read(8)
            if header != HDF5_SIGNATURE:
                raise ValueError(f"File {file_path} is not a valid HDF5 file (invalid magic signature).")
            f.seek(0)
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract_precipitation_variable(self, grid_group: h5py.Group) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """
        Locate and extract the primary calibrated precipitation rate variable.
        In IMERG V07B, the field is '/Grid/precipitation' (formerly 'precipitationCal').
        """
        candidate_keys = ["precipitation", "precipitationCal"]
        var_name = None
        for key in candidate_keys:
            if key in grid_group:
                var_name = key
                break

        if var_name is None:
            available = list(grid_group.keys())
            raise KeyError(f"Neither 'precipitation' nor 'precipitationCal' found in /Grid. Available: {available}")

        ds = grid_group[var_name]
        raw_data = ds[0] if ds.ndim == 3 else ds[:]  # Squeeze time dimension (1, lon, lat) -> (lon, lat)

        attrs = {k: str(ds.attrs[k]) for k in ds.attrs}
        units = attrs.get("Units", attrs.get("units", "mm/hr"))
        return raw_data, var_name, attrs

    def process_granule(self, file_path: Path, verify_sha: bool = True) -> Dict[str, Any]:
        """
        Full processing pipeline for an IMERG Final Run V07B HDF5 file:
        1. Verifies HDF5 magic bytes and computes SHA-256 (if verify_sha=True).
        2. Extracts '/Grid/precipitation' (or 'precipitationCal') and coordinate vectors.
        3. Transposes (lon, lat) to standard image spatial orientation (lat, lon).
        4. Replaces fill values (-9999.9) with NaN without interpolating missing data.
        5. Subsets strictly to the fixed NIO basin [-5, 35] lat, [40, 105] lon without storm-centering.
        6. Preserves provenance attributes.
        """
        if verify_sha:
            sha256 = self.verify_hdf5_file(file_path)
            logger.info(f"Opening genuine IMERG HDF5: {file_path.name} (SHA-256: {sha256[:16]}...)")
        else:
            sha256 = ""

        with h5py.File(file_path, "r") as h5:
            if "Grid" not in h5:
                raise KeyError(f"Root group /Grid not found in {file_path.name}. Genuine GPM IMERG product required.")
            grid = h5["Grid"]

            # Coordinates
            if "lat" not in grid or "lon" not in grid:
                raise KeyError(f"Coordinates lat/lon missing from /Grid in {file_path.name}.")
            native_lat = np.array(grid["lat"][:], dtype=np.float32)
            native_lon = np.array(grid["lon"][:], dtype=np.float32)

            # Precipitation variable
            raw_precip_lonlat, var_name, var_attrs = self.extract_precipitation_variable(grid)

            # Metadata header extraction
            file_header = h5.attrs.get("FileHeader", b"").decode("utf-8", errors="ignore") if isinstance(h5.attrs.get("FileHeader"), bytes) else str(h5.attrs.get("FileHeader", ""))

        # 1. Transpose from native [lon, lat] (3600, 1800) to standard [lat, lon] (1800, 3600)
        precip_latlon = np.transpose(raw_precip_lonlat, (1, 0)).astype(np.float32)

        # 2. Convert fill values to NaN according to official product specification
        # Valid physical precipitation is >= 0.0 mm/hr. Values < 0.0 (e.g. -9999.9) represent missing/fill.
        precip_clean = np.where(precip_latlon >= 0.0, precip_latlon, np.nan)

        # 3. Spatial standardization to fixed North Indian Ocean basin (no storm centering)
        lat_mask = (native_lat >= NIO_LAT_MIN) & (native_lat <= NIO_LAT_MAX)
        lon_mask = (native_lon >= NIO_LON_MIN) & (native_lon <= NIO_LON_MAX)

        sub_precip = precip_clean[np.ix_(lat_mask, lon_mask)]
        sub_lat = native_lat[lat_mask]
        sub_lon = native_lon[lon_mask]

        # 4. Verify coordinate orientation
        assert sub_lat[1] > sub_lat[0], "Latitude must be strictly ascending (South to North)"
        assert sub_lon[1] > sub_lon[0], "Longitude must be strictly ascending (West to East)"

        # 5. Handle optional resampling if requested
        if self.target_resolution == "model":
            # Resample to [72, 116] using bilinear interpolation with NaN preservation
            import torch
            import torch.nn.functional as F

            nan_mask = np.isnan(sub_precip)
            filled = np.where(nan_mask, 0.0, sub_precip)

            t_in = torch.from_numpy(filled).unsqueeze(0).unsqueeze(0).float()
            t_out = F.interpolate(t_in, size=(TARGET_MODEL_H, TARGET_MODEL_W), mode="bilinear", align_corners=False)
            resampled_arr = t_out.squeeze(0).squeeze(0).numpy()

            m_in = torch.from_numpy(nan_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            m_out = F.interpolate(m_in, size=(TARGET_MODEL_H, TARGET_MODEL_W), mode="bilinear", align_corners=False)
            resampled_nan = m_out.squeeze(0).squeeze(0).numpy() > 0.5
            resampled_arr[resampled_nan] = np.nan

            final_precip = resampled_arr
            final_lat = np.linspace(NIO_LAT_MIN, NIO_LAT_MAX, TARGET_MODEL_H)
            final_lon = np.linspace(NIO_LON_MIN, NIO_LON_MAX, TARGET_MODEL_W)
        else:
            final_precip = sub_precip
            final_lat = sub_lat
            final_lon = sub_lon

        provenance = {
            "source_file": str(file_path),
            "file_sha256": sha256,
            "product_collection": "GPM_3IMERGHH.07",
            "algorithm_version": "V07B",
            "variable_extracted": var_name,
            "physical_units": "mm/hr",
            "native_spatial_resolution_deg": 0.1,
            "native_subgrid_shape": [int(np.sum(lat_mask)), int(np.sum(lon_mask))],
            "output_shape": list(final_precip.shape),
            "spatial_domain": {
                "lat_min": float(sub_lat.min()),
                "lat_max": float(sub_lat.max()),
                "lon_min": float(sub_lon.min()),
                "lon_max": float(sub_lon.max()),
                "lat_orientation": "strictly_ascending_south_to_north",
                "lon_orientation": "strictly_ascending_west_to_east",
                "storm_centered_cropping": False,
                "basin": "North Indian Ocean Fixed Synoptic Domain"
            },
            "statistics": {
                "min_mm_hr": float(np.nanmin(final_precip)),
                "max_mm_hr": float(np.nanmax(final_precip)),
                "mean_mm_hr": float(np.nanmean(final_precip)),
                "nan_fraction": float(np.isnan(final_precip).mean())
            },
            "is_synthetic": False,
            "is_gridsat_derived": False
        }

        return {
            "precipitation": final_precip,
            "lat": final_lat,
            "lon": final_lon,
            "provenance": provenance
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process NASA GPM IMERG Final Run V07B HDF5 Granule")
    parser.add_argument("--input", type=str, required=True, help="Path to genuine IMERG .HDF5 file")
    parser.add_argument("--resolution", choices=["native_subgrid", "model"], default="native_subgrid")
    parser.add_argument("--output", type=str, default=None, help="Optional output .npz path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    processor = IMERGProcessor(target_resolution=args.resolution)
    try:
        res = processor.process_granule(input_path)
        stats = res["provenance"]["statistics"]
        print(f"[SUCCESS] Processed {input_path.name}")
        print(f"  Shape: {res['precipitation'].shape}")
        print(f"  Lat bounds: [{res['lat'].min():.2f}, {res['lat'].max():.2f}] deg N")
        print(f"  Lon bounds: [{res['lon'].min():.2f}, {res['lon'].max():.2f}] deg E")
        print(f"  Precipitation stats: min={stats['min_mm_hr']:.2f}, max={stats['max_mm_hr']:.2f}, mean={stats['mean_mm_hr']:.4f} mm/hr")

        if args.output:
            np.savez_compressed(
                args.output,
                precipitation=res["precipitation"],
                lat=res["lat"],
                lon=res["lon"],
                provenance=res["provenance"]
            )
            print(f"[SAVED] Exported calibrated array to {args.output}")
    except Exception as e:
        print(f"[PROCESSING_ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
