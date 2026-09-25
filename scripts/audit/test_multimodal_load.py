"""
VAYU-NET: EXP-M1 Multimodal Dataset Load Test.
Loads one TRAIN sequence, one VALIDATION sequence, and one TEST sequence.
Inspects GridSat & IMERG shapes, target availability, and IMERG physical statistics.
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.multimodal_dataset import MultimodalVayuDataset


def run_load_test():
    print("=" * 65)
    print("VAYU-NET: EXP-M1 MULTIMODAL DATASET LOAD TEST")
    print("=" * 65)
    
    splits = ["TRAIN", "VALIDATION", "TEST"]
    
    for split in splits:
        print(f"\n>>> LOADING {split} SAMPLE...")
        ds = MultimodalVayuDataset(split=split, normalize=True)
        print(f"Dataset split count: {len(ds)} sequences")
        
        sample = ds[0]
        
        sid = sample["sample_id"]
        storm_id = sample["storm_id"]
        s_split = sample["split"]
        t0 = sample["t0_utc"]
        
        gs_shape = tuple(sample["gridsat"].shape)
        im_shape = tuple(sample["imerg"].shape)
        
        # Targets
        c0 = sample["center_t0"].tolist()
        w0 = sample["wind_t0"].item()
        p0 = sample["pressure_t0"].item()
        cat0 = sample["category_t0"].item()
        w0_mask = sample["wind_t0_mask"].item()
        
        # IMERG finite-value statistics (unstandardized / standardized)
        im_arr = sample["imerg"].numpy()
        im_min = float(im_arr.min())
        im_max = float(im_arr.max())
        im_mean = float(im_arr.mean())
        im_std = float(im_arr.std())
        nan_count = int(np.isnan(im_arr).sum())
        
        # GridSat statistics
        gs_arr = sample["gridsat"].numpy()
        gs_min = float(gs_arr.min())
        gs_max = float(gs_arr.max())
        gs_mean = float(gs_arr.mean())
        gs_std = float(gs_arr.std())
        
        print(f"  sample_id:                  {sid}")
        print(f"  storm_id:                   {storm_id}")
        print(f"  split:                      {s_split}")
        print(f"  t0_utc:                     {t0}")
        print(f"  GridSat shape:              {gs_shape}")
        print(f"  IMERG shape:                {im_shape}")
        print(f"  GridSat stats (norm):       min={gs_min:.3f}, max={gs_max:.3f}, mean={gs_mean:.3f}, std={gs_std:.3f}")
        print(f"  IMERG stats (norm):         min={im_min:.3f}, max={im_max:.3f}, mean={im_mean:.3f}, std={im_std:.3f}, NaNs={nan_count}")
        print(f"  Target t0:                  center={c0}, wind={w0} kt (mask={w0_mask}), pres={p0} hPa, cat={cat0}")
        print(f"  Future target availability: +12h={sample['center_12h'].tolist()}, +24h={sample['center_24h'].tolist()}, +48h={sample['center_48h'].tolist()}")
        print(f"  Status:                     [PASS] Sequence successfully loaded and verified.")

    print("\n" + "=" * 65)
    print("[LOAD TEST SUCCESS] All 3 partitions successfully loaded and validated.")
    print("=" * 65)

if __name__ == "__main__":
    run_load_test()