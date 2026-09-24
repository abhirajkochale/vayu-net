"""
VAYU-NET PHASE 5B — MERGED MULTIMODAL HYBRID FEATURE CACHE BUILDER
===================================================================
Merges:
1. Phase 4B Hybrid Cache (Satellite features [6, 132], kinematics A/B, residuals A/B)
2. Phase 5A Environmental Cache (ERA5 normalized wind sequences [6, 8, 41, 66])

Produces:
  data/interim/ml/cache/phase5b_hybrid_features.pt
"""

import os
import sys
import torch
import numpy as np

C_4B_PATH = "data/interim/ml/cache/phase4b_hybrid_features.pt"
C_5A_PATH = "data/interim/ml/cache/era5_environment_features.pt"
OUT_PATH = "data/interim/ml/cache/phase5b_hybrid_features.pt"

def build_phase5b_cache():
    print("=" * 80)
    print("BUILDING PHASE 5B HYBRID MULTIMODAL FEATURE CACHE")
    print("=" * 80)
    
    print(f"Loading Phase 4B features from {C_4B_PATH}...")
    c_4b = torch.load(C_4B_PATH, map_location='cpu', weights_only=False)
    print(f"Loading Phase 5A features from {C_5A_PATH}...")
    c_5a = torch.load(C_5A_PATH, map_location='cpu', weights_only=False)
    
    # Map ERA5 samples by sample_id
    era5_map = {}
    for split in ['TRAIN', 'VALIDATION', 'TEST']:
        for item in c_5a['samples'][split]:
            era5_map[item['sample_id']] = item
            
    merged_samples = []
    
    for item4 in c_4b['samples']:
        sid = item4['sample_id']
        item5 = era5_map[sid]
        
        # Check masks
        t12 = item4['true_12']
        t24 = item4['true_24']
        t48 = item4['true_48']
        
        m12 = 1.0 if not (torch.isnan(t12[0]) or torch.isnan(t12[1])) else 0.0
        m24 = 1.0 if not (torch.isnan(t24[0]) or torch.isnan(t24[1])) else 0.0
        m48 = 1.0 if not (torch.isnan(t48[0]) or torch.isnan(t48[1])) else 0.0
        
        merged_item = {
            'sample_id': sid,
            'storm_id': item4['storm_id'],
            'split': item4['split'],
            't0': item4['t0'],
            'sat_seq': item4['feature_sequence'].float(), # [6, 132]
            'env_seq': item5['env_seq'].float(),          # [6, 8, 41, 66]
            
            # Variant A (Exact-Center Kinematic Anchor & Residuals)
            'kin_ctx_a': item4['kin_ctx_a'].float(),       # [5]
            'kin_a_12': item4['kin_a_12'].float(),         # [2]
            'kin_a_24': item4['kin_a_24'].float(),         # [2]
            'kin_a_48': item4['kin_a_48'].float(),         # [2]
            'res_a_12': item4['res_a_12'].float(),         # [2] km
            'res_a_24': item4['res_a_24'].float(),         # [2] km
            'res_a_48': item4['res_a_48'].float(),         # [2] km
            
            # Variant B (Satellite-Derived Kinematic Anchor & Residuals)
            'kin_ctx_b': item4['kin_ctx_b'].float(),       # [5]
            'kin_b_12': item4['kin_b_12'].float(),         # [2]
            'kin_b_24': item4['kin_b_24'].float(),         # [2]
            'kin_b_48': item4['kin_b_48'].float(),         # [2]
            'res_b_12': item4['res_b_12'].float(),         # [2] km
            'res_b_24': item4['res_b_24'].float(),         # [2] km
            'res_b_48': item4['res_b_48'].float(),         # [2] km
            
            # True Targets & Validity Masks
            'true_12': t12.float(),
            'true_24': t24.float(),
            'true_48': t48.float(),
            'mask_12': torch.tensor(m12, dtype=torch.float32),
            'mask_24': torch.tensor(m24, dtype=torch.float32),
            'mask_48': torch.tensor(m48, dtype=torch.float32),
        }
        merged_samples.append(merged_item)
        
    splits_count = {'TRAIN': 0, 'VALIDATION': 0, 'TEST': 0}
    for s in merged_samples:
        splits_count[s['split']] += 1
        
    payload = {
        'metadata': {
            'phase': 'Phase 5B',
            'description': 'Merged Multimodal Hybrid Feature Cache (GridSat Satellite + ERA5 Environmental Wind + Kinematic Context)',
            'total_samples': len(merged_samples),
            'splits_count': splits_count,
            'sat_feature_dim': 132,
            'env_shape': [6, 8, 41, 66],
            'kin_ctx_dim': 5,
            'source_caches': [C_4B_PATH, C_5A_PATH]
        },
        'samples': merged_samples
    }
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(payload, OUT_PATH)
    file_size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Successfully saved Phase 5B cache to {OUT_PATH} ({file_size_mb:.2f} MB)")
    print(f"  TRAIN: {splits_count['TRAIN']}")
    print(f"  VAL:   {splits_count['VALIDATION']}")
    print(f"  TEST:  {splits_count['TEST']}")
    print("=" * 80)

if __name__ == '__main__':
    build_phase5b_cache()
