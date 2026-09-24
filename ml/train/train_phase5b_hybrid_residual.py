"""
VAYU-NET PHASE 5B — ENVIRONMENT-AWARE HYBRID RESIDUAL TRAINING PIPELINE
========================================================================
Executes:
  1. Experiment 5B-A: Observed-Center Hybrid Residual (Variant A)
     - Exact IMD past centers -> Exact Kinematic Anchor
     - Satellite + ERA5 atmospheric wind sequence -> Learned residual
     - Selected by: Minimum VALIDATION Mean Track DPE
     - Checkpoint: data/interim/ml/checkpoints/best_phase5b_variant_a.pt
  2. Experiment 5B-B: Satellite-Derived Operational Hybrid Residual (Variant B)
     - Phase 3C estimated past centers -> Satellite-Derived Kinematic Anchor
     - Satellite + ERA5 atmospheric wind sequence -> Learned residual
     - Selected by: Minimum VALIDATION Mean Track DPE
     - Checkpoint: data/interim/ml/checkpoints/best_phase5b_variant_b.pt
  3. Single-Pass Test Evaluation on Held-Out TEST partition strictly once.
  4. Full results compiled into data/interim/ml/phase5b_hybrid_results.json
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.phase5b_hybrid_residual import Phase5BHybridResidualModel, RESIDUAL_SCALE_KM

CACHE_PATH = "data/interim/ml/cache/phase5b_hybrid_features.pt"
CHECKPOINT_DIR = "data/interim/ml/checkpoints"
CKPT_PATH_A = os.path.join(CHECKPOINT_DIR, "best_phase5b_variant_a.pt")
CKPT_PATH_B = os.path.join(CHECKPOINT_DIR, "best_phase5b_variant_b.pt")
RESULTS_PATH = "data/interim/ml/phase5b_hybrid_results.json"

def haversine_dpe_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

class Phase5BDataset(Dataset):
    def __init__(self, cache_data, split='TRAIN', variant='A'):
        self.items = [s for s in cache_data['samples'] if s['split'] == split]
        self.variant = variant

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sat_seq = item['sat_seq'] # [6, 132]
        env_seq = item['env_seq'] # [6, 8, 41, 66]
        
        if self.variant == 'A':
            motion_ctx = item['kin_ctx_a'] # [5]
            k12 = item['kin_a_12']
            k24 = item['kin_a_24']
            k48 = item['kin_a_48']
            r12_km = item['res_a_12']
            r24_km = item['res_a_24']
            r48_km = item['res_a_48']
        else:
            motion_ctx = item['kin_ctx_b'] # [5]
            k12 = item['kin_b_12']
            k24 = item['kin_b_24']
            k48 = item['kin_b_48']
            r12_km = item['res_b_12']
            r24_km = item['res_b_24']
            r48_km = item['res_b_48']
            
        r12_norm = r12_km / RESIDUAL_SCALE_KM
        r24_norm = r24_km / RESIDUAL_SCALE_KM
        r48_norm = r48_km / RESIDUAL_SCALE_KM
        
        true_12 = item['true_12']
        true_24 = item['true_24']
        true_48 = item['true_48']
        
        m12 = item['mask_12']
        m24 = item['mask_24']
        m48 = item['mask_48']
        
        return (
            sat_seq, env_seq, motion_ctx,
            k12, k24, k48,
            r12_norm, r24_norm, r48_norm,
            true_12, true_24, true_48,
            m12, m24, m48,
            item['sample_id'], item['storm_id'], item['t0']
        )


def evaluate_variant(model, dataloader, device):
    model.eval()
    
    dpes_12 = []
    dpes_24 = []
    dpes_48 = []
    kin_dpes_12 = []
    kin_dpes_24 = []
    kin_dpes_48 = []
    
    res_mags_12 = []
    res_mags_24 = []
    res_mags_48 = []
    res_n_12 = []
    res_e_12 = []
    res_n_24 = []
    res_e_24 = []
    res_n_48 = []
    res_e_48 = []

    raw_data = {
        'sample_ids': [], 'storm_ids': [], 't0s': [],
        'pred_12': [], 'pred_24': [], 'pred_48': [],
        'kin_12': [], 'kin_24': [], 'kin_48': [],
        'true_12': [], 'true_24': [], 'true_48': [],
        'dpe_12': [], 'dpe_24': [], 'dpe_48': [],
        'kin_dpe_12': [], 'kin_dpe_24': [], 'kin_dpe_48': [],
        'track_dpes': [], 'kin_track_dpes': [],
        'r12_km': [], 'r24_km': [], 'r48_km': []
    }

    with torch.no_grad():
        for batch in dataloader:
            (sat_seq, env_seq, mot_ctx,
             k12, k24, k48,
             r12_tgt, r24_tgt, r48_tgt,
             t12, t24, t48,
             m12, m24, m48,
             sids, storms, t0s) = batch
             
            sat_seq = sat_seq.to(device)
            env_seq = env_seq.to(device)
            mot_ctx = mot_ctx.to(device)
            k12_dev = k12.to(device)
            k24_dev = k24.to(device)
            k48_dev = k48.to(device)
            
            out = model(sat_seq, env_seq, mot_ctx, k12_dev, k24_dev, k48_dev)
            
            p12 = out['pred_12'].cpu().numpy()
            p24 = out['pred_24'].cpu().numpy()
            p48 = out['pred_48'].cpu().numpy()
            
            r12_pred_km = out['r12_km'].cpu().numpy()
            r24_pred_km = out['r24_km'].cpu().numpy()
            r48_pred_km = out['r48_km'].cpu().numpy()

            k12_np = k12.numpy()
            k24_np = k24.numpy()
            k48_np = k48.numpy()

            t12_np = t12.numpy()
            t24_np = t24.numpy()
            t48_np = t48.numpy()

            m12_np = m12.numpy()
            m24_np = m24.numpy()
            m48_np = m48.numpy()

            for i in range(len(p12)):
                sid = sids[i]; st = storms[i]; t_zero = t0s[i]
                
                # Hybrid DPEs
                d12 = haversine_dpe_km(p12[i, 0], p12[i, 1], t12_np[i, 0], t12_np[i, 1]) if m12_np[i] > 0.5 else np.nan
                d24 = haversine_dpe_km(p24[i, 0], p24[i, 1], t24_np[i, 0], t24_np[i, 1]) if m24_np[i] > 0.5 else np.nan
                d48 = haversine_dpe_km(p48[i, 0], p48[i, 1], t48_np[i, 0], t48_np[i, 1]) if m48_np[i] > 0.5 else np.nan
                
                # Kinematic DPEs
                kd12 = haversine_dpe_km(k12_np[i, 0], k12_np[i, 1], t12_np[i, 0], t12_np[i, 1]) if m12_np[i] > 0.5 else np.nan
                kd24 = haversine_dpe_km(k24_np[i, 0], k24_np[i, 1], t24_np[i, 0], t24_np[i, 1]) if m24_np[i] > 0.5 else np.nan
                kd48 = haversine_dpe_km(k48_np[i, 0], k48_np[i, 1], t48_np[i, 0], t48_np[i, 1]) if m48_np[i] > 0.5 else np.nan

                val_ds = [d for d in [d12, d24, d48] if not np.isnan(d)]
                tr_d = np.mean(val_ds) if val_ds else np.nan
                
                val_kds = [kd for kd in [kd12, kd24, kd48] if not np.isnan(kd)]
                tr_kd = np.mean(val_kds) if val_kds else np.nan

                if not np.isnan(d12):
                    dpes_12.append(d12)
                    kin_dpes_12.append(kd12)
                    res_mag = np.linalg.norm(r12_pred_km[i])
                    res_mags_12.append(res_mag)
                    res_n_12.append(r12_pred_km[i, 0])
                    res_e_12.append(r12_pred_km[i, 1])
                    
                if not np.isnan(d24):
                    dpes_24.append(d24)
                    kin_dpes_24.append(kd24)
                    res_mag = np.linalg.norm(r24_pred_km[i])
                    res_mags_24.append(res_mag)
                    res_n_24.append(r24_pred_km[i, 0])
                    res_e_24.append(r24_pred_km[i, 1])

                if not np.isnan(d48):
                    dpes_48.append(d48)
                    kin_dpes_48.append(kd48)
                    res_mag = np.linalg.norm(r48_pred_km[i])
                    res_mags_48.append(res_mag)
                    res_n_48.append(r48_pred_km[i, 0])
                    res_e_48.append(r48_pred_km[i, 1])

                raw_data['sample_ids'].append(sid)
                raw_data['storm_ids'].append(st)
                raw_data['t0s'].append(t_zero)
                raw_data['pred_12'].append(p12[i].tolist())
                raw_data['pred_24'].append(p24[i].tolist())
                raw_data['pred_48'].append(p48[i].tolist())
                raw_data['kin_12'].append(k12_np[i].tolist())
                raw_data['kin_24'].append(k24_np[i].tolist())
                raw_data['kin_48'].append(k48_np[i].tolist())
                raw_data['true_12'].append(t12_np[i].tolist())
                raw_data['true_24'].append(t24_np[i].tolist())
                raw_data['true_48'].append(t48_np[i].tolist())
                raw_data['dpe_12'].append(d12)
                raw_data['dpe_24'].append(d24)
                raw_data['dpe_48'].append(d48)
                raw_data['kin_dpe_12'].append(kd12)
                raw_data['kin_dpe_24'].append(kd24)
                raw_data['kin_dpe_48'].append(kd48)
                raw_data['track_dpes'].append(tr_d)
                raw_data['kin_track_dpes'].append(tr_kd)
                raw_data['r12_km'].append(r12_pred_km[i].tolist())
                raw_data['r24_km'].append(r24_pred_km[i].tolist())
                raw_data['r48_km'].append(r48_pred_km[i].tolist())

    m12 = float(np.mean(dpes_12))
    med12 = float(np.median(dpes_12))
    p90_12 = float(np.percentile(dpes_12, 90))

    m24 = float(np.mean(dpes_24))
    med24 = float(np.median(dpes_24))
    p90_24 = float(np.percentile(dpes_24, 90))

    m48 = float(np.mean(dpes_48))
    med48 = float(np.median(dpes_48))
    p90_48 = float(np.percentile(dpes_48, 90))

    mean_track = float(np.mean([m12, m24, m48]))
    kin_mean_track = float(np.mean([np.mean(kin_dpes_12), np.mean(kin_dpes_24), np.mean(kin_dpes_48)]))

    return {
        'mean_track_dpe_km': mean_track,
        'kin_mean_track_dpe_km': kin_mean_track,
        'improvement_over_kin_mean_km': kin_mean_track - mean_track,
        '12h': {'mean': m12, 'median': med12, 'p90': p90_12, 'count': len(dpes_12)},
        '24h': {'mean': m24, 'median': med24, 'p90': p90_24, 'count': len(dpes_24)},
        '48h': {'mean': m48, 'median': med48, 'p90': p90_48, 'count': len(dpes_48)},
        'residuals': {
            '12h': {
                'mean_mag_km': float(np.mean(res_mags_12)),
                'median_mag_km': float(np.median(res_mags_12)),
                'bias_north_km': float(np.mean(res_n_12)),
                'bias_east_km': float(np.mean(res_e_12))
            },
            '24h': {
                'mean_mag_km': float(np.mean(res_mags_24)),
                'median_mag_km': float(np.median(res_mags_24)),
                'bias_north_km': float(np.mean(res_n_24)),
                'bias_east_km': float(np.mean(res_e_24))
            },
            '48h': {
                'mean_mag_km': float(np.mean(res_mags_48)),
                'median_mag_km': float(np.median(res_mags_48)),
                'bias_north_km': float(np.mean(res_n_48)),
                'bias_east_km': float(np.mean(res_e_48))
            }
        },
        'raw': raw_data
    }


def train_variant(variant_name, cache_data, ckpt_path, device, max_epochs=15, patience=4):
    print("=" * 80)
    print(f"TRAINING PHASE 5B: {variant_name}")
    print("=" * 80)
    
    variant_code = 'A' if 'Variant A' in variant_name else 'B'
    train_ds = Phase5BDataset(cache_data, 'TRAIN', variant_code)
    val_ds = Phase5BDataset(cache_data, 'VALIDATION', variant_code)
    test_ds = Phase5BDataset(cache_data, 'TEST', variant_code)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = Phase5BHybridResidualModel(
        sat_feat_dim=132,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        motion_in_dim=5,
        motion_emb_dim=16,
        fusion_dim=128,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    
    best_val_track_dpe = float('inf')
    best_epoch = -1
    best_val_metrics = None
    patience_counter = 0
    t_start = time.time()

    print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training loop (Model Selection strictly governed by VALIDATION Mean Track DPE)...")
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            (sat_seq, env_seq, mot_ctx,
             k12, k24, k48,
             r12_tgt, r24_tgt, r48_tgt,
             t12, t24, t48,
             m12, m24, m48,
             _, _, _) = batch
             
            sat_seq = sat_seq.to(device)
            env_seq = env_seq.to(device)
            mot_ctx = mot_ctx.to(device)
            r12_tgt = r12_tgt.to(device)
            r24_tgt = r24_tgt.to(device)
            r48_tgt = r48_tgt.to(device)
            m12 = m12.to(device)
            m24 = m24.to(device)
            m48 = m48.to(device)
            
            optimizer.zero_grad()
            out = model(sat_seq, env_seq, mot_ctx)
            
            r12_pred = out['r12_norm']
            r24_pred = out['r24_norm']
            r48_pred = out['r48_norm']
            
            l12 = (criterion(r12_pred, r12_tgt).sum(dim=-1) * m12).sum() / (m12.sum() + 1e-6)
            l24 = (criterion(r24_pred, r24_tgt).sum(dim=-1) * m24).sum() / (m24.sum() + 1e-6)
            l48 = (criterion(r48_pred, r48_tgt).sum(dim=-1) * m48).sum() / (m48.sum() + 1e-6)
            
            loss = l12 + l24 + l48
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        val_metrics = evaluate_variant(model, val_loader, device)
        val_track_dpe = val_metrics['mean_track_dpe_km']
        
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Track DPE: {val_track_dpe:.2f} km "
              f"(Kin Base: {val_metrics['kin_mean_track_dpe_km']:.2f} km, Diff: {val_metrics['improvement_over_kin_mean_km']:+.2f} km)")
        print(f"  --> Horizons: +12h={val_metrics['12h']['mean']:.1f} km, +24h={val_metrics['24h']['mean']:.1f} km, +48h={val_metrics['48h']['mean']:.1f} km")
        
        if val_track_dpe < best_val_track_dpe:
            best_val_track_dpe = val_track_dpe
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'val_mean_track_dpe': val_track_dpe
            }, ckpt_path)
            print(f"  *** Saved new best checkpoint at Epoch {epoch} ***")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    train_duration = time.time() - t_start
    print(f"\n{variant_name} Training Complete in {train_duration:.2f}s. Selected Epoch: {best_epoch} (Val Track DPE: {best_val_track_dpe:.2f} km)")
    
    # -------------------------------------------------------------------------
    # STRICT TEST EVALUATION PROTOCOL (Evaluated ONCE after freeze)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"STRICT TEST EVALUATION FOR {variant_name} (Evaluated ONCE on frozen checkpoint)")
    print("=" * 80)
    
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_metrics = evaluate_variant(model, test_loader, device)
    
    print(f"TEST Mean Track DPE: {test_metrics['mean_track_dpe_km']:.2f} km")
    print(f"  Kinematic Anchor Base: {test_metrics['kin_mean_track_dpe_km']:.2f} km")
    print(f"  Improvement vs Base: {test_metrics['improvement_over_kin_mean_km']:+.2f} km")
    print(f"  +12h: Mean={test_metrics['12h']['mean']:.2f} km, Med={test_metrics['12h']['median']:.2f} km, P90={test_metrics['12h']['p90']:.2f} km")
    print(f"  +24h: Mean={test_metrics['24h']['mean']:.2f} km, Med={test_metrics['24h']['median']:.2f} km, P90={test_metrics['24h']['p90']:.2f} km")
    print(f"  +48h: Mean={test_metrics['48h']['mean']:.2f} km, Med={test_metrics['48h']['median']:.2f} km, P90={test_metrics['48h']['p90']:.2f} km")
    print("  Predicted Residuals:")
    for h in ['12h', '24h', '48h']:
        rinfo = test_metrics['residuals'][h]
        print(f"    {h}: Mean_Mag={rinfo['mean_mag_km']:.2f} km, Bias_N={rinfo['bias_north_km']:+.2f} km, Bias_E={rinfo['bias_east_km']:+.2f} km")
    print("=" * 80)
    
    return {
        'selected_epoch': best_epoch,
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'training_duration': train_duration
    }


def main():
    print("=" * 80)
    print("VAYU-NET PHASE 5B: ENVIRONMENT-AWARE HYBRID RESIDUAL PIPELINE")
    print("=" * 80)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cache_data = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    
    # 1. Train Variant A (Observed-Center Hybrid)
    res_a = train_variant(
        "Variant A — Observed-Center Hybrid",
        cache_data,
        CKPT_PATH_A,
        device,
        max_epochs=15,
        patience=4
    )
    
    # 2. Train Variant B (Satellite-Derived Operational Hybrid)
    res_b = train_variant(
        "Variant B — Satellite-Derived Operational Hybrid",
        cache_data,
        CKPT_PATH_B,
        device,
        max_epochs=15,
        patience=4
    )
    
    # 3. Compile Results JSON
    results_payload = {
        'metadata': {
            'phase': 'Phase 5B',
            'title': 'Environment-Aware Hybrid Residual Track Prediction',
            'device': str(device),
            'random_seed': seed,
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'batch_size': 16,
            'max_epochs': 15,
            'patience': 4,
            'model_selection_criterion': 'Minimum VALIDATION Mean Track DPE (km)',
            'coordinate_system': 'Local tangent-plane geodesic displacement [res_north_km, res_east_km]',
            'residual_scale_km': RESIDUAL_SCALE_KM,
            'sat_feature_dim': 132,
            'env_channels': 8,
            'env_shape': [6, 8, 41, 66],
            'motion_in_dim': 5
        },
        'baselines': {
            'stationary_persistence': {
                '12h': 132.67, '24h': 263.37, '48h': 530.81, 'mean_track': 308.95
            },
            'constant_velocity_exact': {
                '12h': 71.28, '24h': 150.74, '48h': 345.15, 'mean_track': 189.06
            },
            'phase4a_satellite_only_gru': {
                '12h': 823.00, '24h': 832.50, '48h': 890.00, 'mean_track': 848.50
            },
            'phase4b_variant_a': {
                '12h': 72.57, '24h': 151.15, '48h': 345.41, 'mean_track': 189.71
            },
            'phase5a_multimodal': {
                '12h': 941.19, '24h': 963.74, '48h': 1031.22, 'mean_track': 978.72
            }
        },
        'variant_a_observed_center': {
            'selected_epoch': res_a['selected_epoch'],
            'training_duration_seconds': res_a['training_duration'],
            'val_metrics': {k: v for k, v in res_a['val_metrics'].items() if k != 'raw'},
            'test_metrics': {k: v for k, v in res_a['test_metrics'].items() if k != 'raw'},
            'comparison_with_exact_kinematic_base': {
                'kinematic_base_mean_track_dpe_km': res_a['test_metrics']['kin_mean_track_dpe_km'],
                'hybrid_mean_track_dpe_km': res_a['test_metrics']['mean_track_dpe_km'],
                'improvement_km': res_a['test_metrics']['improvement_over_kin_mean_km']
            },
            'comparison_with_phase4b_variant_a': {
                'phase4b_variant_a_mean_track_dpe_km': 189.71,
                'phase5b_variant_a_mean_track_dpe_km': res_a['test_metrics']['mean_track_dpe_km'],
                'delta_vs_phase4b_km': res_a['test_metrics']['mean_track_dpe_km'] - 189.71
            },
            'raw_test': res_a['test_metrics']['raw']
        },
        'variant_b_satellite_derived': {
            'selected_epoch': res_b['selected_epoch'],
            'training_duration_seconds': res_b['training_duration'],
            'val_metrics': {k: v for k, v in res_b['val_metrics'].items() if k != 'raw'},
            'test_metrics': {k: v for k, v in res_b['test_metrics'].items() if k != 'raw'},
            'comparison_with_satellite_kinematic_base': {
                'satellite_kinematic_base_mean_track_dpe_km': res_b['test_metrics']['kin_mean_track_dpe_km'],
                'hybrid_mean_track_dpe_km': res_b['test_metrics']['mean_track_dpe_km'],
                'improvement_km': res_b['test_metrics']['improvement_over_kin_mean_km'],
                'percentage_error_reduction': (res_b['test_metrics']['improvement_over_kin_mean_km'] / res_b['test_metrics']['kin_mean_track_dpe_km']) * 100
            },
            'comparison_with_phase4b_variant_b': {
                'phase4b_variant_b_mean_track_dpe_km': 2041.43,
                'phase5b_variant_b_mean_track_dpe_km': res_b['test_metrics']['mean_track_dpe_km'],
                'delta_vs_phase4b_km': res_b['test_metrics']['mean_track_dpe_km'] - 2041.43
            },
            'raw_test': res_b['test_metrics']['raw']
        }
    }
    
    with open(RESULTS_PATH, 'w') as fp:
        json.dump(results_payload, fp, indent=2)
    print(f"\nSuccessfully compiled results to {RESULTS_PATH}")
    print("=" * 80)

if __name__ == '__main__':
    main()
