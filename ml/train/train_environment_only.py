"""
VAYU-NET PHASE 5A — EXPERIMENT B: ENVIRONMENT-ONLY TRACK TRAINING
=================================================================
Trains a standalone environmental track model using 6-frame ERA5 atmospheric
wind sequences [B, 6, 8, 41, 66] to predict +12h, +24h, and +48h cyclone tracks.

Loss: Multi-horizon Smooth L1 loss on (lat, lon) coordinates
Checkpoint Selection: Minimum VALIDATION Mean Track DPE
Evaluation: Evaluates held-out TEST set strictly ONCE after checkpoint freeze.
Saves checkpoint to: data/interim/ml/checkpoints/best_environment_only.pt
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.environment_encoder import EnvironmentOnlyTrackModel

CACHE_PATH = "data/interim/ml/cache/era5_environment_features.pt"
CHECKPOINT_DIR = "data/interim/ml/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_environment_only.pt")

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

def prepare_tensors_from_cache(cache_data, split):
    items = cache_data['samples'][split]
    
    env_seqs = torch.stack([item['env_seq'].float() for item in items])
    t12 = torch.stack([item['target_12'] for item in items])
    t24 = torch.stack([item['target_24'] for item in items])
    t48 = torch.stack([item['target_48'] for item in items])
    
    m12 = torch.stack([item['mask_12'] for item in items])
    m24 = torch.stack([item['mask_24'] for item in items])
    m48 = torch.stack([item['mask_48'] for item in items])
    
    dataset = TensorDataset(env_seqs, t12, t24, t48, m12, m24, m48)
    return dataset, items

def evaluate_model(model, dataloader, device):
    model.eval()
    all_dpes_12 = []
    all_dpes_24 = []
    all_dpes_48 = []
    
    with torch.no_grad():
        for batch in dataloader:
            env_seq, t12, t24, t48, m12, m24, m48 = [x.to(device) for x in batch]
            out = model(env_seq)
            
            p12 = out['pred_12'].cpu().numpy()
            p24 = out['pred_24'].cpu().numpy()
            p48 = out['pred_48'].cpu().numpy()
            
            t12_np = t12.cpu().numpy()
            t24_np = t24.cpu().numpy()
            t48_np = t48.cpu().numpy()
            
            m12_np = m12.cpu().numpy()
            m24_np = m24.cpu().numpy()
            m48_np = m48.cpu().numpy()
            
            for i in range(len(p12)):
                if m12_np[i] > 0.5:
                    d = haversine_dpe_km(p12[i, 0], p12[i, 1], t12_np[i, 0], t12_np[i, 1])
                    all_dpes_12.append(d)
                if m24_np[i] > 0.5:
                    d = haversine_dpe_km(p24[i, 0], p24[i, 1], t24_np[i, 0], t24_np[i, 1])
                    all_dpes_24.append(d)
                if m48_np[i] > 0.5:
                    d = haversine_dpe_km(p48[i, 0], p48[i, 1], t48_np[i, 0], t48_np[i, 1])
                    all_dpes_48.append(d)
                    
    m_12 = np.mean(all_dpes_12) if all_dpes_12 else float('nan')
    med_12 = np.median(all_dpes_12) if all_dpes_12 else float('nan')
    p90_12 = np.percentile(all_dpes_12, 90) if all_dpes_12 else float('nan')

    m_24 = np.mean(all_dpes_24) if all_dpes_24 else float('nan')
    med_24 = np.median(all_dpes_24) if all_dpes_24 else float('nan')
    p90_24 = np.percentile(all_dpes_24, 90) if all_dpes_24 else float('nan')

    m_48 = np.mean(all_dpes_48) if all_dpes_48 else float('nan')
    med_48 = np.median(all_dpes_48) if all_dpes_48 else float('nan')
    p90_48 = np.percentile(all_dpes_48, 90) if all_dpes_48 else float('nan')

    mean_track = np.mean([m_12, m_24, m_48])
    return {
        'mean_track_dpe_km': float(mean_track),
        '12h': {'mean': float(m_12), 'median': float(med_12), 'p90': float(p90_12), 'count': len(all_dpes_12)},
        '24h': {'mean': float(m_24), 'median': float(med_24), 'p90': float(p90_24), 'count': len(all_dpes_24)},
        '48h': {'mean': float(m_48), 'median': float(med_48), 'p90': float(p90_48), 'count': len(all_dpes_48)},
        'dpes_12': all_dpes_12,
        'dpes_24': all_dpes_24,
        'dpes_48': all_dpes_48
    }

def train_environment_only():
    print("=" * 80)
    print("VAYU-NET PHASE 5A: EXPERIMENT B — ENVIRONMENT-ONLY TRACK MODEL")
    print("=" * 80)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(f"Missing cache at {CACHE_PATH}. Build it first!")
        
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    
    train_ds, train_items = prepare_tensors_from_cache(cache, 'TRAIN')
    val_ds, val_items = prepare_tensors_from_cache(cache, 'VALIDATION')
    test_ds, test_items = prepare_tensors_from_cache(cache, 'TEST')
    
    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model = EnvironmentOnlyTrackModel(in_channels=8, spatial_dim=64, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(reduction='none')
    
    best_val_track_dpe = float('inf')
    best_epoch = -1
    best_val_metrics = None
    
    max_epochs = 15
    patience = 4
    patience_counter = 0
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    t_start = time.time()
    
    print("\nStarting training loop (Selection governed by VALIDATION Mean Track DPE)...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            env_seq, t12, t24, t48, m12, m24, m48 = [x.to(device) for x in batch]
            optimizer.zero_grad()
            
            out = model(env_seq)
            p12, p24, p48 = out['pred_12'], out['pred_24'], out['pred_48']
            
            l12 = (criterion(p12, t12).sum(dim=-1) * m12).sum() / (m12.sum() + 1e-6)
            l24 = (criterion(p24, t24).sum(dim=-1) * m24).sum() / (m24.sum() + 1e-6)
            l48 = (criterion(p48, t48).sum(dim=-1) * m48).sum() / (m48.sum() + 1e-6)
            
            loss = l12 + l24 + l48
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        val_metrics = evaluate_model(model, val_loader, device)
        val_dpe = val_metrics['mean_track_dpe_km']
        
        print(f"Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val Track DPE: {val_dpe:.1f} km "
              f"(+12h: {val_metrics['12h']['mean']:.1f}, +24h: {val_metrics['24h']['mean']:.1f}, +48h: {val_metrics['48h']['mean']:.1f})")
              
        if val_dpe < best_val_track_dpe:
            best_val_track_dpe = val_dpe
            best_epoch = epoch
            best_val_metrics = val_metrics
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'val_mean_track_dpe': val_dpe
            }, CHECKPOINT_PATH)
            print(f"  --> Saved new best checkpoint at Epoch {epoch} (Val Track DPE: {val_dpe:.2f} km)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
    train_duration = time.time() - t_start
    print(f"\nTraining complete in {train_duration:.2f}s. Best Epoch: {best_epoch} with Val Track DPE: {best_val_track_dpe:.2f} km")
    
    # -------------------------------------------------------------------------
    # STRICT TEST EVALUATION PROTOCOL (Evaluated ONCE after freeze)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRICT TEST EVALUATION (Evaluated exactly ONCE on frozen checkpoint)")
    print("=" * 80)
    
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"TEST Mean Track DPE: {test_metrics['mean_track_dpe_km']:.2f} km")
    print(f"  +12h: Mean={test_metrics['12h']['mean']:.2f} km, Med={test_metrics['12h']['median']:.2f} km, P90={test_metrics['12h']['p90']:.2f} km")
    print(f"  +24h: Mean={test_metrics['24h']['mean']:.2f} km, Med={test_metrics['24h']['median']:.2f} km, P90={test_metrics['24h']['p90']:.2f} km")
    print(f"  +48h: Mean={test_metrics['48h']['mean']:.2f} km, Med={test_metrics['48h']['median']:.2f} km, P90={test_metrics['48h']['p90']:.2f} km")
    print("=" * 80)
    
    return {
        'best_epoch': best_epoch,
        'val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'training_duration': train_duration
    }

if __name__ == '__main__':
    train_environment_only()
