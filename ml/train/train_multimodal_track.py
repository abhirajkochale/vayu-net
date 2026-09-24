"""
VAYU-NET PHASE 5A — EXPERIMENT C: MULTIMODAL TRACK PREDICTION TRAINING
======================================================================
Trains the joint satellite + environmental multimodal model:
  - Satellite branch: 6-frame precomputed Phase 3C/4A spatial features [B, 6, 132]
  - Environmental branch: 6-frame ERA5 multi-level wind tensors [B, 6, 8, 41, 66]
  - Late concatenation fusion: [h_sat, h_env] -> MLP track heads
  - Predicts future cyclone tracks: +12h, +24h, +48h

Loss: Multi-horizon Smooth L1 loss on (lat, lon) coordinates
Checkpoint Selection: Minimum VALIDATION Mean Track DPE
Evaluation: Evaluates held-out TEST set strictly ONCE after checkpoint freeze.
Saves checkpoint to: data/interim/ml/checkpoints/best_phase5a_multimodal.pt
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
from ml.models.environment_encoder import EnvironmentTemporalGRU

ERA5_CACHE_PATH = "data/interim/ml/cache/era5_environment_features.pt"
SAT_CACHE_PATH = "data/interim/ml/cache/temporal_track_features.pt"
CHECKPOINT_DIR = "data/interim/ml/checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_phase5a_multimodal.pt")

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

class MultimodalDataset(Dataset):
    def __init__(self, era5_cache, sat_cache, split='TRAIN'):
        self.era5_items = era5_cache['samples'][split]
        # Build map for sat items by sample_id
        sat_map = {item['sample_id']: item for item in sat_cache['samples'] if item['split'] == split}
        
        self.pairs = []
        for e_item in self.era5_items:
            sid = e_item['sample_id']
            if sid in sat_map:
                s_item = sat_map[sid]
                self.pairs.append((e_item, s_item))
            else:
                raise KeyError(f"Sample {sid} not found in satellite cache!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        e_item, s_item = self.pairs[idx]
        env_seq = e_item['env_seq'].float() # [6, 8, 41, 66]
        sat_seq = s_item['feature_sequence'].float() # [6, 132]
        
        t12 = e_item['target_12']
        t24 = e_item['target_24']
        t48 = e_item['target_48']
        
        m12 = e_item['mask_12']
        m24 = e_item['mask_24']
        m48 = e_item['mask_48']
        
        return env_seq, sat_seq, t12, t24, t48, m12, m24, m48, e_item['sample_id'], e_item['storm_id'], e_item['t0']


class MultimodalFusionModel(nn.Module):
    """
    Lightweight, fast multimodal model operating on precomputed satellite features
    and ERA5 environmental tensors.
    """
    def __init__(self, sat_feat_dim=132, sat_hidden_dim=128, env_channels=8, env_spatial_dim=64, env_hidden_dim=64, fusion_dim=128, dropout=0.1):
        super().__init__()
        # Satellite GRU
        self.sat_gru = nn.GRU(
            input_size=sat_feat_dim,
            hidden_size=sat_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Environmental Branch
        self.env_gru = EnvironmentTemporalGRU(
            in_channels=env_channels,
            spatial_dim=env_spatial_dim,
            hidden_dim=env_hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        # Late Fusion
        multimodal_dim = sat_hidden_dim + env_hidden_dim # 128 + 64 = 192
        self.fusion = nn.Sequential(
            nn.Linear(multimodal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.head_12 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.head_24 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        self.head_48 = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, sat_seq, env_seq):
        # sat_seq: [B, 6, 132]
        # env_seq: [B, 6, 8, 41, 66]
        _, h_sat_n = self.sat_gru(sat_seq)
        h_sat = h_sat_n[-1] # [B, 128]
        
        h_env, _ = self.env_gru(env_seq) # [B, 64]
        
        fused = torch.cat([h_sat, h_env], dim=-1) # [B, 192]
        h_fused = self.fusion(fused) # [B, fusion_dim]
        
        pred_12 = self.head_12(h_fused)
        pred_24 = self.head_24(h_fused)
        pred_48 = self.head_48(h_fused)
        
        return {
            'pred_12': pred_12,
            'pred_24': pred_24,
            'pred_48': pred_48,
            'h_sat': h_sat,
            'h_env': h_env
        }


def evaluate_multimodal(model, dataloader, device):
    model.eval()
    all_dpes_12 = []
    all_dpes_24 = []
    all_dpes_48 = []
    raw_results = {
        'sample_ids': [], 'storm_ids': [], 't0s': [],
        'pred_12': [], 'pred_24': [], 'pred_48': [],
        'true_12': [], 'true_24': [], 'true_48': [],
        'dpe_12': [], 'dpe_24': [], 'dpe_48': [],
        'track_dpes': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            env_seq, sat_seq, t12, t24, t48, m12, m24, m48, sids, storms, t0s = batch
            env_seq = env_seq.to(device)
            sat_seq = sat_seq.to(device)
            
            out = model(sat_seq, env_seq)
            p12 = out['pred_12'].cpu().numpy()
            p24 = out['pred_24'].cpu().numpy()
            p48 = out['pred_48'].cpu().numpy()
            
            t12_np = t12.numpy()
            t24_np = t24.numpy()
            t48_np = t48.numpy()
            
            m12_np = m12.numpy()
            m24_np = m24.numpy()
            m48_np = m48.numpy()
            
            for i in range(len(p12)):
                d12 = haversine_dpe_km(p12[i, 0], p12[i, 1], t12_np[i, 0], t12_np[i, 1]) if m12_np[i] > 0.5 else np.nan
                d24 = haversine_dpe_km(p24[i, 0], p24[i, 1], t24_np[i, 0], t24_np[i, 1]) if m24_np[i] > 0.5 else np.nan
                d48 = haversine_dpe_km(p48[i, 0], p48[i, 1], t48_np[i, 0], t48_np[i, 1]) if m48_np[i] > 0.5 else np.nan
                
                valid_ds = [d for d in [d12, d24, d48] if not np.isnan(d)]
                tr_d = np.mean(valid_ds) if valid_ds else np.nan
                
                if not np.isnan(d12): all_dpes_12.append(d12)
                if not np.isnan(d24): all_dpes_24.append(d24)
                if not np.isnan(d48): all_dpes_48.append(d48)
                
                raw_results['sample_ids'].append(sids[i])
                raw_results['storm_ids'].append(storms[i])
                raw_results['t0s'].append(t0s[i])
                raw_results['pred_12'].append(p12[i].tolist())
                raw_results['pred_24'].append(p24[i].tolist())
                raw_results['pred_48'].append(p48[i].tolist())
                raw_results['true_12'].append(t12_np[i].tolist())
                raw_results['true_24'].append(t24_np[i].tolist())
                raw_results['true_48'].append(t48_np[i].tolist())
                raw_results['dpe_12'].append(d12)
                raw_results['dpe_24'].append(d24)
                raw_results['dpe_48'].append(d48)
                raw_results['track_dpes'].append(tr_d)

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
        'raw': raw_results
    }


def train_multimodal_track():
    print("=" * 80)
    print("VAYU-NET PHASE 5A: EXPERIMENT C — MULTIMODAL TRACK MODEL")
    print("=" * 80)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    era5_cache = torch.load(ERA5_CACHE_PATH, map_location='cpu', weights_only=False)
    sat_cache = torch.load(SAT_CACHE_PATH, map_location='cpu', weights_only=False)

    train_ds = MultimodalDataset(era5_cache, sat_cache, 'TRAIN')
    val_ds = MultimodalDataset(era5_cache, sat_cache, 'VALIDATION')
    test_ds = MultimodalDataset(era5_cache, sat_cache, 'TEST')

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = MultimodalFusionModel(
        sat_feat_dim=132,
        sat_hidden_dim=128,
        env_channels=8,
        env_spatial_dim=64,
        env_hidden_dim=64,
        fusion_dim=128,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable model parameters: {total_params:,}")

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
            env_seq, sat_seq, t12, t24, t48, m12, m24, m48, _, _, _ = batch
            env_seq = env_seq.to(device)
            sat_seq = sat_seq.to(device)
            t12, t24, t48 = t12.to(device), t24.to(device), t48.to(device)
            m12, m24, m48 = m12.to(device), m24.to(device), m48.to(device)

            optimizer.zero_grad()
            out = model(sat_seq, env_seq)
            p12, p24, p48 = out['pred_12'], out['pred_24'], out['pred_48']

            l12 = (criterion(p12, t12).sum(dim=-1) * m12).sum() / (m12.sum() + 1e-6)
            l24 = (criterion(p24, t24).sum(dim=-1) * m24).sum() / (m24.sum() + 1e-6)
            l48 = (criterion(p48, t48).sum(dim=-1) * m48).sum() / (m48.sum() + 1e-6)

            loss = l12 + l24 + l48
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_metrics = evaluate_multimodal(model, val_loader, device)
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
    test_metrics = evaluate_multimodal(model, test_loader, device)

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
    train_multimodal_track()
