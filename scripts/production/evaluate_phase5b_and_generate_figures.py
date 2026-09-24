"""
VAYU-NET PHASE 5B — EVALUATION, RESULTS SERIALIZATION & FIGURE SUITE
====================================================================
Evaluates frozen checkpoints on held-out TEST split strictly ONCE:
  - best_phase5b_variant_a.pt
  - best_phase5b_variant_b.pt

Saves:
  data/interim/ml/phase5b_hybrid_results.json

Generates all 10 required diagnostic figures in docs/figures/phase5b_hybrid/:
  1. validation_track_dpe.png
  2. horizon_comparison.png
  3. kinematic_vs_hybrid.png
  4. residual_predictions.png
  5. residual_bias.png
  6. test_dpe_distributions.png
  7. variant_a_vs_variant_b.png
  8. arabian_sea_vs_bob.png
  9. representative_tracks.png
  10. worst_case_tracks.png
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from ml.models.phase5b_hybrid_residual import Phase5BHybridResidualModel
from ml.train.train_phase5b_hybrid_residual import (
    Phase5BDataset,
    evaluate_variant,
    CACHE_PATH,
    CKPT_PATH_A,
    CKPT_PATH_B,
    RESULTS_PATH
)

FIG_DIR = "docs/figures/phase5b_hybrid"
os.makedirs(FIG_DIR, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def run_evaluation_and_figures():
    print("=" * 80)
    print("PHASE 5B: EVALUATING FROZEN CHECKPOINTS ON HELD-OUT TEST SPLIT")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_data = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    
    # Load frozen checkpoints
    ckpt_a = torch.load(CKPT_PATH_A, map_location=device, weights_only=False)
    ckpt_b = torch.load(CKPT_PATH_B, map_location=device, weights_only=False)
    
    # -------------------------------------------------------------------------
    # 1. EVALUATE VARIANT A ON TEST
    # -------------------------------------------------------------------------
    print("Evaluating Variant A on TEST partition (strictly once)...")
    test_ds_a = Phase5BDataset(cache_data, 'TEST', 'A')
    test_loader_a = DataLoader(test_ds_a, batch_size=16, shuffle=False)
    
    model_a = Phase5BHybridResidualModel().to(device)
    model_a.load_state_dict(ckpt_a['model_state_dict'])
    test_metrics_a = evaluate_variant(model_a, test_loader_a, device)
    
    print(f"Variant A TEST Mean Track DPE: {test_metrics_a['mean_track_dpe_km']:.2f} km "
          f"(Kin Base: {test_metrics_a['kin_mean_track_dpe_km']:.2f} km, Diff: {test_metrics_a['improvement_over_kin_mean_km']:+.2f} km)")
    print(f"  +12h: {test_metrics_a['12h']['mean']:.2f} km | +24h: {test_metrics_a['24h']['mean']:.2f} km | +48h: {test_metrics_a['48h']['mean']:.2f} km")

    # -------------------------------------------------------------------------
    # 2. EVALUATE VARIANT B ON TEST
    # -------------------------------------------------------------------------
    print("\nEvaluating Variant B on TEST partition (strictly once)...")
    test_ds_b = Phase5BDataset(cache_data, 'TEST', 'B')
    test_loader_b = DataLoader(test_ds_b, batch_size=16, shuffle=False)
    
    model_b = Phase5BHybridResidualModel().to(device)
    model_b.load_state_dict(ckpt_b['model_state_dict'])
    test_metrics_b = evaluate_variant(model_b, test_loader_b, device)
    
    print(f"Variant B TEST Mean Track DPE: {test_metrics_b['mean_track_dpe_km']:.2f} km "
          f"(Kin Base: {test_metrics_b['kin_mean_track_dpe_km']:.2f} km, Diff: {test_metrics_b['improvement_over_kin_mean_km']:+.2f} km)")
    print(f"  +12h: {test_metrics_b['12h']['mean']:.2f} km | +24h: {test_metrics_b['24h']['mean']:.2f} km | +48h: {test_metrics_b['48h']['mean']:.2f} km")

    # -------------------------------------------------------------------------
    # 3. SAVE RESULTS JSON
    # -------------------------------------------------------------------------
    results_payload = {
        'metadata': {
            'phase': 'Phase 5B',
            'title': 'Environment-Aware Hybrid Residual Track Prediction',
            'device': str(device),
            'random_seed': 42,
            'optimizer': 'AdamW',
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'batch_size': 16,
            'coordinate_system': 'Local tangent-plane geodesic displacement [res_north_km, res_east_km]',
            'residual_scale_km': 100.0,
            'sat_feature_dim': 132,
            'env_channels': 8,
            'env_shape': [6, 8, 41, 66],
            'motion_in_dim': 5,
            'variant_a_selected_epoch': ckpt_a['epoch'],
            'variant_b_selected_epoch': ckpt_b['epoch'],
            'variant_a_val_dpe': ckpt_a['val_mean_track_dpe'],
            'variant_b_val_dpe': ckpt_b['val_mean_track_dpe']
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
            'phase4b_variant_b': {
                '12h': 1619.80, '24h': 2000.45, '48h': 2504.03, 'mean_track': 2041.43
            },
            'phase5a_multimodal': {
                '12h': 941.19, '24h': 963.74, '48h': 1031.22, 'mean_track': 978.72
            }
        },
        'variant_a_observed_center': {
            'selected_epoch': ckpt_a['epoch'],
            'val_metrics': {k: v for k, v in ckpt_a['val_metrics'].items() if k != 'raw'},
            'test_metrics': {k: v for k, v in test_metrics_a.items() if k != 'raw'},
            'comparison_with_exact_kinematic_base': {
                'kinematic_base_mean_track_dpe_km': test_metrics_a['kin_mean_track_dpe_km'],
                'hybrid_mean_track_dpe_km': test_metrics_a['mean_track_dpe_km'],
                'improvement_km': test_metrics_a['improvement_over_kin_mean_km'],
                'percentage_improvement': (test_metrics_a['improvement_over_kin_mean_km'] / test_metrics_a['kin_mean_track_dpe_km']) * 100
            },
            'comparison_with_phase4b_variant_a': {
                'phase4b_variant_a_mean_track_dpe_km': 189.71,
                'phase5b_variant_a_mean_track_dpe_km': test_metrics_a['mean_track_dpe_km'],
                'improvement_vs_phase4b_km': 189.71 - test_metrics_a['mean_track_dpe_km']
            },
            'raw_test': test_metrics_a['raw']
        },
        'variant_b_satellite_derived': {
            'selected_epoch': ckpt_b['epoch'],
            'val_metrics': {k: v for k, v in ckpt_b['val_metrics'].items() if k != 'raw'},
            'test_metrics': {k: v for k, v in test_metrics_b.items() if k != 'raw'},
            'comparison_with_satellite_kinematic_base': {
                'satellite_kinematic_base_mean_track_dpe_km': test_metrics_b['kin_mean_track_dpe_km'],
                'hybrid_mean_track_dpe_km': test_metrics_b['mean_track_dpe_km'],
                'improvement_km': test_metrics_b['improvement_over_kin_mean_km'],
                'percentage_error_reduction': (test_metrics_b['improvement_over_kin_mean_km'] / test_metrics_b['kin_mean_track_dpe_km']) * 100
            },
            'comparison_with_phase4b_variant_b': {
                'phase4b_variant_b_mean_track_dpe_km': 2041.43,
                'phase5b_variant_b_mean_track_dpe_km': test_metrics_b['mean_track_dpe_km'],
                'improvement_vs_phase4b_km': 2041.43 - test_metrics_b['mean_track_dpe_km']
            },
            'raw_test': test_metrics_b['raw']
        }
    }

    with open(RESULTS_PATH, 'w') as fp:
        json.dump(results_payload, fp, indent=2, cls=NumpyEncoder)
    print(f"\nSaved results payload to {RESULTS_PATH}")

    # -------------------------------------------------------------------------
    # 4. GENERATE ALL 10 DIAGNOSTIC FIGURES
    # -------------------------------------------------------------------------
    print("\nGenerating 10 diagnostic figures in docs/figures/phase5b_hybrid/...")
    raw_a = test_metrics_a['raw']
    raw_b = test_metrics_b['raw']

    # 1. validation_track_dpe.png
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = ["Variant A (Observed-Center)", "Variant B (Satellite-Derived)"]
    val_scores = [ckpt_a['val_mean_track_dpe'], ckpt_b['val_mean_track_dpe']]
    bars = ax.bar(variants, val_scores, color=['#2ca02c', '#d62728'], width=0.45, edgecolor='black')
    for b in bars:
        y = b.get_height()
        ax.text(b.get_x() + b.get_width()/2.0, y + 25, f"{y:.1f} km", ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylabel("Validation Mean Track DPE (km)", fontsize=11)
    ax.set_title("Phase 5B Validation Model Selection Performance", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    ax.set_ylim(0, max(val_scores) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "validation_track_dpe.png"), dpi=200); plt.close()
    print("  Saved: validation_track_dpe.png")

    # 2. horizon_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = ["+12h", "+24h", "+48h"]
    x = np.arange(len(horizons))
    w = 0.2
    ax.bar(x - 1.5*w, [71.28, 150.74, 345.15], w, label='Exact Constant Velocity', color='#ff7f0e', edgecolor='black')
    ax.bar(x - 0.5*w, [72.57, 151.15, 345.41], w, label='Phase 4B Variant A (Sat-only Res)', color='#1f77b4', edgecolor='black')
    ax.bar(x + 0.5*w, [test_metrics_a['12h']['mean'], test_metrics_a['24h']['mean'], test_metrics_a['48h']['mean']], w, label='Phase 5B Variant A (Sat+ERA5 Res)', color='#2ca02c', edgecolor='black')
    ax.bar(x + 1.5*w, [test_metrics_b['12h']['mean'], test_metrics_b['24h']['mean'], test_metrics_b['48h']['mean']], w, label='Phase 5B Variant B (Operational Res)', color='#d62728', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11)
    ax.set_ylabel("TEST Mean DPE (km)", fontsize=11)
    ax.set_title("Forecast Error Across Lead Times (+12h, +24h, +48h)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "horizon_comparison.png"), dpi=200); plt.close()
    print("  Saved: horizon_comparison.png")

    # 3. kinematic_vs_hybrid.png
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pairs = ["Variant A vs Exact Kin", "Variant B vs Sat-Derived Kin"]
    kin_vals = [test_metrics_a['kin_mean_track_dpe_km'], test_metrics_b['kin_mean_track_dpe_km']]
    hyb_vals = [test_metrics_a['mean_track_dpe_km'], test_metrics_b['mean_track_dpe_km']]
    x = np.arange(len(x_pairs))
    w = 0.3
    ax.bar(x - w/2, kin_vals, w, label='Kinematic Anchor Base', color='#ffbb78', edgecolor='black')
    ax.bar(x + w/2, hyb_vals, w, label='Phase 5B Hybrid Forecast', color='#2ca02c', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(x_pairs, fontsize=11)
    ax.set_ylabel("TEST Mean Track DPE (km)", fontsize=11)
    ax.set_title("Direct Fairness Evaluation: Kinematic Anchor vs Phase 5B Hybrid", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "kinematic_vs_hybrid.png"), dpi=200); plt.close()
    print("  Saved: kinematic_vs_hybrid.png")

    # 4. residual_predictions.png
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, h in enumerate(['12h', '24h', '48h']):
        ax = axes[i]
        rinfo_a = test_metrics_a['residuals'][h]
        rinfo_b = test_metrics_b['residuals'][h]
        labels = ['Variant A', 'Variant B']
        mags = [rinfo_a['mean_mag_km'], rinfo_b['mean_mag_km']]
        bars = ax.bar(labels, mags, color=['#2ca02c', '#d62728'], edgecolor='black', width=0.45)
        for b in bars:
            y = b.get_height()
            ax.text(b.get_x() + b.get_width()/2.0, y + 10, f"{y:.1f} km", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_title(f"Mean Residual Magnitude: +{h}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Residual Displacement (km)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residual_predictions.png"), dpi=200); plt.close()
    print("  Saved: residual_predictions.png")

    # 5. residual_bias.png
    fig, ax = plt.subplots(figsize=(9, 5))
    hs = ['12h', '24h', '48h']
    x = np.arange(len(hs))
    w = 0.2
    ax.bar(x - 1.5*w, [test_metrics_a['residuals'][h]['bias_north_km'] for h in hs], w, label='Var A North Bias', color='#1f77b4', edgecolor='black')
    ax.bar(x - 0.5*w, [test_metrics_a['residuals'][h]['bias_east_km'] for h in hs], w, label='Var A East Bias', color='#aec7e8', edgecolor='black')
    ax.bar(x + 0.5*w, [test_metrics_b['residuals'][h]['bias_north_km'] for h in hs], w, label='Var B North Bias', color='#d62728', edgecolor='black')
    ax.bar(x + 1.5*w, [test_metrics_b['residuals'][h]['bias_east_km'] for h in hs], w, label='Var B East Bias', color='#ff9896', edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"+{h}" for h in hs], fontsize=11)
    ax.set_ylabel("Displacement Bias (km)", fontsize=11)
    ax.set_title("Directional Residual Bias Across Lead Times (North & East)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "residual_bias.png"), dpi=200); plt.close()
    print("  Saved: residual_bias.png")

    # 6. test_dpe_distributions.png
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 800, 35)
    tr_a = np.array(raw_a['track_dpes'])[~np.isnan(raw_a['track_dpes'])]
    tr_kin_a = np.array(raw_a['kin_track_dpes'])[~np.isnan(raw_a['kin_track_dpes'])]
    ax.hist(tr_kin_a, bins=bins, alpha=0.5, label=f'Exact Constant Velocity (Mean: 189.1 km)', color='#ff7f0e', density=True)
    ax.hist(tr_a, bins=bins, alpha=0.5, label=f'Phase 5B Variant A (Mean: {test_metrics_a["mean_track_dpe_km"]:.1f} km)', color='#2ca02c', density=True)
    ax.set_xlabel("Track Great-Circle DPE (km)", fontsize=11)
    ax.set_ylabel("Empirical Density", fontsize=11)
    ax.set_title("TEST Track Error Distribution: Kinematic Base vs Phase 5B Variant A", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "test_dpe_distributions.png"), dpi=200); plt.close()
    print("  Saved: test_dpe_distributions.png")

    # 7. variant_a_vs_variant_b.png
    fig, ax = plt.subplots(figsize=(9, 5))
    models = ["Phase 5B Variant A (Observed-Center)", "Phase 5B Variant B (Satellite-Derived)"]
    means = [test_metrics_a['mean_track_dpe_km'], test_metrics_b['mean_track_dpe_km']]
    medians = [test_metrics_a['24h']['median'], test_metrics_b['24h']['median']]
    x = np.arange(len(models))
    w = 0.3
    ax.bar(x - w/2, means, w, label='TEST Mean Track DPE', color=['#2ca02c', '#d62728'], edgecolor='black')
    ax.bar(x + w/2, medians, w, label='TEST +24h Median DPE', color=['#98df8a', '#ff9896'], edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("DPE (km)", fontsize=11)
    ax.set_title("Controlled Comparison: Variant A vs Variant B", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "variant_a_vs_variant_b.png"), dpi=200); plt.close()
    print("  Saved: variant_a_vs_variant_b.png")

    # 8. arabian_sea_vs_bob.png
    as_mask = np.array([r[1] <= 77.5 for r in raw_a['kin_12']])
    bob_mask = ~as_mask
    
    as_dpe_a = np.mean(np.array(raw_a['track_dpes'])[as_mask])
    bob_dpe_a = np.mean(np.array(raw_a['track_dpes'])[bob_mask])
    as_kin_a = np.mean(np.array(raw_a['kin_track_dpes'])[as_mask])
    bob_kin_a = np.mean(np.array(raw_a['kin_track_dpes'])[bob_mask])

    fig, ax = plt.subplots(figsize=(9, 5))
    basins = [f"Arabian Sea (N={np.sum(as_mask)})", f"Bay of Bengal (N={np.sum(bob_mask)})"]
    x = np.arange(len(basins))
    w = 0.3
    ax.bar(x - w/2, [as_kin_a, bob_kin_a], w, label='Exact Kinematic Base', color='#ffbb78', edgecolor='black')
    ax.bar(x + w/2, [as_dpe_a, bob_dpe_a], w, label='Phase 5B Variant A Hybrid', color='#2ca02c', edgecolor='black')
    for i, (k_val, h_val) in enumerate(zip([as_kin_a, bob_kin_a], [as_dpe_a, bob_dpe_a])):
        diff = k_val - h_val
        ax.text(i + w/2, h_val + 5, f"{h_val:.1f} km\n({diff:+.1f} km)", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(basins, fontsize=11)
    ax.set_ylabel("Mean Track DPE (km)", fontsize=11)
    ax.set_title("Geographic Error Stratification: Arabian Sea vs Bay of Bengal", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "arabian_sea_vs_bob.png"), dpi=200); plt.close()
    print("  Saved: arabian_sea_vs_bob.png")

    # 9. representative_tracks.png
    dpes = np.array(raw_a['track_dpes'])
    kin_dpes = np.array(raw_a['kin_track_dpes'])
    diffs = kin_dpes - dpes
    best_idx = np.argsort(diffs)[::-1][:3]
    worst_idx = np.argsort(diffs)[:3]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for plot_i, idx in enumerate(list(best_idx) + list(worst_idx)):
        ax = axes[plot_i // 3, plot_i % 3]
        sid = raw_a['sample_ids'][idx]
        d = dpes[idx]; kd = kin_dpes[idx]; gain = diffs[idx]
        t_lats = [raw_a['true_12'][idx][0], raw_a['true_24'][idx][0], raw_a['true_48'][idx][0]]
        t_lons = [raw_a['true_12'][idx][1], raw_a['true_24'][idx][1], raw_a['true_48'][idx][1]]
        k_lats = [raw_a['kin_12'][idx][0], raw_a['kin_24'][idx][0], raw_a['kin_48'][idx][0]]
        k_lons = [raw_a['kin_12'][idx][1], raw_a['kin_24'][idx][1], raw_a['kin_48'][idx][1]]
        p_lats = [raw_a['pred_12'][idx][0], raw_a['pred_24'][idx][0], raw_a['pred_48'][idx][0]]
        p_lons = [raw_a['pred_12'][idx][1], raw_a['pred_24'][idx][1], raw_a['pred_48'][idx][1]]
        
        ax.plot(t_lons, t_lats, 'b-o', lw=2, markersize=6, label='True IMD')
        ax.plot(k_lons, k_lats, 'k--', lw=1.5, markersize=5, label=f'Kin Base ({kd:.1f} km)')
        ax.plot(p_lons, p_lats, 'g-^', lw=2, markersize=6, label=f'5B Hybrid ({d:.1f} km)')
        status = f"Improvement: +{gain:.1f} km" if gain > 0 else f"Degradation: {gain:.1f} km"
        ax.set_title(f"Case #{plot_i+1}: {sid}\n{status}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "representative_tracks.png"), dpi=200); plt.close()
    print("  Saved: representative_tracks.png")

    # 10. worst_case_tracks.png
    worst_overall = np.argsort(raw_a['track_dpes'])[::-1][:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for plot_i, idx in enumerate(worst_overall):
        ax = axes[plot_i // 3, plot_i % 3]
        sid = raw_a['sample_ids'][idx]; d = raw_a['track_dpes'][idx]
        t_lats = [raw_a['true_12'][idx][0], raw_a['true_24'][idx][0], raw_a['true_48'][idx][0]]
        t_lons = [raw_a['true_12'][idx][1], raw_a['true_24'][idx][1], raw_a['true_48'][idx][1]]
        p_lats = [raw_a['pred_12'][idx][0], raw_a['pred_24'][idx][0], raw_a['pred_48'][idx][0]]
        p_lons = [raw_a['pred_12'][idx][1], raw_a['pred_24'][idx][1], raw_a['pred_48'][idx][1]]
        ax.plot(t_lons, t_lats, 'b-o', lw=2, label='True Track')
        ax.plot(p_lons, p_lats, 'r--^', lw=2, label=f'5B-A Hybrid ({d:.1f} km)')
        ax.set_title(f"Worst Case #{plot_i+1}: {sid}\nTrack DPE: {d:.1f} km", fontsize=10, fontweight='bold')
        ax.set_xlabel("Lon (°E)"); ax.set_ylabel("Lat (°N)")
        ax.grid(True, linestyle=":", alpha=0.5); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "worst_case_tracks.png"), dpi=200); plt.close()
    print("  Saved: worst_case_tracks.png")

    print("\nALL 10 FIGURES GENERATED AND SAVED TO docs/figures/phase5b_hybrid/!")
    print("=" * 80)

if __name__ == '__main__':
    run_evaluation_and_figures()
