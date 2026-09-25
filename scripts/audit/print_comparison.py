import json
import pandas as pd

with open('data/interim/ml/multisource_transfer_expanded_results.json') as f:
    d = json.load(f)

print('=== EXPANDED TRAIN RESULTS (N=207) ===')
for exp in ['EXP-1', 'EXP-2', 'EXP-3']:
    res = d['experiments'][exp]
    tm = res['test_metrics']
    id_m = tm['identification']
    w_m = tm['wind_regression']
    tr_m = tm['track_prediction']
    c_m = tm['classification']
    mode = res['mode']
    ch = res['in_channels_insat']
    val_loss = res['val_loss']
    test_loss = res['test_loss']
    epoch = res['best_epoch']
    print(f"\n{exp} ({mode}, ch={ch}):")
    print(f"  Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}, Best Epoch: {epoch}")
    print(f"  Center: Mean DPE={id_m['center_mean_dpe_km']:.2f}km, Med={id_m['center_median_dpe_km']:.2f}km, P90={id_m['center_p90_dpe_km']:.2f}km, Lat MAE={id_m['center_lat_mae_deg']:.3f}, Lon MAE={id_m['center_lon_mae_deg']:.3f}")
    print(f"  Wind:   MAE={w_m['wind_mae_kt']:.2f}kt, RMSE={w_m['wind_rmse_kt']:.2f}kt, Med={w_m['wind_median_ae_kt']:.2f}kt, P90={w_m['wind_p90_ae_kt']:.2f}kt, Bias={w_m['wind_bias_kt']:.2f}kt, Corr={w_m['wind_pearson_r']:.4f}")
    print(f"  Track:  +12h={tr_m['track_12h_mean_dpe_km']:.2f}km, +24h={tr_m['track_24h_mean_dpe_km']:.2f}km, +48h={tr_m['track_48h_mean_dpe_km']:.2f}km, Agg DPE={tr_m['track_aggregate_mean_dpe_km']:.2f}km")
    print(f"  Class:  Acc={c_m['accuracy']:.4f}, Macro-F1={c_m['macro_f1']:.4f}")

with open('data/interim/ml/multisource_transfer_results.json') as f:
    d_old = json.load(f)

print('\n=== PREVIOUS TRAIN RESULTS (N=175) ===')
for exp in ['EXP-1', 'Channel_B_TIR1_TIR2', 'EXP-2']:
    res = d_old.get(exp, {})
    tm = res.get('test_metrics', {})
    id_m = tm.get('identification', {})
    w_m = tm.get('wind_regression', {})
    tr_m = tm.get('track_prediction', {})
    c_m = tm.get('classification', {})
    c_dpe = id_m.get('center_mean_dpe_km', 0)
    c_med = id_m.get('center_median_dpe_km', 0)
    c_p90 = id_m.get('center_p90_dpe_km', 0)
    w_mae = w_m.get('wind_mae_kt', 0)
    w_rmse = w_m.get('wind_rmse_kt', 0)
    w_r = w_m.get('wind_pearson_r', 0)
    tr_12 = tr_m.get('track_12h_mean_dpe_km', 0)
    tr_24 = tr_m.get('track_24h_mean_dpe_km', 0)
    tr_48 = tr_m.get('track_48h_mean_dpe_km', 0)
    tr_dpe = tr_m.get('track_aggregate_mean_dpe_km', 0)
    print(f"\n{exp}:")
    print(f"  Center: Mean DPE={c_dpe:.2f}km, Med={c_med:.2f}km, P90={c_p90:.2f}km")
    print(f"  Wind:   MAE={w_mae:.2f}kt, RMSE={w_rmse:.2f}kt, Corr={w_r:.4f}")
    print(f"  Track:  +12h={tr_12:.2f}km, +24h={tr_24:.2f}km, +48h={tr_48:.2f}km, Agg DPE={tr_dpe:.2f}km")
