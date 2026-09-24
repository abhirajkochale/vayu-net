import pandas as pd
from datetime import timedelta
from pathlib import Path

csv_path = Path("data/processed/imd_best_track_v2.csv")
df = pd.read_csv(csv_path)

fani = df[df["storm_id"] == "NIO_2019_FANI"].copy()
fani["dt"] = pd.to_datetime(fani["timestamp_utc"])
fani = fani.sort_values("dt").reset_index(drop=True)

imd_dt_set = set(fani["dt"])

candidates = []
for idx, r in fani.iterrows():
    t0 = r["dt"]
    if t0.minute != 0 or t0.second != 0 or (t0.hour % 3 != 0):
        continue
    
    t_plus_12 = t0 + timedelta(hours=12)
    t_plus_24 = t0 + timedelta(hours=24)
    t_plus_48 = t0 + timedelta(hours=48)
    
    has_12 = t_plus_12 in imd_dt_set
    has_24 = t_plus_24 in imd_dt_set
    has_48 = t_plus_48 in imd_dt_set
    
    if has_12 and has_24 and has_48:
        hist_seq = [t0 - timedelta(hours=h) for h in [15, 12, 9, 6, 3, 0]]
        candidates.append({
            "t0": t0,
            "t0_iso": t0.isoformat(),
            "lat": r["latitude"],
            "lon": r["longitude"],
            "wind": r["maximum_sustained_wind_kt"],
            "pres": r["central_pressure_hpa"],
            "cat": r["category"],
            "t_plus_12": t_plus_12.isoformat(),
            "t_plus_24": t_plus_24.isoformat(),
            "t_plus_48": t_plus_48.isoformat(),
            "history": [h.isoformat() for h in hist_seq]
        })

print(f"Total FANI observations: {len(fani)}")
print(f"Total candidates found: {len(candidates)}")
print("=" * 80)
for idx, c in enumerate(candidates):
    print(f"Candidate #{idx+1}: {c['t0_iso']} | Lat: {c['lat']}, Lon: {c['lon']}, Wind: {c['wind']} kt, Pres: {c['pres']} hPa, Cat: {c['cat']}")
    print(f"  Targets: +12h: {c['t_plus_12']}, +24h: {c['t_plus_24']}, +48h: {c['t_plus_48']}")

if candidates:
    sel = candidates[0]
    print("\n" + "=" * 80)
    print("EARLIEST VALID CANDIDATE t0 (SELECTED):")
    print(f"Selected t0: {sel['t0_iso']}")
    print(f"History (6 frames): {sel['history']}")
    print(f"Target +12h: {sel['t_plus_12']}")
    print(f"Target +24h: {sel['t_plus_24']}")
    print(f"Target +48h: {sel['t_plus_48']}")
