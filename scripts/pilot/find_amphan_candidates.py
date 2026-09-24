"""
Identify all valid candidate t0 timestamps for Cyclone AMPHAN (2020)
and verify exact required timestamps.
"""
import pandas as pd
from datetime import timedelta

IMD_V2_CSV = "data/processed/imd_best_track_v2.csv"
STORM_ID = "NIO_2020_AMPHAN"

def main():
    df = pd.read_csv(IMD_V2_CSV)
    amphan = df[df['storm_id'] == STORM_ID].copy()
    amphan['dt'] = pd.to_datetime(amphan['timestamp_utc'])
    amphan = amphan.sort_values('dt').reset_index(drop=True)
    
    available_timestamps = set(amphan['dt'])
    
    print(f"Cyclone AMPHAN has {len(amphan)} observations from {amphan['dt'].min()} to {amphan['dt'].max()}")
    
    candidates = []
    
    for idx, row in amphan.iterrows():
        t0 = row['dt']
        
        # 1. Native 3-hourly cadence check
        if t0.minute != 0 or t0.second != 0 or (t0.hour % 3 != 0):
            continue
            
        # 2. Check future IMD observations exist at +12h, +24h, +48h
        t_plus_12 = t0 + timedelta(hours=12)
        t_plus_24 = t0 + timedelta(hours=24)
        t_plus_48 = t0 + timedelta(hours=48)
        
        has_future_targets = (
            (t_plus_12 in available_timestamps) and
            (t_plus_24 in available_timestamps) and
            (t_plus_48 in available_timestamps)
        )
        
        if not has_future_targets:
            continue
            
        # 3. 6 historical GridSat timestamps
        # t-15h, t-12h, t-9h, t-6h, t-3h, t0
        hist_steps = [t0 - timedelta(hours=h) for h in [15, 12, 9, 6, 3, 0]]
        
        candidates.append({
            "t0_str": row['timestamp_utc'],
            "t0_dt": t0,
            "lat": row['latitude'],
            "lon": row['longitude'],
            "wind_kt": row['maximum_sustained_wind_kt'],
            "pres_hpa": row['central_pressure_hpa'],
            "category": row['category'],
            "t_plus_12": t_plus_12.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "t_plus_24": t_plus_24.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "t_plus_48": t_plus_48.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "history": [ts.strftime("%Y-%m-%dT%H:%M:%S+00:00") for ts in hist_steps]
        })
        
    print(f"\nTotal valid candidate t0 timestamps for AMPHAN: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"  [{i+1:02d}] t0: {c['t0_str']} | Lat: {c['lat']}, Lon: {c['lon']} | Wind: {c['wind_kt']} kt | Pres: {c['pres_hpa']} hPa | Cat: {c['category']}")
        
    if candidates:
        selected = candidates[0]
        print(f"\nEARLIEST VALID CANDIDATE SELECTED:")
        print(f"  t0: {selected['t0_str']}")
        print(f"  6 historical timestamps:")
        for h in selected['history']:
            print(f"    - {h}")
        print(f"  Targets:")
        print(f"    +12h: {selected['t_plus_12']}")
        print(f"    +24h: {selected['t_plus_24']}")
        print(f"    +48h: {selected['t_plus_48']}")

if __name__ == "__main__":
    main()
