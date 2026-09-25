import os
import pandas as pd
from datetime import timedelta

imd = pd.read_csv("data/processed/imd_best_track_v2.csv")
imd["dt"] = pd.to_datetime(imd["timestamp_utc"])
obs_set = set(zip(imd["storm_id"], imd["dt"]))

multi = pd.read_csv("data/manifests/vayu_net_multisource_sample_index.csv")
multi_train = multi[multi["split"] == "TRAIN"].copy()
multi_train["dt"] = pd.to_datetime(multi_train["t0"])

def gridsat_exists(dt):
    p = f"data/interim/gridsat/{dt.year}/gridsat_{dt.year:04d}.{dt.month:02d}.{dt.day:02d}.{dt.hour:02d}.npz"
    return os.path.exists(p)

def has_6_frames(dt):
    steps = [dt - timedelta(hours=h) for h in [15, 12, 9, 6, 3, 0]]
    return all(gridsat_exists(s) for s in steps)

train_imd = imd[(imd["split"] == "TRAIN") & (imd["year"] >= 2014)].copy()
train_imd["has_6_frames"] = train_imd["dt"].apply(has_6_frames)
valid_207 = train_imd[train_imd["has_6_frames"]].copy()

existing_keys = set(zip(multi_train["storm_id"], multi_train["dt"]))
diff_df = valid_207[valid_207.apply(lambda r: (r["storm_id"], r["dt"]) not in existing_keys, axis=1)].copy()

print(f"Total diff samples: {len(diff_df)}")
for idx, r in diff_df.iterrows():
    s = r["storm_id"]
    t0 = r["dt"]
    has_12 = (s, t0 + timedelta(hours=12)) in obs_set
    has_24 = (s, t0 + timedelta(hours=24)) in obs_set
    has_48 = (s, t0 + timedelta(hours=48)) in obs_set
    print(f"{s} {t0} -> +12h: {has_12}, +24h: {has_24}, +48h: {has_48}")
