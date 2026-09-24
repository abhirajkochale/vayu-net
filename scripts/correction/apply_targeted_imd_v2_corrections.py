"""
Targeted IMD V2 Label Correction Pipeline
Applies verified, forensic QA repairs to data/processed/imd_best_track_v2.csv and parquet.

Scope of corrections:
1. Cyclone MEKUNU 2018 (NIO_2018_MEKUNU): 37 rows on page 70 (restoring wind, category, quality_flag).
2. Cyclone GONU 2007 (NIO_2007_UNNAMED_2_3_1): 2 rows casing normalization (SUCS -> SuCS).
3. 2002 OCR coordinates (NIO_2002_UNNAMED_25 & NIO_2002_UNNAMED_29): 6 coordinate fixes from scanned tables.
4. Cyclone PHYAN 2009 (NIO_2009_PHYAN): 3 rows post-landfall column shift (wind and category).
5. Cyclone WARD 2009 (NIO_2009_WARD): 1 row raw wind '2 5' repaired to 25.0 kt.
"""

import os
import json
import pandas as pd
import numpy as np

CSV_PATH = "data/processed/imd_best_track_v2.csv"
PARQUET_PATH = "data/processed/imd_best_track_v2.parquet"
DIFF_LOG_PATH = "data/interim/imd_v2_corrections_diff.json"

def apply_corrections():
    df = pd.read_csv(CSV_PATH)
    initial_rows = len(df)
    changes = []

    def record_change(idx, storm_id, timestamp, field, old_val, new_val, reason, source_doc, page, table):
        changes.append({
            "index": int(idx),
            "storm_id": str(storm_id),
            "timestamp_utc": str(timestamp),
            "field": str(field),
            "old_value": None if pd.isna(old_val) else (str(old_val) if not isinstance(old_val, (int, float, np.integer, np.floating)) else float(old_val)),
            "new_value": None if pd.isna(new_val) else (str(new_val) if not isinstance(new_val, (int, float, np.integer, np.floating)) else float(new_val)),
            "reason": str(reason),
            "source_doc": str(source_doc),
            "page": int(page) if pd.notna(page) else None,
            "table": str(table)
        })

    # =========================================================================
    # 1. MEKUNU 2018 (37 rows on physical page 70)
    # =========================================================================
    mekunu_p70_data = [
        # (timestamp, wind, category, ci, pressure_drop)
        ("2018-05-22T06:00:00+00:00", 30.0, "DD", "2.0", 6),
        ("2018-05-22T12:00:00+00:00", 35.0, "CS", "2.5", 7),
        ("2018-05-22T15:00:00+00:00", 35.0, "CS", "2.5", 7),
        ("2018-05-22T18:00:00+00:00", 40.0, "CS", "2.5", 8),
        ("2018-05-22T21:00:00+00:00", 40.0, "CS", "2.5", 9),
        ("2018-05-23T00:00:00+00:00", 45.0, "CS", "3.0", 10),
        ("2018-05-23T03:00:00+00:00", 55.0, "SCS", "3.5", 15),
        ("2018-05-23T06:00:00+00:00", 60.0, "SCS", "3.5", 18),
        ("2018-05-23T09:00:00+00:00", 65.0, "VSCS", "3.5", 20),
        ("2018-05-23T12:00:00+00:00", 65.0, "VSCS", "4.0", 22),
        ("2018-05-23T15:00:00+00:00", 70.0, "VSCS", "4.0", 24),
        ("2018-05-23T18:00:00+00:00", 70.0, "VSCS", "4.0", 24),
        ("2018-05-23T21:00:00+00:00", 70.0, "VSCS", "4.0", 24),
        ("2018-05-24T00:00:00+00:00", 70.0, "VSCS", "4.0", 24),
        ("2018-05-24T03:00:00+00:00", 75.0, "VSCS", "4.5", 28),
        ("2018-05-24T06:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-24T09:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-24T12:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-24T15:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-24T18:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-24T21:00:00+00:00", 80.0, "VSCS", "4.5", 32),
        ("2018-05-25T00:00:00+00:00", 85.0, "VSCS", "4.5", 36),
        ("2018-05-25T03:00:00+00:00", 90.0, "ESCS", "4.5", 40),
        ("2018-05-25T06:00:00+00:00", 90.0, "ESCS", "5.0", 40),
        ("2018-05-25T09:00:00+00:00", 90.0, "ESCS", "5.0", 42),
        ("2018-05-25T12:00:00+00:00", 95.0, "ESCS", "5.0", 45),
        ("2018-05-25T15:00:00+00:00", 95.0, "ESCS", "5.0", 45),
        ("2018-05-25T18:00:00+00:00", 95.0, "ESCS", "5.0", 45),
        ("2018-05-25T21:00:00+00:00", 90.0, "ESCS", "-", 40),
        ("2018-05-26T00:00:00+00:00", 75.0, "VSCS", "-", 28),
        ("2018-05-26T03:00:00+00:00", 60.0, "SCS", "-", 18),
        ("2018-05-26T06:00:00+00:00", 50.0, "SCS", "-", 12),
        ("2018-05-26T09:00:00+00:00", 45.0, "CS", "-", 10),
        ("2018-05-26T12:00:00+00:00", 40.0, "CS", "-", 8),
        ("2018-05-26T15:00:00+00:00", 35.0, "CS", "-", 7),
        ("2018-05-26T18:00:00+00:00", 30.0, "DD", "-", 5),
        ("2018-05-27T00:00:00+00:00", 25.0, "D", "-", 3)
    ]

    for ts, wind, cat, ci, pdrop in mekunu_p70_data:
        mask = (df["storm_id"] == "NIO_2018_MEKUNU") & (df["timestamp_utc"] == ts)
        idxs = df[mask].index
        assert len(idxs) == 1, f"Expected 1 row for MEKUNU at {ts}, got {len(idxs)}"
        idx = idxs[0]

        # Check wind
        old_wind = df.at[idx, "maximum_sustained_wind_kt"]
        if pd.isna(old_wind) or old_wind != wind:
            record_change(idx, "NIO_2018_MEKUNU", ts, "maximum_sustained_wind_kt", old_wind, wind,
                          "Restored maximum sustained wind from Table 2.3.1 page 70 (resolved shifted empty column)",
                          "27_60dae9_rsmc-2018.pdf", 70, "Table 2.3.1")
            df.at[idx, "maximum_sustained_wind_kt"] = wind

        # Check category
        old_cat = df.at[idx, "category"]
        if old_cat != cat:
            record_change(idx, "NIO_2018_MEKUNU", ts, "category", old_cat, cat,
                          "Restored intended category grade from Table 2.3.1 page 70 (previously replaced by pressure drop)",
                          "27_60dae9_rsmc-2018.pdf", 70, "Table 2.3.1")
            df.at[idx, "category"] = cat

        # Quality flag & review flag
        if df.at[idx, "quality_flag"] != "VERIFIED":
            record_change(idx, "NIO_2018_MEKUNU", ts, "quality_flag", df.at[idx, "quality_flag"], "VERIFIED",
                          "Updated quality flag from SOURCE_MISSING to VERIFIED after recovering ground truth",
                          "27_60dae9_rsmc-2018.pdf", 70, "Table 2.3.1")
            df.at[idx, "quality_flag"] = "VERIFIED"

        df.at[idx, "manual_review_required"] = False
        df.at[idx, "manual_review_reason"] = "CORRECTED_POST_QA"

    # =========================================================================
    # 2. GONU 2007 (Casing normalization SUCS -> SuCS)
    # =========================================================================
    gonu_mask = (df["storm_id"] == "NIO_2007_UNNAMED_2_3_1") & (df["category"] == "SUCS")
    for idx in df[gonu_mask].index:
        ts = df.at[idx, "timestamp_utc"]
        record_change(idx, "NIO_2007_UNNAMED_2_3_1", ts, "category", "SUCS", "SuCS",
                      "Normalized category casing to canonical schema SuCS (Super Cyclonic Storm)",
                      "27_e56ca9_rsmc-2007.pdf", df.at[idx, "source_page"], df.at[idx, "source_table"])
        df.at[idx, "category"] = "SuCS"
        df.at[idx, "manual_review_reason"] = "CORRECTED_POST_QA"

    # =========================================================================
    # 3. 2002 OCR COORDINATE REPAIRS
    # =========================================================================
    coord_fixes = [
        # (storm_id, timestamp, new_lat, new_lon, page, table)
        ("NIO_2002_UNNAMED_25", "2002-11-10T03:00:00+00:00", 12.0, 82.5, 25, "Table 2.4.1"),
        ("NIO_2002_UNNAMED_25", "2002-11-10T12:00:00+00:00", 12.0, 82.5, 25, "Table 2.4.1"),
        ("NIO_2002_UNNAMED_25", "2002-11-11T00:00:00+00:00", 13.5, 82.5, 25, "Table 2.4.1"),
        ("NIO_2002_UNNAMED_25", "2002-11-12T00:00:00+00:00", 19.0, 86.5, 25, "Table 2.4.1"),
        ("NIO_2002_UNNAMED_29", "2002-11-23T12:00:00+00:00", 12.0, 87.0, 29, "Table 2.5.1"),
        ("NIO_2002_UNNAMED_29", "2002-11-25T00:00:00+00:00", 15.5, 88.0, 29, "Table 2.5.1"),
    ]

    for storm_id, ts, n_lat, n_lon, page, table in coord_fixes:
        mask = (df["storm_id"] == storm_id) & (df["timestamp_utc"] == ts)
        idxs = df[mask].index
        assert len(idxs) == 1, f"Expected 1 row for {storm_id} at {ts}, got {len(idxs)}"
        idx = idxs[0]

        old_lat = df.at[idx, "latitude"]
        old_lon = df.at[idx, "longitude"]

        if old_lat != n_lat:
            record_change(idx, storm_id, ts, "latitude", old_lat, n_lat,
                          "Corrected OCR coordinate parse error using scanned source table evidence",
                          "27_54cee6_35_6641c5_2002.pdf", page, table)
            df.at[idx, "latitude"] = n_lat

        if old_lon != n_lon:
            record_change(idx, storm_id, ts, "longitude", old_lon, n_lon,
                          "Corrected OCR coordinate parse error using scanned source table evidence",
                          "27_54cee6_35_6641c5_2002.pdf", page, table)
            df.at[idx, "longitude"] = n_lon

        df.at[idx, "manual_review_required"] = False
        df.at[idx, "manual_review_reason"] = "CORRECTED_POST_QA"

    # =========================================================================
    # 4. PHYAN 2009 (Post-landfall column shift recovery)
    # =========================================================================
    phyan_fixes = [
        # (timestamp, wind, category, pdrop)
        ("2009-11-11T12:00:00+00:00", 30.0, "DD", 5),
        ("2009-11-11T15:00:00+00:00", 30.0, "DD", 5),
        ("2009-11-11T18:00:00+00:00", 20.0, "D", 4),
    ]

    for ts, wind, cat, pdrop in phyan_fixes:
        mask = (df["storm_id"] == "NIO_2009_PHYAN") & (df["timestamp_utc"] == ts)
        idxs = df[mask].index
        assert len(idxs) == 1, f"Expected 1 row for PHYAN at {ts}, got {len(idxs)}"
        idx = idxs[0]

        old_wind = df.at[idx, "maximum_sustained_wind_kt"]
        old_cat = df.at[idx, "category"]

        if pd.isna(old_wind) or old_wind != wind:
            record_change(idx, "NIO_2009_PHYAN", ts, "maximum_sustained_wind_kt", old_wind, wind,
                          "Recovered post-landfall sustained wind from page 78 table (column shift after landfall banner)",
                          "27_4e34f3_rsmc-2009.pdf", 78, "Table on p.78")
            df.at[idx, "maximum_sustained_wind_kt"] = wind

        if pd.isna(old_cat) or old_cat != cat:
            record_change(idx, "NIO_2009_PHYAN", ts, "category", old_cat, cat,
                          "Recovered post-landfall category from page 78 table (column shift after landfall banner)",
                          "27_4e34f3_rsmc-2009.pdf", 78, "Table on p.78")
            df.at[idx, "category"] = cat

        df.at[idx, "quality_flag"] = "VERIFIED"
        df.at[idx, "manual_review_required"] = False
        df.at[idx, "manual_review_reason"] = "CORRECTED_POST_QA"

    # =========================================================================
    # 5. WARD 2009 (Wind repair '2 5' -> 25.0 kt)
    # =========================================================================
    ward_mask = (df["storm_id"] == "NIO_2009_WARD") & (df["timestamp_utc"] == "2009-12-15T00:00:00+00:00")
    idxs = df[ward_mask].index
    assert len(idxs) == 1, f"Expected 1 row for WARD, got {len(idxs)}"
    idx = idxs[0]
    old_wind = df.at[idx, "maximum_sustained_wind_kt"]
    record_change(idx, "NIO_2009_WARD", "2009-12-15T00:00:00+00:00", "maximum_sustained_wind_kt", old_wind, 25.0,
                  "Repaired OCR whitespace split '2 5' to intended 25.0 kt for Depression (D)",
                  "27_4e34f3_rsmc-2009.pdf", 83, "Table on p.83")
    df.at[idx, "maximum_sustained_wind_kt"] = 25.0
    df.at[idx, "manual_review_required"] = False
    df.at[idx, "manual_review_reason"] = "CORRECTED_POST_QA"

    # =========================================================================
    # INTEGRITY CHECKS BEFORE SAVING
    # =========================================================================
    assert len(df) == initial_rows, f"Row count changed: {initial_rows} -> {len(df)}"

    # Save outputs
    df.to_csv(CSV_PATH, index=False)
    df.to_parquet(PARQUET_PATH, index=False)

    os.makedirs(os.path.dirname(DIFF_LOG_PATH), exist_ok=True)
    with open(DIFF_LOG_PATH, "w") as f:
        json.dump(changes, f, indent=2)

    print(f"Successfully applied {len(changes)} targeted corrections to {len(set(c['index'] for c in changes))} rows.")
    print(f"Saved {CSV_PATH} and {PARQUET_PATH}.")
    print(f"Diff log saved to {DIFF_LOG_PATH}.")

if __name__ == "__main__":
    apply_corrections()
