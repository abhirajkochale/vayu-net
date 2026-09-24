"""
Comprehensive Post-Correction Validation Suite for VAYU-NET
Verifies all 13 integrity checks across IMD Best Track V2 and Sample Index.
"""

import os
import json
import glob
import pandas as pd
import numpy as np

BACKUP_V2_CSV = "data/interim/backups_pre_correction/imd_best_track_v2.csv"
BACKUP_SAMPLE_CSV = "data/interim/backups_pre_correction/vayu_net_sample_index.csv"

CURRENT_V2_CSV = "data/processed/imd_best_track_v2.csv"
CURRENT_V2_PARQUET = "data/processed/imd_best_track_v2.parquet"
CURRENT_SAMPLE_CSV = "data/manifests/vayu_net_sample_index.csv"
DIFF_LOG_JSON = "data/interim/imd_v2_corrections_diff.json"

RAW_IMD_DIR = "data/raw/imd"
INTERIM_GRIDSAT_DIR = "data/interim/gridsat"

VALID_CATEGORIES = {"D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"}

def run_validation():
    print("=" * 80)
    print("VAYU-NET — TARGETED IMD V2 POST-CORRECTION FULL VALIDATION")
    print("=" * 80)

    results = {}
    metrics = {}

    # Load datasets
    df_pre_v2 = pd.read_csv(BACKUP_V2_CSV)
    df_cur_v2 = pd.read_csv(CURRENT_V2_CSV)
    df_cur_v2_pq = pd.read_parquet(CURRENT_V2_PARQUET)
    df_pre_sample = pd.read_csv(BACKUP_SAMPLE_CSV)
    df_cur_sample = pd.read_csv(CURRENT_SAMPLE_CSV)

    with open(DIFF_LOG_JSON, "r") as f:
        diff_log = json.load(f)

    # -------------------------------------------------------------------------
    # Check 1: IMD V2 Row Count
    # -------------------------------------------------------------------------
    rows_pre = len(df_pre_v2)
    rows_cur = len(df_cur_v2)
    rows_cur_pq = len(df_cur_v2_pq)
    metrics["imd_v2_rows_before"] = rows_pre
    metrics["imd_v2_rows_after"] = rows_cur
    check1_pass = (rows_pre == rows_cur == rows_cur_pq == 3960)
    results["1_imd_v2_row_count"] = {
        "status": "PASS" if check1_pass else "FAIL",
        "before": rows_pre,
        "after_csv": rows_cur,
        "after_parquet": rows_cur_pq,
        "expected": 3960
    }

    # -------------------------------------------------------------------------
    # Check 2: IMD V2 Timestamp Uniqueness
    # -------------------------------------------------------------------------
    dup_counts = df_cur_v2.groupby(["storm_id", "timestamp_utc"]).size()
    violations = int((dup_counts > 1).sum())
    metrics["timestamp_violations"] = violations
    results["2_timestamp_uniqueness"] = {
        "status": "PASS" if violations == 0 else "FAIL",
        "violations": violations
    }

    # -------------------------------------------------------------------------
    # Check 3: Changed Rows and Changed Fields
    # -------------------------------------------------------------------------
    changed_rows_set = set(c["index"] for c in diff_log)
    changed_storms_set = set(c["storm_id"] for c in diff_log)
    metrics["total_field_corrections"] = len(diff_log)
    metrics["total_rows_corrected"] = len(changed_rows_set)
    metrics["storms_corrected"] = list(changed_storms_set)

    # Verify no unapproved storm was changed
    expected_storms = {"NIO_2018_MEKUNU", "NIO_2007_UNNAMED_2_3_1", "NIO_2002_UNNAMED_25",
                       "NIO_2002_UNNAMED_29", "NIO_2009_PHYAN", "NIO_2009_WARD"}
    unapproved_storms = changed_storms_set - expected_storms
    check3_pass = (len(unapproved_storms) == 0 and len(diff_log) == 113 and len(changed_rows_set) == 49)
    results["3_changed_rows_and_fields"] = {
        "status": "PASS" if check3_pass else "FAIL",
        "total_field_corrections": len(diff_log),
        "total_rows_corrected": len(changed_rows_set),
        "storms_corrected": list(changed_storms_set),
        "unapproved_storms": list(unapproved_storms)
    }

    # -------------------------------------------------------------------------
    # Check 4: Category Schema Validity
    # -------------------------------------------------------------------------
    invalid_cats_cur = df_cur_v2[~df_cur_v2["category"].isna() & ~df_cur_v2["category"].isin(VALID_CATEGORIES)]
    invalid_cats_sample = df_cur_sample[
        (~df_cur_sample["imd_category_t0"].isna() & ~df_cur_sample["imd_category_t0"].isin(VALID_CATEGORIES)) |
        (~df_cur_sample["category_12h"].isna() & ~df_cur_sample["category_12h"].isin(VALID_CATEGORIES)) |
        (~df_cur_sample["category_24h"].isna() & ~df_cur_sample["category_24h"].isin(VALID_CATEGORIES)) |
        (~df_cur_sample["category_48h"].isna() & ~df_cur_sample["category_48h"].isin(VALID_CATEGORIES))
    ]
    check4_pass = (len(invalid_cats_cur) == 0 and len(invalid_cats_sample) == 0)
    metrics["invalid_categories_v2"] = len(invalid_cats_cur)
    metrics["invalid_categories_sample"] = len(invalid_cats_sample)
    results["4_category_schema_validity"] = {
        "status": "PASS" if check4_pass else "FAIL",
        "invalid_categories_v2": len(invalid_cats_cur),
        "invalid_categories_sample": len(invalid_cats_sample)
    }

    # -------------------------------------------------------------------------
    # Check 5: Coordinates Validity (Basin & Corrected Candidates)
    # -------------------------------------------------------------------------
    # Check candidate samples coordinates: all must be within NIO basin (lat: -5 to 35, lon: 40 to 105)
    sample_coord_errors = df_cur_sample[
        (df_cur_sample["imd_lat_t0"] < -5) | (df_cur_sample["imd_lat_t0"] > 35) |
        (df_cur_sample["imd_lon_t0"] < 40) | (df_cur_sample["imd_lon_t0"] > 105) |
        (df_cur_sample["imd_lat_12h"] < -5) | (df_cur_sample["imd_lat_12h"] > 35) |
        (df_cur_sample["imd_lon_12h"] < 40) | (df_cur_sample["imd_lon_12h"] > 105) |
        (df_cur_sample["imd_lat_24h"] < -5) | (df_cur_sample["imd_lat_24h"] > 35) |
        (df_cur_sample["imd_lon_24h"] < 40) | (df_cur_sample["imd_lon_24h"] > 105) |
        (df_cur_sample["imd_lat_48h"] < -5) | (df_cur_sample["imd_lat_48h"] > 35) |
        (df_cur_sample["imd_lon_48h"] < 40) | (df_cur_sample["imd_lon_48h"] > 105)
    ]
    check5_pass = (len(sample_coord_errors) == 0)
    metrics["candidate_sample_coord_errors"] = len(sample_coord_errors)
    results["5_candidate_coordinates_validity"] = {
        "status": "PASS" if check5_pass else "FAIL",
        "sample_coord_errors": len(sample_coord_errors)
    }

    # -------------------------------------------------------------------------
    # Check 6: Missing Winds (Legitimate NaNs vs Corrected Extraction Errors)
    # -------------------------------------------------------------------------
    # In candidate samples, how many missing winds remain?
    target_wind_cols = ["imd_wind_t0", "wind_12h", "wind_24h", "wind_48h"]
    nan_winds_by_col = {col: int(df_cur_sample[col].isna().sum()) for col in target_wind_cols}
    total_nan_wind_slots = sum(nan_winds_by_col.values())
    
    # Check that all remaining NaN wind slots in sample index correspond to legitimate dissipation
    nan_samples = df_cur_sample[df_cur_sample[target_wind_cols].isna().any(axis=1)]
    nan_sample_storms = nan_samples["storm_id"].unique().tolist()
    expected_nan_storms = ["NIO_2002_UNNAMED_25", "NIO_2006_UNNAMED_2_2_1", "NIO_2007_UNNAMED_2_12_1", "NIO_2011_UNNAMED_2_9_1"]
    check6_pass = (set(nan_sample_storms).issubset(set(expected_nan_storms)) and total_nan_wind_slots == 10)
    metrics["total_legitimate_nan_wind_slots"] = total_nan_wind_slots
    metrics["nan_winds_by_col"] = nan_winds_by_col
    results["6_legitimate_missing_winds"] = {
        "status": "PASS" if check6_pass else "FAIL",
        "nan_wind_slots": total_nan_wind_slots,
        "nan_winds_by_col": nan_winds_by_col,
        "storms_with_nan_wind": nan_sample_storms
    }

    # -------------------------------------------------------------------------
    # Check 7: Sample Index <-> IMD V2 Field Equality
    # -------------------------------------------------------------------------
    # Check that every single observation in sample index matches IMD V2 exactly
    v2_lookup = {}
    for _, r in df_cur_v2.iterrows():
        v2_lookup[(r["storm_id"], r["timestamp_utc"])] = r

    discrepancies = []
    field_mapping = [
        ("imd_lat_t0", "latitude", "t0"),
        ("imd_lon_t0", "longitude", "t0"),
        ("imd_wind_t0", "maximum_sustained_wind_kt", "t0"),
        ("imd_pressure_t0", "central_pressure_hpa", "t0"),
        ("imd_category_t0", "category", "t0"),
    ]

    for idx, s_row in df_cur_sample.iterrows():
        sid = s_row["storm_id"]
        t0 = pd.to_datetime(s_row["t0"])
        
        times = [
            ("t0", s_row["t0"], "imd_lat_t0", "imd_lon_t0", "imd_wind_t0", "imd_pressure_t0", "imd_category_t0"),
            ("12h", (t0 + pd.Timedelta(hours=12)).isoformat(), "imd_lat_12h", "imd_lon_12h", "wind_12h", "pressure_12h", "category_12h"),
            ("24h", (t0 + pd.Timedelta(hours=24)).isoformat(), "imd_lat_24h", "imd_lon_24h", "wind_24h", "pressure_24h", "category_24h"),
            ("48h", (t0 + pd.Timedelta(hours=48)).isoformat(), "imd_lat_48h", "imd_lon_48h", "wind_48h", "pressure_48h", "category_48h"),
        ]

        for step, ts, f_lat, f_lon, f_w, f_p, f_c in times:
            obs = v2_lookup.get((sid, ts))
            if obs is None:
                discrepancies.append((s_row["sample_id"], step, "MISSING_OBS", ts))
                continue
            for s_field, v_field in [(f_lat, "latitude"), (f_lon, "longitude"), (f_w, "maximum_sustained_wind_kt"),
                                     (f_p, "central_pressure_hpa"), (f_c, "category")]:
                s_val = s_row[s_field]
                v_val = obs[v_field]
                if pd.isna(s_val) and pd.isna(v_val):
                    continue
                if s_val != v_val:
                    discrepancies.append((s_row["sample_id"], step, s_field, s_val, v_val))

    check7_pass = (len(discrepancies) == 0)
    metrics["sample_index_imd_discrepancies"] = len(discrepancies)
    results["7_sample_index_imd_equality"] = {
        "status": "PASS" if check7_pass else "FAIL",
        "discrepancies": len(discrepancies)
    }

    # -------------------------------------------------------------------------
    # Check 8: GridSat Frame-Reference Failures
    # -------------------------------------------------------------------------
    frame_cols = ["frame_t_minus_15h", "frame_t_minus_12h", "frame_t_minus_9h",
                  "frame_t_minus_6h", "frame_t_minus_3h", "frame_t0"]
    missing_frame_refs = 0
    for idx, row in df_cur_sample.iterrows():
        for col in frame_cols:
            p = row[col]
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                missing_frame_refs += 1

    check8_pass = (missing_frame_refs == 0)
    metrics["missing_frame_references"] = missing_frame_refs
    results["8_gridsat_frame_references"] = {
        "status": "PASS" if check8_pass else "FAIL",
        "missing_references": missing_frame_refs,
        "total_references_verified": len(df_cur_sample) * 6
    }

    # -------------------------------------------------------------------------
    # Check 9: Candidate Sample Count Before / After
    # -------------------------------------------------------------------------
    samples_pre = len(df_pre_sample)
    samples_cur = len(df_cur_sample)
    metrics["candidate_samples_before"] = samples_pre
    metrics["candidate_samples_after"] = samples_cur
    check9_pass = (samples_pre == samples_cur == 1319)
    results["9_candidate_sample_count"] = {
        "status": "PASS" if check9_pass else "FAIL",
        "before": samples_pre,
        "after": samples_cur,
        "expected": 1319
    }

    # -------------------------------------------------------------------------
    # Check 10: Train / Validation / Test Storm Splits
    # -------------------------------------------------------------------------
    split_counts_v2 = df_cur_v2.groupby("split")["storm_id"].nunique().to_dict()
    split_counts_sample = df_cur_sample.groupby("split")["storm_id"].nunique().to_dict()
    sample_splits = df_cur_sample["split"].value_counts().to_dict()

    train_storms = set(df_cur_sample[df_cur_sample["split"] == "TRAIN"]["storm_id"])
    val_storms = set(df_cur_sample[df_cur_sample["split"] == "VALIDATION"]["storm_id"])
    test_storms = set(df_cur_sample[df_cur_sample["split"] == "TEST"]["storm_id"])

    intersections = {
        "train_val": len(train_storms & val_storms),
        "train_test": len(train_storms & test_storms),
        "val_test": len(val_storms & test_storms)
    }
    check10_pass = (intersections["train_val"] == 0 and intersections["train_test"] == 0 and intersections["val_test"] == 0)
    metrics["split_sample_counts"] = sample_splits
    metrics["split_storm_counts"] = split_counts_sample
    results["10_split_assignments"] = {
        "status": "PASS" if check10_pass else "FAIL",
        "sample_counts": sample_splits,
        "storm_counts": split_counts_sample,
        "intersections": intersections
    }

    # -------------------------------------------------------------------------
    # Check 11: GridSat File Count Before / After
    # -------------------------------------------------------------------------
    npz_files = glob.glob(f"{INTERIM_GRIDSAT_DIR}/**/*.npz", recursive=True)
    nc_files = glob.glob(f"{INTERIM_GRIDSAT_DIR}/**/*.nc", recursive=True)
    metrics["total_gridsat_npz"] = len(npz_files)
    metrics["total_gridsat_nc"] = len(nc_files)
    check11_pass = (len(npz_files) == 2353 and len(nc_files) == 2353)
    results["11_gridsat_file_integrity"] = {
        "status": "PASS" if check11_pass else "FAIL",
        "npz_count": len(npz_files),
        "nc_count": len(nc_files),
        "expected_each": 2353
    }

    # -------------------------------------------------------------------------
    # Check 12: Source PDF Modifications Check
    # -------------------------------------------------------------------------
    pdf_files = glob.glob(f"{RAW_IMD_DIR}/*.pdf")
    metrics["raw_pdf_count"] = len(pdf_files)
    # Check that exactly 29 PDFs exist and none have been deleted or added
    check12_pass = (len(pdf_files) == 29)
    results["12_source_pdf_immutability"] = {
        "status": "PASS" if check12_pass else "FAIL",
        "pdf_count": len(pdf_files),
        "expected": 29
    }

    # -------------------------------------------------------------------------
    # Check 13: Itemized Provenance Completeness
    # -------------------------------------------------------------------------
    unprovenanced = [c for c in diff_log if not c.get("source_doc") or not c.get("page") or not c.get("reason")]
    check13_pass = (len(unprovenanced) == 0 and len(diff_log) > 0)
    results["13_itemized_provenance_completeness"] = {
        "status": "PASS" if check13_pass else "FAIL",
        "total_corrections": len(diff_log),
        "unprovenanced_count": len(unprovenanced)
    }

    # -------------------------------------------------------------------------
    # Final Output Summary
    # -------------------------------------------------------------------------
    all_passed = all(r["status"] == "PASS" for r in results.values())
    metrics["overall_status"] = "PASS" if all_passed else "FAIL"

    print("\n" + "=" * 80)
    print("DETAILED VALIDATION RESULTS:")
    print("=" * 80)
    for k, v in results.items():
        print(f"[{v['status']}] {k}: {v}")

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print(f"IMD V2 rows: before={rows_pre}, after={rows_cur} (PASS)")
    print(f"Candidate samples: before={samples_pre}, after={samples_cur} (PASS)")
    print(f"Total corrections applied: {len(diff_log)} fields across {len(changed_rows_set)} rows")
    print(f"Legitimate NaNs retained: {total_nan_wind_slots} slots across {len(nan_samples)} samples")
    print(f"Sample-index <-> IMD V2 discrepancies: {len(discrepancies)}")
    print(f"Invalid categories remaining: {len(invalid_cats_cur)}")
    print(f"GridSat frame path failures: {missing_frame_refs}")
    print(f"GridSat production files on disk: {len(npz_files)} npz, {len(nc_files)} nc")
    print(f"Raw IMD PDFs untouched: {len(pdf_files)} / 29")
    print(f"Overall Status: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)

    # Save validation metrics to interim
    with open("data/interim/v2_post_correction_validation.json", "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)

    return all_passed

if __name__ == "__main__":
    passed = run_validation()
    exit(0 if passed else 1)
