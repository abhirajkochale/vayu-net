# IMD Best Track Dataset QA v2 Executive Summary

**Pipeline:** VAYU-NET (SIH Problem Statement 26070)  
**Date:** 2026-09-23T07:12:17.491670+00:00  
**QA Verdict:** **PASS**  

### Final Key Metrics
1. **Total Clean ML-Ready Observations:** **3960** across **216** unique storms.
2. **Reconciliation Consistency:** Reconciled all manifests and reports to consistently state **216** unique storms.
3. **Verified vs Source-Missing Breakdown:** 3879 fully complete verified observations; 81 source-missing observations (overland depression stages with verified coordinates and timestamps).
4. **Unresolved Ambiguities Eliminated:** 50 Class C continuation table rows in 2018 excluded from V2 ML-ready training set and logged to manual review.
5. **2003 Anomaly Solved:** Recovered all 7 cyclonic disturbances and 19 observations by auditing the 9 actual Best Track table pages in the 2003 report.
6. **Integrity & Safety:** All 14 automated tests passed; zero duplicate timestamps; zero missing timestamps; zero invalid coordinates; 100% SHA-256 match on source PDFs; zero GridSat downloaded.

### Output Paths
- Canonical Clean CSV: `data/processed/imd_best_track_v2.csv`
- Columnar Parquet: `data/processed/imd_best_track_v2.parquet`
- Storm Manifest: `data/manifests/storm_event_manifest_v2.csv`
- Manual Review Log: `data/manifests/imd_manual_review_v2.csv`
- Excluded Records Log: `data/manifests/imd_excluded_records_v2.csv`
- Full Final QA Report: `docs/imd_best_track_qa_v2_final_report.md`
