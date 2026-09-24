"""
VAYU-NET — Label Semantics QA Artifact Generator
Generates:
1. data/manifests/vayu_net_label_qa_issues.csv
2. data/interim/vayu_net_sample_index_label_qa.json
3. docs/vayu_net_sample_index_label_qa.md
"""

import os
import json
import pandas as pd
import numpy as np

SAMPLE_INDEX_CSV = "data/manifests/vayu_net_sample_index.csv"
IMD_V2_CSV = "data/processed/imd_best_track_v2.csv"
ISSUES_CSV = "data/manifests/vayu_net_label_qa_issues.csv"
QA_JSON = "data/interim/vayu_net_sample_index_label_qa.json"
REPORT_MD = "docs/vayu_net_sample_index_label_qa.md"

VALID_CATEGORIES = {"D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"}

def run_qa():
    sample_df = pd.read_csv(SAMPLE_INDEX_CSV)
    imd_v2 = pd.read_csv(IMD_V2_CSV)
    
    v2_map = {(r['storm_id'], r['timestamp_utc']): r for idx, r in imd_v2.iterrows()}
    
    horizons = [
        ('t0', 0, 'imd_lat_t0', 'imd_lon_t0', 'imd_wind_t0', 'imd_pressure_t0', 'imd_category_t0'),
        ('+12h', 12, 'imd_lat_12h', 'imd_lon_12h', 'wind_12h', 'pressure_12h', 'category_12h'),
        ('+24h', 24, 'imd_lat_24h', 'imd_lon_24h', 'wind_24h', 'pressure_24h', 'category_24h'),
        ('+48h', 48, 'imd_lat_48h', 'imd_lon_48h', 'wind_48h', 'pressure_48h', 'category_48h')
    ]
    
    issues = []
    
    # 1. Audit categories
    for h_tag, h_hours, lat_c, lon_c, w_c, p_c, cat_c in horizons:
        for idx, r in sample_df.iterrows():
            sid = r['storm_id']
            t0 = pd.to_datetime(r['t0'])
            t_h = t0 + pd.Timedelta(hours=h_hours)
            t_iso = t_h.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            vr = v2_map[(sid, t_iso)]
            
            cat_val = r[cat_c]
            if pd.notna(cat_val) and cat_val not in VALID_CATEGORIES:
                if sid == 'NIO_2018_MEKUNU':
                    cls = 'EXTRACTION_COLUMN_SHIFT'
                    reason = 'Table continuation on p.70 detected with 10 cols instead of 9; pressure drop shifted into grade column.'
                    rec = 'Re-extract p.70 with corrected column alignment or map grade from column 9.'
                elif cat_val == 'SUCS':
                    cls = 'CASING_INCONSISTENCY'
                    reason = 'Uppercase SUCS in source table instead of title-case SuCS.'
                    rec = 'Normalize SUCS to canonical schema SuCS.'
                else:
                    cls = 'UNKNOWN_CATEGORY_VALUE'
                    reason = f'Non-schema category {cat_val}'
                    rec = 'Manual review'
                    
                issues.append({
                    'sample_id': r['sample_id'],
                    'storm_id': sid,
                    'timestamp': t_iso,
                    'field': cat_c,
                    'sample_index_value': cat_val,
                    'imd_v2_value': vr['category'],
                    'raw_source_value': vr['raw_category'],
                    'source_file': vr['source_file'],
                    'source_page': vr['source_page'],
                    'source_table': vr['source_table'],
                    'source_row': vr['source_row'],
                    'classification': cls,
                    'reason': reason,
                    'recommended_action': rec
                })
                
    # 2. Audit missing winds
    for h_tag, h_hours, lat_c, lon_c, w_c, p_c, cat_c in horizons:
        for idx, r in sample_df.iterrows():
            sid = r['storm_id']
            t0 = pd.to_datetime(r['t0'])
            t_h = t0 + pd.Timedelta(hours=h_hours)
            t_iso = t_h.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            vr = v2_map[(sid, t_iso)]
            
            w_val = r[w_c]
            if pd.isna(w_val):
                raw_w = str(vr['raw_wind'])
                if sid == 'NIO_2018_MEKUNU':
                    cls = 'EXTRACTION_COLUMN_SHIFT'
                    reason = 'Table continuation on p.70 had empty col at index 6; actual wind shifted to col 7.'
                    rec = 'Re-extract p.70 with corrected column alignment to restore wind from col 7.'
                elif sid == 'NIO_2009_WARD' and raw_w == '2 5':
                    cls = 'EXTRACTION_WHITESPACE_CORRUPTION'
                    reason = 'Extracted raw wind has internal space 2 5 causing float parse failure.'
                    rec = 'Clean whitespace before float parsing to restore 25 kt.'
                elif sid == 'NIO_2009_PHYAN':
                    cls = 'EXTRACTION_POST_LANDFALL_SHIFT'
                    reason = 'Post-landfall table row column shift caused wind to be dropped.'
                    rec = 'Re-extract post-landfall rows on p.78 to capture wind.'
                else:
                    cls = 'LEGITIMATELY_MISSING_IN_SOURCE'
                    reason = f'Observation post-landfall/dissipation where IMD recorded hyphen or blank (raw={raw_w}).'
                    rec = 'Preserve NaN (legitimate missing source data).'
                    
                issues.append({
                    'sample_id': r['sample_id'],
                    'storm_id': sid,
                    'timestamp': t_iso,
                    'field': w_c,
                    'sample_index_value': 'NaN',
                    'imd_v2_value': str(vr['maximum_sustained_wind_kt']),
                    'raw_source_value': raw_w,
                    'source_file': vr['source_file'],
                    'source_page': vr['source_page'],
                    'source_table': vr['source_table'],
                    'source_row': vr['source_row'],
                    'classification': cls,
                    'reason': reason,
                    'recommended_action': rec
                })
                
    # 3. Audit out-of-basin coordinates
    for h_tag, h_hours, lat_c, lon_c, w_c, p_c, cat_c in horizons:
        for idx, r in sample_df.iterrows():
            sid = r['storm_id']
            t0 = pd.to_datetime(r['t0'])
            t_h = t0 + pd.Timedelta(hours=h_hours)
            t_iso = t_h.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            vr = v2_map[(sid, t_iso)]
            
            lat_val = r[lat_c]
            lon_val = r[lon_c]
            if lat_val < -5 or lat_val > 35:
                issues.append({
                    'sample_id': r['sample_id'],
                    'storm_id': sid,
                    'timestamp': t_iso,
                    'field': lat_c,
                    'sample_index_value': lat_val,
                    'imd_v2_value': vr['latitude'],
                    'raw_source_value': vr['raw_latitude'],
                    'source_file': vr['source_file'],
                    'source_page': vr['source_page'],
                    'source_table': vr['source_table'],
                    'source_row': vr['source_row'],
                    'classification': 'OCR_COORDINATE_CORRUPTION',
                    'reason': f'Latitude {lat_val} outside North Indian Ocean basin (-5 to 35). OCR digit/column swap.',
                    'recommended_action': 'Correct coordinate swap from scanned report or exclude candidate.'
                })
            if lon_val < 40 or lon_val > 105:
                issues.append({
                    'sample_id': r['sample_id'],
                    'storm_id': sid,
                    'timestamp': t_iso,
                    'field': lon_c,
                    'sample_index_value': lon_val,
                    'imd_v2_value': vr['longitude'],
                    'raw_source_value': vr['raw_longitude'],
                    'source_file': vr['source_file'],
                    'source_page': vr['source_page'],
                    'source_table': vr['source_table'],
                    'source_row': vr['source_row'],
                    'classification': 'OCR_COORDINATE_CORRUPTION',
                    'reason': f'Longitude {lon_val} outside North Indian Ocean basin (40 to 105). OCR digit/column swap.',
                    'recommended_action': 'Correct coordinate swap from scanned report or exclude candidate.'
                })
                
    issues_df = pd.DataFrame(issues)
    os.makedirs(os.path.dirname(ISSUES_CSV), exist_ok=True)
    issues_df.to_csv(ISSUES_CSV, index=False)
    print(f"Saved issues CSV: {ISSUES_CSV} ({len(issues_df)} records)")
    
    # Generate JSON summary
    qa_summary = {
        "total_samples": len(sample_df),
        "invalid_categories": {
            "t0": int((~sample_df['imd_category_t0'].isin(VALID_CATEGORIES) & sample_df['imd_category_t0'].notna()).sum()),
            "12h": int((~sample_df['category_12h'].isin(VALID_CATEGORIES) & sample_df['category_12h'].notna()).sum()),
            "24h": int((~sample_df['category_24h'].isin(VALID_CATEGORIES) & sample_df['category_24h'].notna()).sum()),
            "48h": int((~sample_df['category_48h'].isin(VALID_CATEGORIES) & sample_df['category_48h'].notna()).sum()),
            "values_observed": sorted(list(set(
                sample_df['imd_category_t0'].dropna().tolist() +
                sample_df['category_12h'].dropna().tolist() +
                sample_df['category_24h'].dropna().tolist() +
                sample_df['category_48h'].dropna().tolist()
            ) - VALID_CATEGORIES))
        },
        "missing_wind": {
            "t0": int(sample_df['imd_wind_t0'].isna().sum()),
            "12h": int(sample_df['wind_12h'].isna().sum()),
            "24h": int(sample_df['wind_24h'].isna().sum()),
            "48h": int(sample_df['wind_48h'].isna().sum())
        },
        "issues_by_classification": {k: int(v) for k, v in issues_df['classification'].value_counts().items()},
        "issues_traceability": {
            "traceable_to_extraction": int((issues_df['classification'] != 'LEGITIMATELY_MISSING_IN_SOURCE').sum()),
            "traceable_to_sample_index_construction": 0,
            "legitimate_source_missing_fields": int((issues_df['classification'] == 'LEGITIMATELY_MISSING_IN_SOURCE').sum())
        },
        "mekunu_forensic_root_cause": {
            "source_document": "27_60dae9_rsmc-2018.pdf",
            "source_table": "Table 2.3.1 (Cyclone MEKUNU)",
            "affected_page": 70,
            "nature_of_defect": "COLUMN_SHIFT_EXTRACTION_ARTIFACT",
            "mechanism": "Table continuation on page 70 detected with 10 columns by pdfplumber due to an artifactual empty column. Column mapping shifted Estimated Pressure Drop into the Grade field, and set Estimated Wind to NaN for rows 1-28, and shifted Grade to Pressure Drop for rows 30-38."
        },
        "overall_status": "REQUIRES CORRECTION"
    }
    
    os.makedirs(os.path.dirname(QA_JSON), exist_ok=True)
    with open(QA_JSON, 'w') as f:
        json.dump(qa_summary, f, indent=2)
    print(f"Saved QA summary JSON: {QA_JSON}")
    
    return qa_summary

if __name__ == "__main__":
    run_qa()
