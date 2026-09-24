# VAYU-NET — Supabase Storage Architecture & Layout

**Module:** Cloud Infrastructure & Object Storage  
**Project:** VAYU-NET — North Indian Ocean Tropical Cyclone Intelligence System  
**Baseline:** SIH 2026 Problem Statement 26070

---

## 1. Storage Purpose & Philosophy

VAYU-NET utilizes **Supabase Storage** (S3-compatible object storage backed by PostgreSQL row-level security) to host production model weights, runtime metadata, and curated demo satellite visualization assets. 

> [!IMPORTANT]
> **Zero Raw Dataset Ingestion:**  
> The 7.7+ GB raw satellite and historical PDF archives must **NEVER** be uploaded to Supabase Storage. Supabase Storage is reserved strictly for operational runtime assets (~115 MB total).

---

## 2. Bucket Architecture & Access Control

```
Supabase Project
 ├── Bucket: vayu-net-runtime (PRIVATE, Read: Service Role / Signed URLs)
 │    ├── models/
 │    └── cache/
 └── Bucket: vayu-net-public (PUBLIC, CDN-accelerated)
      ├── explainability/
      ├── demo/
      └── metadata/
```

| Bucket Name | Access Intent | Access Mechanism | Max Size | Content |
| :--- | :--- | :--- | :--- | :--- |
| `vayu-net-runtime` | **PRIVATE** | Server-side only (Backend `SUPABASE_SERVICE_ROLE_KEY` or signed URLs) | ~150 MB | PyTorch model weights (`.pt`), sensitive configuration caches |
| `vayu-net-public` | **PUBLIC** | Direct HTTPS CDN URL | ~30 MB | Pre-rendered Grad-CAM overlays, demo frame PNGs, public manifests |

---

## 3. Directory Layout Specification

### 3.1 `vayu-net-runtime/` (Private Bucket)

```
vayu-net-runtime/
 └── models/
      ├── center_localization_cnn/
      │    └── v1.0/
      │         ├── best_center_localization_cnn.pt  (49.0 MB, SHA-256: 5ef11b59...)
      │         └── metadata.json
      ├── temporal_track_gru/
      │    └── v1.0/
      │         ├── best_temporal_track_gru.pt       (46.2 MB, SHA-256: 7973c12a...)
      │         └── metadata.json
      ├── environment_hybrid_track_gru/
      │    └── v1.0/
      │         ├── best_phase5b_variant_b.pt        (1.7 MB, SHA-256: 2f0ec1e1...)
      │         └── metadata.json
      └── intensity_wind_multitask/
           └── v1.0/
                ├── best_phase6_intensity_wind.pt    (1.4 MB, SHA-256: a014e06d...)
                └── metadata.json
```

### 3.2 `vayu-net-public/` (Public CDN Bucket)

```
vayu-net-public/
 ├── explainability/
 │    ├── fani_2019/
 │    │    ├── original.png
 │    │    ├── heatmap.png
 │    │    ├── overlay.png
 │    │    └── metadata.json
 │    ├── amphan_2020/
 │    │    ├── original.png
 │    │    ├── heatmap.png
 │    │    ├── overlay.png
 │    │    └── metadata.json
 │    ├── tauktae_2021/
 │    │    ├── ...
 │    ├── biparjoy_2023/
 │    │    ├── ...
 │    └── remal_2024/
 │         ├── ...
 ├── metadata/
 │    ├── train_normalization_stats.json
 │    ├── uncertainty_parameters.json
 │    ├── storm_event_manifest_v2.json
 │    └── explainability_manifest.json
 └── demo/
      └── sample_cyclone_index.json
```

---

## 4. Object Metadata Standards

Every object stored in Supabase must include standard HTTP headers and custom metadata:

```json
{
  "Content-Type": "application/octet-stream", // or "image/png", "application/json"
  "Cache-Control": "public, max-age=31536000, immutable",
  "metadata": {
    "project": "vayu-net",
    "version": "1.0.0",
    "sha256": "5ef11b596ef94ece71079c1329eab1e50191127eb7b9b1ff13f7dcdd3185f1c9",
    "sih_problem": "SIH26070",
    "uploaded_by": "devops_automation"
  }
}
```

---

## 5. Security & Key Separation Invariants

> [!CAUTION]
> ### STRICT SECURITY BOUNDARIES
> 1. **Zero Service Role Keys in Frontend:**  
>    The frontend application (`apps/frontend`) must **NEVER** receive or bundle `SUPABASE_SERVICE_ROLE_KEY`. It may only consume `VITE_SUPABASE_ANON_KEY` or read through the backend API.
> 2. **Backend Server-Side Isolation:**  
>    Only `apps/backend` running on Render or local servers may read from `vayu-net-runtime`.
> 3. **Public Bucket Read-Only Policies:**  
>    Row-Level Security (RLS) policies on `vayu-net-public` must enforce `SELECT` permissions for `anon`, with `INSERT/UPDATE/DELETE` strictly restricted to `authenticated` service roles.
