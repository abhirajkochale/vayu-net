# VAYU-NET — Supabase Runtime Model & Asset Storage Architecture

**Problem Statement:** SIH 2026 PS 26070  
**Ministry / Department:** Ministry of Earth Sciences (MoES) / India Meteorological Department (IMD)  
**Security Classification:** Restricted Model Assets (Internal Service Access Only)  
**Status:** `NOT YET CONFIGURED — MANUAL DEPLOYMENT STEP`

> [!NOTE]
> **Cloud Storage Status:** Cloud deployment has not yet been performed. Manual provisioning and upload are pending.
> All model checkpoints and runtime assets are fully present and **VERIFIED LOCALLY**. This document provides the blueprint for manual cloud configuration.

---

## 1. Storage Architecture Overview

To support cloud-native deployments on Render without bloating git repositories with large neural network binary checkpoints, model weights and precomputed calibration assets are archived in private object storage (**Supabase Storage**).

```
vayu-net-runtime (Private Bucket)
│
├── models/
│   ├── phase3c/
│   │   └── best_center_localization_cnn_v1.0.pt
│   ├── phase4a/
│   │   └── best_temporal_track_gru_v1.0.pt
│   ├── phase5b/
│   │   ├── best_phase5b_variant_a_v1.0.pt
│   │   └── best_phase5b_variant_b_v1.0.pt
│   └── phase6/
│       └── best_phase6_intensity_wind_v1.0.pt
│
├── runtime/
│   ├── train_normalization_stats_v1.0.json
│   ├── era5_train_normalization_stats_v1.0.json
│   ├── uncertainty_parameters_v1.0.json
│   └── analog_retrieval_cache_v1.0.json
│
├── demo/
│   ├── phase5b_hybrid_results_v1.0.json
│   ├── vayu_net_sample_index_v1.0.csv
│   └── storm_event_manifest_v2.csv
│
└── explainability/
    ├── overlay_fani_20190502T060000+0000.png
    ├── heatmap_fani_20190502T060000+0000.png
    ├── original_fani_20190502T060000+0000.png
    └── explainability_manifest.json
```

---

## 2. Security & Access Control Model

### 2.1 Private Server-Side Model Access
- **Private Bucket Configuration:** The `vayu-net-runtime` bucket is created with `public: false`.
- **Zero Frontend Weight Access:** The client-side React application (`apps/frontend/`) **NEVER** communicates directly with the model storage bucket.
- **Service-Role Authentication:** Backend workers download checkpoints during the build step using either signed URLs or the service role key (`SUPABASE_SERVICE_ROLE_KEY`).
- **Public Assets Restriction:** Only post-processed, rendered interpretability PNG images in `explainability/` are publicly accessible via the `/static/explainability/` backend mount.

### 2.2 SHA-256 Checksum Enforcement
Every artifact stored in Supabase must match the exact cryptographic SHA-256 hash defined in `artifacts/manifest.json`.

| Artifact Name | Filename | Size (Bytes) | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `center_localization_cnn` | `best_center_localization_cnn.pt` | 51,406,686 | `5ef11b596ef94ece71079c1329eab1e50191127eb7b9b1ff13f7dcdd3185f1c9` |
| `temporal_track_gru` | `best_temporal_track_gru.pt` | 48,407,839 | `7973c12a1d1b15fcb4c201deb9491bdb51d4ae9022881aff5233d85606676b46` |
| `phase5b_variant_a` | `best_phase5b_variant_a.pt` | 1,754,009 | `c1a51fb9d2e55a6043c4b903a4b4696d4c98cb9034082e4825fb6682fefd9216` |
| `phase5b_variant_b` | `best_phase5b_variant_b.pt` | 1,754,009 | `2f0ec1e1e136c552a1ab4a4a8002e1af3293489d5fb649d95b30c2ca5e4a9533` |
| `phase6_intensity_wind` | `best_phase6_intensity_wind.pt` | 1,506,213 | `a014e06dbf6fc17f37d04bb545a58746e57be6cb309808735660989009a460ef` |
| `uncertainty_params` | `uncertainty_parameters.json` | 1,970 | `a08ba1d55edd75758c88d26e04578eec4395b59cc892f2ebcd44287d9930f3c1` |
| `analog_cache` | `analog_retrieval_cache.json` | 651,441 | `ec5ee152c1ba5871ad5ea24c9642146c5a436f2a329422dbcc34335e61cb6ac3` |

### 2.3 Immutable Versioning Strategy
- Checkpoints are named immutably with semver suffixing (e.g. `_v1.0.pt`).
- Modifying weights requires a new version tag (e.g. `_v1.1.pt`). Overwriting existing weights in place is prohibited to preserve full experiment reproducibility.

---

## 3. Provisioning Steps (For DevOps / Cloud Engineer)

When Supabase project credentials are ready:

1. **Create Bucket:**
   ```sql
   insert into storage.buckets (id, name, public)
   values ('vayu-net-runtime', 'vayu-net-runtime', false);
   ```

2. **Upload Artifacts:**
   Upload the local weights from `data/interim/ml/checkpoints/` into the `models/` directory in the bucket.

3. **Configure Environment Variables:**
   Set the following variables in Render dashboard:
   - `MODEL_STORAGE_BASE_URL`: Base URL (or Supabase signed URL endpoint)
   - `SUPABASE_URL`: Supabase project URL
   - `SUPABASE_SERVICE_ROLE_KEY`: Service role key for private artifact download

4. **Verify Deployment:**
   Trigger the Render build pipeline. `download_runtime_artifacts.py` will retrieve missing checkpoints, verify SHA-256 hashes, and start the FastAPI service.
