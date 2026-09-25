# VAYU-NET: NASA GPM IMERG Storage Inventory & Reconciliation Audit

**Date:** 2026-09-25  
**Auditor:** Antigravity research pipeline  
**Constraint:** Read-only audit. No deletion without explicit proof. No models trained. No commit.  

---

## 1. Context & Discrepancy Statement

During the bulk download validation of the NASA GPM IMERG Final Run V07B dataset, local file inventory identified:
- **Total physical HDF5 files on disk:** 2,356 files
- **Required canonical IMERG granules:** 2,353 granules
- **Discrepancy:** +3 surplus files

---

## 2. Root Cause Investigation

Cross-referencing `data/manifests/imerg_bulk_download_manifest.csv` and the earlier pilot download manifest `data/manifests/imerg_pilot_manifest.csv`:

The 3 surplus files originated during initial Phase 1 feasibility testing, where pilot granules for storms Fani (2019), Amphan (2020), and Remal (2024) were saved without the standard NASA prefix `3B-HHR.MS.MRG.3IMERG.`.

The 3 files in question are:
1. `20190429-S120000-E122959.0720.V07B.HDF5`
2. `20200517-S060000-E062959.0360.V07B.HDF5`
3. `20240525-S060000-E062959.0360.V07B.HDF5`

Their canonical NASA GES DISC counterparts are:
1. `3B-HHR.MS.MRG.3IMERG.20190429-S120000-E122959.0720.V07B.HDF5`
2. `3B-HHR.MS.MRG.3IMERG.20200517-S060000-E062959.0360.V07B.HDF5`
3. `3B-HHR.MS.MRG.3IMERG.20240525-S060000-E062959.0360.V07B.HDF5`

---

## 3. Cryptographic & Byte-Level Verification

All 3 pairs were evaluated for byte size and SHA-256 cryptographic checksums:

| Surplus Filename | Canonical Filename | Surplus SHA-256 | Canonical SHA-256 | Byte Size | Match |
|---|---|---|---|---|---|
| `20190429-S120000-E122959.0720.V07B.HDF5` | `3B-HHR.MS.MRG.3IMERG.20190429-S120000-E122959.0720.V07B.HDF5` | `49ebb825dd1b8790ed8332eda2144d97a9fb9e96768ba641a51fd23cd1b26d47` | `49ebb825dd1b8790ed8332eda2144d97a9fb9e96768ba641a51fd23cd1b26d47` | 8,213,594 | **EXACT BYTE MATCH** |
| `20200517-S060000-E062959.0360.V07B.HDF5` | `3B-HHR.MS.MRG.3IMERG.20200517-S060000-E062959.0360.V07B.HDF5` | `50fee96caab48362110593824e1b406955fc0f2ffe1e3236510a2326835beacf` | `50fee96caab48362110593824e1b406955fc0f2ffe1e3236510a2326835beacf` | 8,098,612 | **EXACT BYTE MATCH** |
| `20240525-S060000-E062959.0360.V07B.HDF5` | `3B-HHR.MS.MRG.3IMERG.20240525-S060000-E062959.0360.V07B.HDF5` | `a9a70cef7ace8922dc175e4656b70807d9311b4711f9b5809cd6a35a9604d0e6` | `a9a70cef7ace8922dc175e4656b70807d9311b4711f9b5809cd6a35a9604d0e6` | 8,068,764 | **EXACT BYTE MATCH** |

---

## 4. Policy & Resolution

1. **Storage Inventory Manifest:**
   A full catalog has been created at `data/manifests/imerg_storage_inventory.csv` (2,356 rows) explicitly flagging:
   - 2,353 `CANONICAL_PRIMARY` granules (`is_canonical=True`, `is_redundant_duplicate=False`)
   - 3 `REDUNDANT_PILOT_ALIAS` files (`is_canonical=False`, `is_redundant_duplicate=True`)

2. **Retention Policy:**
   Because legacy pilot test suites (`scripts/audit/validate_imerg.py`) and historical manifests (`data/manifests/imerg_pilot_manifest.csv`) specifically check these exact pilot filenames to certify initial phase reproducibility, the 3 redundant alias copies are retained on disk alongside the 2,353 canonical files.
   - Extra disk overhead: ~24.4 MB total (0.14% of the 17.3 GB archive).
   - Zero risk of dataset contamination: the multimodal dataset construction pipeline strictly filters against `imerg_storage_inventory.csv` where `is_canonical == True` and reads exclusively canonical filenames matching `data/manifests/gridsat_imerg_pairing_manifest.csv`.

3. **Status:**
   Fully resolved and audited.