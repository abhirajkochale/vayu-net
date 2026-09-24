#!/usr/bin/env python3
"""VAYU-NET — Runtime Artifacts Verification & Download Utility
============================================================
Verifies and downloads required model weights and runtime assets
according to the specifications in artifacts/manifest.json.

Usage:
  python scripts/setup/download_runtime_artifacts.py --check
  python scripts/setup/download_runtime_artifacts.py --dry-run
  python scripts/setup/download_runtime_artifacts.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "artifacts/manifest.json"


def compute_sha256(file_path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_artifacts(manifest: Dict[str, Any]) -> Tuple[int, int, List[str]]:
    """Checks the presence and SHA-256 integrity of all manifest artifacts."""
    print("=" * 68)
    print("VAYU-NET RUNTIME ARTIFACT INTEGRITY CHECK")
    print("=" * 68)

    passed = 0
    failed = 0
    missing_files: List[str] = []

    # Check models
    print("\n[1] Checking Runtime Model Checkpoints:")
    for m in manifest.get("models", []):
        name = m["name"]
        rel_path = m["path"]
        expected_sha = m["sha256"]
        full_path = PROJECT_ROOT / rel_path

        if not full_path.exists():
            print(f"  [MISSING] {name:30s} -> {rel_path}")
            failed += 1
            missing_files.append(rel_path)
            continue

        actual_sha = compute_sha256(full_path)
        if actual_sha.lower() == expected_sha.lower():
            print(f"  [OK]      {name:30s} ({full_path.stat().st_size / (1024*1024):.1f} MB)")
            passed += 1
        else:
            print(f"  [MISMATCH]{name:30s}")
            print(f"     Expected: {expected_sha}")
            print(f"     Actual:   {actual_sha}")
            failed += 1

    # Check runtime assets
    print("\n[2] Checking Runtime Metadata & Calibration Assets:")
    for a in manifest.get("runtime_assets", []):
        name = a["name"]
        rel_path = a["path"]
        expected_sha = a["sha256"]
        full_path = PROJECT_ROOT / rel_path

        if not full_path.exists():
            print(f"  [MISSING] {name:30s} -> {rel_path}")
            failed += 1
            missing_files.append(rel_path)
            continue

        actual_sha = compute_sha256(full_path)
        if actual_sha.lower() == expected_sha.lower():
            print(f"  [OK]      {name:30s}")
            passed += 1
        else:
            print(f"  [MISMATCH]{name:30s}")
            failed += 1

    # Check demo assets
    print("\n[3] Checking Curated Demo Cyclone Assets:")
    demo_dir = PROJECT_ROOT / "data/interim/ml/explainability"
    for d in manifest.get("demo_assets", []):
        cyclone = d["cyclone"]
        matches = list(demo_dir.glob(f"*{cyclone.lower()}*.png"))
        if matches:
            print(f"  [OK]      Cyclone {cyclone:10s} ({len(matches)} visualization assets)")
            passed += 1
        else:
            print(f"  [MISSING] Cyclone {cyclone:10s} demo assets in {demo_dir}")
            failed += 1

    print("\n" + "=" * 68)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED / MISSING")
    print("=" * 68)
    return passed, failed, missing_files


def download_artifacts(manifest: Dict[str, Any], dry_run: bool = False) -> bool:
    """Downloads missing artifacts from configured external sources."""
    print("=" * 68)
    print(f"VAYU-NET ARTIFACT DOWNLOAD {'(DRY RUN)' if dry_run else ''}")
    print("=" * 68)

    models_to_fetch = []
    for m in manifest.get("models", []):
        rel_path = m["path"]
        full_path = PROJECT_ROOT / rel_path
        expected_sha = m["sha256"]

        if full_path.exists():
            actual_sha = compute_sha256(full_path)
            if actual_sha.lower() == expected_sha.lower():
                print(f"Artifact already present and verified: {m['name']} -> {rel_path}")
                continue

        models_to_fetch.append(m)

    if not models_to_fetch:
        print("\nAll runtime models are already present and verified. Nothing to download.")
        return True

    print(f"\n{len(models_to_fetch)} model(s) require download:")
    for m in models_to_fetch:
        print(f"  - {m['name']}: {m['size_bytes'] / (1024*1024):.1f} MB from {m['source_url']}")

    if dry_run:
        print("\nDry run complete. No network requests initiated.")
        return True

    base_storage_url = os.getenv("MODEL_STORAGE_BASE_URL")
    download_failed = False

    for m in models_to_fetch:
        name = m["name"]
        url = m.get("source_url")
        status = m.get("source_status", "UNKNOWN")

        # Resolve remote URL if base storage is configured
        if base_storage_url and (not url or "placeholder" in str(url) or status == "NOT_PROVISIONED"):
            filename = Path(m["path"]).name
            url = f"{base_storage_url.rstrip('/')}/models/{filename}"

        dest = PROJECT_ROOT / m["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not url or "placeholder" in str(url) or status == "NOT_PROVISIONED":
            print(f"\n[MANUAL STEP PENDING] Cannot retrieve '{name}': Remote cloud storage is not yet configured (status: {status}).")
            print(f"       Cloud deployment has not yet been performed. Manual deployment is pending.")
            print(f"       Configure MODEL_STORAGE_BASE_URL or upload weights to Supabase Storage bucket 'vayu-net-runtime'.")
            download_failed = True
            continue

        print(f"\nDownloading {name} from {url} ...")
        temp_dest = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, temp_dest)
            actual_sha = compute_sha256(temp_dest)
            if actual_sha.lower() != m["sha256"].lower():
                temp_dest.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {name}! Expected {m['sha256']}, got {actual_sha}")
            temp_dest.rename(dest)
            print(f"[OK]   Successfully downloaded and verified: {name}")
        except Exception as e:
            temp_dest.unlink(missing_ok=True)
            print(f"[FAIL] Failed to download {name}: {e}")
            download_failed = True

    return not download_failed


def main():
    parser = argparse.ArgumentParser(description="VAYU-NET Runtime Artifact Management Utility")
    parser.add_argument("--check", action="store_true", help="Verify integrity of local runtime artifacts only (no downloads)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate download without performing requests")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if args.check:
        passed, failed, missing = check_artifacts(manifest)
        sys.exit(0 if failed == 0 else 1)

    success = download_artifacts(manifest, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
