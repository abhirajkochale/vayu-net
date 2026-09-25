#!/usr/bin/env python3
"""VAYU-NET — Runtime Artifacts Verification & Download Utility
============================================================
Verifies and downloads required model weights, feature cache,
and GridSat satellite observation frames according to artifacts/manifest.json.

Supports both flat GitHub Release Asset endpoints and static HTTPS storage.

Usage:
  python scripts/setup/download_runtime_artifacts.py --check
  python scripts/setup/download_runtime_artifacts.py --dry-run
  python scripts/setup/download_runtime_artifacts.py
  python scripts/setup/download_runtime_artifacts.py --release-tag v1.1.0
  python scripts/setup/download_runtime_artifacts.py --base-url https://github.com/abhirajkochale/vayu-net/releases/download/v1.1.0
  python scripts/setup/download_runtime_artifacts.py --base-url http://127.0.0.1:8765 --target-dir /path/to/target
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "artifacts/manifest.json"
DEFAULT_RELEASE_REPO = "abhirajkochale/vayu-net"
USER_AGENT = "VAYU-NET-Downloader/1.1 (Operational Cyclone Intelligence)"


def compute_sha256(file_path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def check_artifacts(manifest: Dict[str, Any], target_root: Path = PROJECT_ROOT) -> Tuple[int, int, List[str]]:
    """Checks the presence and SHA-256 integrity of all manifest artifacts."""
    print("=" * 68)
    print("VAYU-NET RUNTIME ARTIFACT INTEGRITY CHECK")
    print(f"Target Root: {target_root}")
    print("=" * 68)

    passed = 0
    failed = 0
    missing_files: List[str] = []

    def _check_group(title: str, items: List[Dict[str, Any]]) -> None:
        nonlocal passed, failed
        print(f"\n{title}")
        for item in items:
            name = item["name"]
            rel_path = item.get("destination_path") or item["path"]
            expected_sha = item["sha256"]
            full_path = target_root / rel_path

            if not full_path.exists():
                print(f"  [MISSING] {name:35s} -> {rel_path}")
                failed += 1
                missing_files.append(rel_path)
                continue

            actual_sha = compute_sha256(full_path)
            if actual_sha.lower() == expected_sha.lower():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                print(f"  [OK]      {name:35s} ({size_mb:6.2f} MB)")
                passed += 1
            else:
                print(f"  [MISMATCH]{name:35s}")
                print(f"     Expected: {expected_sha}")
                print(f"     Actual:   {actual_sha}")
                failed += 1

    # 1. Models
    _check_group("[1] Checking Runtime Model Checkpoints:", manifest.get("models", []))

    # 2. Feature Cache
    _check_group("[2] Checking Runtime Feature Store Cache:", manifest.get("feature_cache", []))

    # 3. GridSat Observation Frames
    _check_group("[3] Checking GridSat Satellite Observations:", manifest.get("gridsat_frames", []))

    # 4. Metadata & Calibration Assets
    _check_group("[4] Checking Runtime Metadata & Calibration Assets:", manifest.get("runtime_assets", []))

    # 5. Curated Demo Cyclone Assets
    print("\n[5] Checking Curated Demo Cyclone Assets:")
    demo_dir = target_root / "data/interim/ml/explainability"
    for d in manifest.get("demo_assets", []):
        cyclone = d["cyclone"]
        matches = list(demo_dir.glob(f"*{cyclone.lower()}*.png")) if demo_dir.exists() else []
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


def resolve_asset_url(
    item: Dict[str, Any],
    base_storage_url: str | None,
    hierarchical: bool = False,
) -> str | None:
    """
    Resolves the remote download URL for a manifest item.
    Defaults to flat GitHub Release Asset layout:
      <base_url>/<asset_filename>
    If hierarchical=True and storage_rel_url is set, uses:
      <base_url>/<storage_rel_url>
    """
    if not base_storage_url:
        return item.get("source_url")

    asset_filename = item.get("asset_filename") or Path(item.get("destination_path") or item["path"]).name

    if hierarchical and item.get("storage_rel_url"):
        return f"{base_storage_url.rstrip('/')}/{item['storage_rel_url'].lstrip('/')}"

    # Default: Flat URL (GitHub Release Asset standard)
    return f"{base_storage_url.rstrip('/')}/{asset_filename}"


def download_artifacts(
    manifest: Dict[str, Any],
    dry_run: bool = False,
    base_storage_url: str | None = None,
    target_root: Path = PROJECT_ROOT,
    hierarchical: bool = False,
) -> bool:
    """Downloads missing artifacts from configured external sources."""
    print("=" * 68)
    print(f"VAYU-NET ARTIFACT DOWNLOAD {'(DRY RUN)' if dry_run else ''}")
    print(f"Target Root: {target_root}")
    print(f"Storage URL: {base_storage_url or 'NOT_CONFIGURED'}")
    print(f"URL Mode:    {'HIERARCHICAL' if hierarchical else 'FLAT (GitHub Release Standard)'}")
    print("=" * 68)

    base_storage_url = base_storage_url or os.getenv("MODEL_STORAGE_BASE_URL")

    # Collect all items requiring verification or download
    all_targets: List[Dict[str, Any]] = []
    all_targets.extend(manifest.get("models", []))
    all_targets.extend(manifest.get("feature_cache", []))
    all_targets.extend(manifest.get("gridsat_frames", []))

    items_to_fetch = []
    for item in all_targets:
        rel_path = item.get("destination_path") or item["path"]
        full_path = target_root / rel_path
        expected_sha = item["sha256"]

        if full_path.exists():
            actual_sha = compute_sha256(full_path)
            if actual_sha.lower() == expected_sha.lower():
                print(f"[VERIFIED] {item['name']:35s} -> {rel_path}")
                continue

        items_to_fetch.append(item)

    if not items_to_fetch:
        print("\nAll required runtime artifacts are already present and verified. Nothing to download.")
        return True

    print(f"\n{len(items_to_fetch)} artifact(s) require download:")
    for item in items_to_fetch:
        size_mb = item.get("size_bytes", 0) / (1024 * 1024)
        print(f"  - {item['name']:35s}: {size_mb:6.2f} MB")

    if dry_run:
        print("\nDry run complete. No network requests initiated.")
        return True

    download_failed = False

    for item in items_to_fetch:
        name = item["name"]
        rel_path = item.get("destination_path") or item["path"]
        dest = target_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        url = resolve_asset_url(item, base_storage_url=base_storage_url, hierarchical=hierarchical)
        status = item.get("source_status", "UNKNOWN")

        if not url or "placeholder" in str(url) or (status == "NOT_PROVISIONED" and not base_storage_url):
            print(f"\n[FAIL] Cannot retrieve '{name}': Remote storage URL is not configured.")
            print(f"       Set MODEL_STORAGE_BASE_URL to a valid GitHub Releases or HTTPS endpoint.")
            download_failed = True
            continue

        print(f"\nDownloading {name} from {url} ...")
        temp_dest = dest.with_suffix(".tmp")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            hasher = hashlib.sha256()

            with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as out_f:
                if resp.status not in (200, 206):
                    raise urllib.error.HTTPError(url, resp.status, f"HTTP Error {resp.status}", resp.headers, None)
                while chunk := resp.read(1024 * 1024):
                    out_f.write(chunk)
                    hasher.update(chunk)

            actual_sha = hasher.hexdigest()
            expected_sha = item["sha256"]
            if actual_sha.lower() != expected_sha.lower():
                temp_dest.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {name}! Expected {expected_sha}, got {actual_sha}")

            temp_dest.replace(dest)
            print(f"[OK]   Successfully downloaded and verified: {name} -> {rel_path}")
        except Exception as e:
            temp_dest.unlink(missing_ok=True)
            print(f"[FAIL] Failed to download {name} from {url}: {e}")
            download_failed = True

    return not download_failed


def main():
    parser = argparse.ArgumentParser(description="VAYU-NET Runtime Artifact Management Utility")
    parser.add_argument("--check", action="store_true", help="Verify integrity of local runtime artifacts only (no downloads)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate download without performing requests")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL of static HTTPS/HTTP or GitHub Releases endpoint")
    parser.add_argument("--release-tag", type=str, default=None, help="GitHub release tag (e.g. v1.1.0) to construct download URL")
    parser.add_argument("--target-dir", type=str, default=None, help="Target root directory to verify/download into")
    parser.add_argument("--hierarchical", action="store_true", help="Use nested directory layout instead of flat release assets")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    target_root = Path(args.target_dir).resolve() if args.target_dir else PROJECT_ROOT

    if args.check:
        passed, failed, missing = check_artifacts(manifest, target_root=target_root)
        sys.exit(0 if failed == 0 else 1)

    # Determine base URL:
    # 1. Explicit --base-url
    # 2. Constructed from --release-tag / RELEASE_TAG
    # 3. Environment variable MODEL_STORAGE_BASE_URL
    base_url = args.base_url
    if not base_url:
        tag = args.release_tag or os.getenv("RELEASE_TAG")
        if tag:
            base_url = f"https://github.com/{DEFAULT_RELEASE_REPO}/releases/download/{tag}"
        else:
            base_url = os.getenv("MODEL_STORAGE_BASE_URL")

    success = download_artifacts(
        manifest,
        dry_run=args.dry_run,
        base_storage_url=base_url,
        target_root=target_root,
        hierarchical=args.hierarchical,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
