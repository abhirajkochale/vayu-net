"""
MOSDAC Authenticated Ingestion Client for Native INSAT-3D Observations.

Target Product: INSAT-3D Imager Level-1C Asian Sector (3DIMG_L1C_ASIA_MER)
Data Format: Hierarchical Data Format 5 (.h5)

Security & Provenance Rules:
1. Credentials MUST come exclusively from MOSDAC_USERNAME and MOSDAC_PASSWORD environment variables.
2. NEVER hardcode, print, log, commit, or serialize credentials.
3. If credentials are unavailable, stop immediately without fabricating files.
4. Zero synthetic data generation.
"""

import os
import sys
import hashlib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.parse
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MOSDACClient")

MOSDAC_BASE_URL = "https://mosdac.gov.in"
MOSDAC_API_URL = "https://mosdac.gov.in/apios"
TARGET_PRODUCT_ID = "3DIMG_L1C_ASIA_MER"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


class CredentialsUnavailableError(Exception):
    """Raised when MOSDAC_USERNAME or MOSDAC_PASSWORD environment variables are missing."""
    pass


class MOSDACAuthenticationError(Exception):
    """Raised when MOSDAC authentication fails."""
    pass


class MOSDACDownloadClient:
    """Authenticated client for querying and downloading native INSAT-3D HDF5 files from ISRO/MOSDAC."""

    def __init__(self, output_dir: str = "data/raw/insat3d/"):
        self.output_dir = Path(output_dir)
        self._username: Optional[str] = os.environ.get("MOSDAC_USERNAME")
        self._password: Optional[str] = os.environ.get("MOSDAC_PASSWORD")
        self.is_authenticated: bool = False
        self.session_token: Optional[str] = None

    def has_credentials(self) -> bool:
        """Verify presence of credentials in environment without exposing values."""
        return bool(self._username and self._password)

    def require_credentials(self) -> None:
        """Enforce credential availability or raise explicit error."""
        if not self.has_credentials():
            missing = []
            if not self._username:
                missing.append("MOSDAC_USERNAME")
            if not self._password:
                missing.append("MOSDAC_PASSWORD")
            raise CredentialsUnavailableError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Set these environment variables before attempting native MOSDAC downloads. "
                f"DO NOT hardcode credentials into files or commits."
            )

    def authenticate(self) -> bool:
        """
        Authenticate against MOSDAC API using environment credentials.
        Returns True if successful, raises MOSDACAuthenticationError otherwise.
        """
        self.require_credentials()

        auth_url = f"{MOSDAC_API_URL}/authenticate"
        payload = json.dumps({
            "username": self._username,
            "password": self._password
        }).encode("utf-8")

        req = urllib.request.Request(
            auth_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "VAYU-NET-Research-Pipeline/1.0"}
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.session_token = data.get("token") or data.get("session_id")
                    self.is_authenticated = True
                    logger.info("MOSDAC authentication successful. Session token established.")
                    return True
                else:
                    raise MOSDACAuthenticationError(f"MOSDAC authentication returned HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code == 401 or e.code == 403:
                raise MOSDACAuthenticationError("MOSDAC authentication rejected: Invalid username or password.")
            elif e.code == 500 or e.code == 503:
                logger.warning(f"MOSDAC server endpoint {auth_url} returned HTTP {e.code} (Service Unavailable / Maintenance).")
                raise MOSDACAuthenticationError(f"MOSDAC API server returned HTTP {e.code}. Service may be under maintenance.")
            raise MOSDACAuthenticationError(f"HTTP error during MOSDAC authentication: {e.code}")
        except urllib.error.URLError as e:
            logger.error(f"Network error contacting MOSDAC: {e.reason}")
            raise MOSDACAuthenticationError(f"Network connection to MOSDAC failed: {e.reason}")

    def query_catalog(
        self,
        product_id: str = TARGET_PRODUCT_ID,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Query MOSDAC catalog for product granules.
        Requires valid authentication.
        """
        self.require_credentials()
        if not self.is_authenticated:
            self.authenticate()

        query_params = {
            "productId": product_id,
            "format": "HDF5",
            "limit": str(limit)
        }
        if start_date:
            query_params["startDate"] = start_date
        if end_date:
            query_params["endDate"] = end_date

        url = f"{MOSDAC_API_URL}/granules?{urllib.parse.urlencode(query_params)}"
        headers = {
            "Authorization": f"Bearer {self.session_token}" if self.session_token else "",
            "User-Agent": "VAYU-NET-Research-Pipeline/1.0"
        }

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("granules", [])
        except urllib.error.HTTPError as e:
            logger.error(f"Catalog query failed: HTTP {e.code}")
            raise

    def download_granule(
        self,
        download_url: str,
        target_filename: str,
        expected_sha256: Optional[str] = None
    ) -> Path:
        """
        Download a single native HDF5 granule and verify SHA-256 and HDF5 magic header.
        """
        self.require_credentials()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self.output_dir / target_filename

        headers = {
            "Authorization": f"Bearer {self.session_token}" if self.session_token else "",
            "User-Agent": "VAYU-NET-Research-Pipeline/1.0"
        }
        req = urllib.request.Request(download_url, headers=headers)

        logger.info(f"Downloading granule to {dest_path}...")
        hasher = hashlib.sha256()
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    hasher.update(chunk)

        calculated_hash = hasher.hexdigest()
        if expected_sha256 and calculated_hash.lower() != expected_sha256.lower():
            dest_path.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {dest_path.name}: expected {expected_sha256}, got {calculated_hash}")

        # Validate HDF5 signature
        with open(dest_path, "rb") as f_check:
            header = f_check.read(8)
            if header != HDF5_SIGNATURE:
                dest_path.unlink(missing_ok=True)
                raise ValueError(f"Downloaded file {dest_path.name} is NOT a valid HDF5 file (invalid magic signature).")

        logger.info(f"Successfully downloaded and verified {dest_path.name} (SHA-256: {calculated_hash[:16]}...)")
        return dest_path

    @staticmethod
    def inspect_local_file(file_path: Path) -> Dict[str, Any]:
        """Verify local file format and calculate SHA-256 without loading data."""
        if not file_path.exists():
            return {"exists": False, "is_hdf5": False}

        size = file_path.stat().st_size
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            header = f.read(8)
            f.seek(0)
            while chunk := f.read(65536):
                hasher.update(chunk)

        return {
            "exists": True,
            "path": str(file_path),
            "size_bytes": size,
            "sha256": hasher.hexdigest(),
            "is_hdf5": header == HDF5_SIGNATURE
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MOSDAC Ingestion Client for INSAT-3D")
    parser.add_argument("--check-auth", action="store_true", help="Check if credentials exist and test auth")
    parser.add_argument("--inspect-file", type=str, help="Inspect a local file for HDF5 signature and SHA-256")
    args = parser.parse_args()

    client = MOSDACDownloadClient()

    if args.check_auth:
        print("[CHECK] Evaluating MOSDAC environment credentials...")
        if not client.has_credentials():
            print("[STATUS] MOSDAC_USERNAME / MOSDAC_PASSWORD environment variables are NOT set.")
            print("[SAFETY] Halted downloader interface in compliance with Part 2 / Part 10 protocol.")
            sys.exit(0)
        else:
            print("[STATUS] MOSDAC credentials found in environment. Attempting test authentication...")
            try:
                client.authenticate()
                print("[SUCCESS] MOSDAC authentication succeeded.")
            except Exception as e:
                print(f"[ERROR] MOSDAC authentication failed: {e}")
                sys.exit(1)

    if args.inspect_file:
        info = client.inspect_local_file(Path(args.inspect_file))
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
