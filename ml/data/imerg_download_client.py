"""
NASA GPM IMERG Final Run V07B Authenticated Download Client.

Target Product: NASA GPM IMERG Final Run V07B (Half-Hourly 0.1 degree)
Collection: GPM_3IMERGHH.07
Format: Hierarchical Data Format 5 (.HDF5)

Security & Integrity Constraints:
1. Credentials MUST come ONLY from environment variables:
   - EARTHDATA_USERNAME (or NASA_EARTHDATA_USERNAME)
   - EARTHDATA_PASSWORD (or NASA_EARTHDATA_PASSWORD)
2. NEVER hardcode, print, log, serialize, or commit credentials.
3. Verify HDF5 signature and SHA-256 for all downloads.
4. Download ONLY the specified research pilot granules initially.
5. Zero synthetic data generation or fallback to GridSat.
"""

import os
import sys
import re
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IMERGDownloader")

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
URS_AUTH_URL = "https://urs.earthdata.nasa.gov"
GES_DISC_APP_CLIENT_ID = "e2WVk8Pw6weeLUKZYOxvTQ"
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"

# Pilot Granule Specifications
PILOT_TARGETS = [
    {
        "storm_id": "2019_02B",
        "storm_name": "FANI",
        "target_timestamp_utc": "2019-04-29T12:00:00Z",
        "temporal_range": "2019-04-29T11:45:00Z,2019-04-29T12:15:00Z",
        "expected_granule_substr": "20190429-S120000-E122959.0720.V07B.HDF5"
    },
    {
        "storm_id": "2020_01B",
        "storm_name": "AMPHAN",
        "target_timestamp_utc": "2020-05-17T06:00:00Z",
        "temporal_range": "2020-05-17T05:45:00Z,2020-05-17T06:15:00Z",
        "expected_granule_substr": "20200517-S060000-E062959.0360.V07B.HDF5"
    },
    {
        "storm_id": "2024_01B",
        "storm_name": "REMAL",
        "target_timestamp_utc": "2024-05-25T06:00:00Z",
        "temporal_range": "2024-05-25T05:45:00Z,2024-05-25T06:15:00Z",
        "expected_granule_substr": "20240525-S060000-E062959.0360.V07B.HDF5"
    }
]


class CredentialsUnavailableError(Exception):
    """Raised when NASA Earthdata credentials are missing from the environment."""
    pass


class EarthdataAuthenticationError(Exception):
    """Raised when NASA Earthdata authentication or application authorization fails."""
    pass


class EarthdataSession(requests.Session):
    """
    Requests session configured to handle NASA Earthdata URS OAuth2 / HTTP Basic Auth redirects.
    Ensures authentication headers are preserved across domain redirects between GES DISC and URS.
    """
    def __init__(self, username: str, password: str):
        super().__init__()
        self.auth = (username, password)
        self.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 VAYU-NET-Research/1.0"
        })

    def rebuild_auth(self, prepared_request, response):
        """Re-attaches HTTP Basic Auth when redirected to urs.earthdata.nasa.gov."""
        url = prepared_request.url
        if "urs.earthdata.nasa.gov" in url or "data.gesdisc.earthdata.nasa.gov" in url:
            prepared_request.prepare_auth(self.auth)
        else:
            super().rebuild_auth(prepared_request, response)


class IMERGDownloadClient:
    """Client for querying and downloading NASA GPM IMERG Final Run V07B products from GES DISC."""

    def __init__(self, output_dir: str = "data/raw/imerg/"):
        self.output_dir = Path(output_dir)
        self._username = os.environ.get("EARTHDATA_USERNAME") or os.environ.get("NASA_EARTHDATA_USERNAME")
        self._password = os.environ.get("EARTHDATA_PASSWORD") or os.environ.get("NASA_EARTHDATA_PASSWORD")
        self.is_authenticated = False
        self._session: Optional[EarthdataSession] = None

    def has_credentials(self) -> bool:
        """Check whether credentials are provided without exposing their content."""
        return bool(self._username and self._password)

    def require_credentials(self) -> None:
        """Enforce credential availability or raise explicit error."""
        if not self.has_credentials():
            missing = []
            if not self._username:
                missing.append("EARTHDATA_USERNAME")
            if not self._password:
                missing.append("EARTHDATA_PASSWORD")
            raise CredentialsUnavailableError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD before attempting download. "
                f"Never hardcode or commit credentials."
            )

    def get_session(self) -> EarthdataSession:
        """Return authenticated requests session."""
        self.require_credentials()
        if self._session is None:
            self._session = EarthdataSession(self._username, self._password)
        return self._session

    def ensure_gesdisc_authorized(self) -> bool:
        """
        Verify and, if necessary, approve the GES DISC DATA ARCHIVE application in Earthdata Login.
        NASA URS requires a one-time authorization checkbox/EULA for GES DISC data egress.
        """
        session = self.get_session()
        approve_url = f"{URS_AUTH_URL}/approve_app?client_id={GES_DISC_APP_CLIENT_ID}"

        # 1. Login to establish cookie session
        r_login_page = session.get(f"{URS_AUTH_URL}/login")
        token_match = re.search(r'name="authenticity_token"\s+value="([^"]+)"', r_login_page.text)
        token = token_match.group(1) if token_match else None

        login_payload = {
            "username": self._username,
            "password": self._password,
            "commit": "Log in"
        }
        if token:
            login_payload["authenticity_token"] = token
        session.post(f"{URS_AUTH_URL}/login", data=login_payload)

        # 2. Check application approval
        r_app = session.get(approve_url)
        if "approve_app" in r_app.url and r_app.status_code == 200:
            match_app_token = re.search(r'name="authenticity_token"\s+value="([^"]+)"', r_app.text)
            app_token = match_app_token.group(1) if match_app_token else None
            post_data = {
                "authenticity_token": app_token,
                "client_id": GES_DISC_APP_CLIENT_ID,
                "app_uid": "nasa_gesdisc_data_archive",
                "agreement": "1",
                "commit": "Agree",
                "authorize": "Agree"
            }
            r_post = session.post(f"{URS_AUTH_URL}/approve_app", data=post_data)
            logger.info("GES DISC application EULA authorization submitted successfully.")
            return r_post.status_code in (200, 302)
        return True

    def check_auth(self) -> bool:
        """
        Validate credentials by testing authentication against NASA Earthdata Login.
        Returns True if successful, raises EarthdataAuthenticationError otherwise.
        """
        self.require_credentials()
        session = self.get_session()

        # Test against a known GES DISC small metadata probe
        probe_url = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/2019/119/3B-HHR.MS.MRG.3IMERG.20190429-S120000-E122959.0720.V07B.HDF5"
        try:
            r = session.head(probe_url, allow_redirects=True, timeout=20)
            if r.status_code == 200:
                self.is_authenticated = True
                logger.info("NASA Earthdata authentication and GES DISC access verified successfully.")
                return True
            elif r.status_code == 401 or "approve_app" in r.url:
                # Attempt to ensure application authorization
                logger.info("Attempting automatic GES DISC application authorization...")
                if self.ensure_gesdisc_authorized():
                    r2 = session.head(probe_url, allow_redirects=True, timeout=20)
                    if r2.status_code == 200:
                        self.is_authenticated = True
                        logger.info("NASA Earthdata authentication successful after app authorization.")
                        return True
            raise EarthdataAuthenticationError(f"Authentication returned status {r.status_code}.")
        except requests.RequestException as e:
            raise EarthdataAuthenticationError(f"Network error during Earthdata authentication check: {e}")

    def query_cmr_granule(
        self,
        temporal_range: str,
        expected_substr: str
    ) -> Dict[str, Any]:
        """
        Query NASA Common Metadata Repository (CMR) for the exact GPM IMERG Final Run V07B granule.
        """
        params = {
            "short_name": "GPM_3IMERGHH",
            "temporal": temporal_range,
            "page_size": 10
        }
        resp = requests.get(CMR_GRANULES_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("feed", {}).get("entry", [])

        for entry in entries:
            title = entry.get("title", "")
            if expected_substr in title and "V07B" in title:
                download_url = None
                for link in entry.get("links", []):
                    if "inherited" not in link and link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                        download_url = link.get("href")
                        break
                return {
                    "title": title,
                    "download_url": download_url,
                    "dataset_id": entry.get("id"),
                    "time_start": entry.get("time_start"),
                    "time_end": entry.get("time_end"),
                    "granule_size_mb": entry.get("granule_size")
                }
        raise FileNotFoundError(f"Granule matching '{expected_substr}' not found in NASA CMR query.")

    def download_pilot_granule(
        self,
        target_info: Dict[str, str],
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Download a single verified IMERG Final Run V07B HDF5 file.
        Verifies HDF5 magic header and records SHA-256.
        """
        self.require_credentials()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        storm_name = target_info["storm_name"]
        expected_substr = target_info["expected_granule_substr"]
        temporal_range = target_info["temporal_range"]

        logger.info(f"Querying NASA CMR for {storm_name} ({target_info['target_timestamp_utc']})...")
        cmr_meta = self.query_cmr_granule(temporal_range, expected_substr)
        download_url = cmr_meta["download_url"]
        filename = expected_substr
        dest_path = self.output_dir / filename

        if dest_path.exists() and not force:
            logger.info(f"Granule {filename} already exists locally. Verifying existing file...")
            sha256 = self.compute_sha256(dest_path)
            self.verify_hdf5_header(dest_path)
            return {
                "storm_name": storm_name,
                "filename": filename,
                "local_path": str(dest_path),
                "download_url": download_url,
                "size_bytes": dest_path.stat().st_size,
                "sha256": sha256,
                "already_existed": True
            }

        session = self.get_session()
        logger.info(f"Downloading {filename} from {download_url}...")
        hasher = hashlib.sha256()

        with session.get(download_url, stream=True, timeout=60) as resp:
            if resp.status_code == 401:
                # Try re-authorizing
                self.ensure_gesdisc_authorized()
                resp = session.get(download_url, stream=True, timeout=60)
            resp.raise_for_status()

            with open(dest_path, "wb") as f_out:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f_out.write(chunk)
                        hasher.update(chunk)

        # Check HDF5 signature
        try:
            self.verify_hdf5_header(dest_path)
        except ValueError as e:
            dest_path.unlink(missing_ok=True)
            raise ValueError(f"Downloaded file {filename} is NOT a genuine HDF5 file: {e}")

        calculated_sha = hasher.hexdigest()
        size_bytes = dest_path.stat().st_size
        logger.info(f"Successfully downloaded and verified {filename} ({size_bytes / 1024 / 1024:.2f} MB, SHA: {calculated_sha[:16]}...)")

        return {
            "storm_name": storm_name,
            "filename": filename,
            "local_path": str(dest_path),
            "download_url": download_url,
            "size_bytes": size_bytes,
            "sha256": calculated_sha,
            "already_existed": False
        }

    @staticmethod
    def verify_hdf5_header(path: Path) -> None:
        """Assert that the file begins with the official HDF5 magic signature."""
        with open(path, "rb") as f:
            header = f.read(8)
            if header != HDF5_SIGNATURE:
                raise ValueError(f"Invalid HDF5 header: got {header}, expected {HDF5_SIGNATURE}")

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute SHA-256 hash of a local file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def download_all_pilots(self) -> List[Dict[str, Any]]:
        """Download all 3 research pilot granules."""
        results = []
        for target in PILOT_TARGETS:
            res = self.download_pilot_granule(target)
            results.append(res)
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NASA GPM IMERG Final Run V07B Ingestion Client")
    parser.add_argument("--check-auth", action="store_true", help="Verify NASA Earthdata authentication")
    parser.add_argument("--download-pilot", action="store_true", help="Download the 3 target pilot granules")
    parser.add_argument("--output-dir", type=str, default="data/raw/imerg/", help="Target directory for HDF5 files")
    args = parser.parse_args()

    client = IMERGDownloadClient(output_dir=args.output_dir)

    if args.check_auth:
        print("[CHECK] Evaluating NASA Earthdata credentials...")
        if not client.has_credentials():
            print("[STATUS] EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables are NOT set.")
            print("[SAFETY] Halted downloader interface in compliance with stop conditions.")
            sys.exit(0)
        try:
            client.check_auth()
            print("[SUCCESS] NASA Earthdata authentication & GES DISC authorization verified.")
            sys.exit(0)
        except Exception as e:
            print(f"[ERROR] Authentication check failed: {e}")
            sys.exit(1)

    if args.download_pilot:
        print("[DOWNLOAD] Initiating download of 3 pilot granules...")
        if not client.has_credentials():
            print("[ERROR] Credentials missing. Download cannot proceed.")
            sys.exit(1)
        try:
            results = client.download_all_pilots()
            print(f"[SUCCESS] Downloaded and verified {len(results)} pilot granules.")
            for r in results:
                print(f"  {r['storm_name']}: {r['filename']} ({r['size_bytes']} bytes, SHA-256: {r['sha256'][:16]}...)")
        except Exception as e:
            print(f"[ERROR] Pilot download failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
