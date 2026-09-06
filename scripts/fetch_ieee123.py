"""Script to fetch and verify the IEEE 123-node OpenDSS test feeder files."""

import hashlib
import os
import sys
import urllib.request
from typing import Dict

OFFICIAL_SOURCE_URL = "https://raw.githubusercontent.com/tshort/OpenDSS/master/Distrib/IEEETestCases/123Bus/"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "external", "ieee123")

# Known SHA-256 checksums for verified IEEE 123 feeder files
EXPECTED_CHECKSUMS: Dict[str, str] = {
    "IEEE123Master.dss": "21bfab1a3c956d347dc491436ae839281bf7db4ececc405e10ddeb04d3ab00ec",
    "IEEELineCodes.DSS": "4bc8e3a0b24ad7b1cb4321f1c3b6c8458b646dd3a83b3cd11cccc02444624cb2",
    "IEEE123Loads.DSS": "fd28f52836205506a1980b52c5a7e12ca585aa626c52a6249502c1c3b2fc2255",
    "IEEE123Regulators.DSS": "cd6ad77980dc018631ce39faab528afce40b0b9a50a6bc1c9ac8049ffb8207d7",
    "Run_IEEE123Bus.DSS": "559ccd5d3173f9204a7a5a1a764c3f405a0f46941f5e4fd7713f1ebf7ebbc0ef",
}


def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hex digest of file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def fetch_ieee123(target_dir: str = TARGET_DIR, verify_checksums: bool = True) -> bool:
    """Download IEEE 123 OpenDSS test feeder files and verify checksums.

    Args:
        target_dir: Directory to save downloaded files.
        verify_checksums: Whether to enforce SHA-256 integrity checks.

    Returns:
        True if all files are downloaded and verified successfully.
    """
    os.makedirs(target_dir, exist_ok=True)
    print(f"Fetching IEEE 123-bus test feeder from: {OFFICIAL_SOURCE_URL}")
    print(f"Destination directory: {target_dir}")

    all_ok = True
    for filename, expected_hash in EXPECTED_CHECKSUMS.items():
        file_path = os.path.join(target_dir, filename)
        url = OFFICIAL_SOURCE_URL + filename

        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "GridGuard-Research/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read()
                with open(file_path, "wb") as fp:
                    fp.write(content)
            except Exception as e:
                print(f"Error downloading {filename} from {url}: {e}")
                all_ok = False
                continue

        actual_hash = compute_sha256(file_path)
        if verify_checksums:
            if actual_hash.lower() == expected_hash.lower():
                print(f"  [PASS] {filename} (SHA-256 verified)")
            else:
                print(f"  [FAIL] {filename} checksum mismatch!")
                print(f"         Expected: {expected_hash}")
                print(f"         Actual:   {actual_hash}")
                all_ok = False
        else:
            print(f"  [OK] {filename} exists (hash: {actual_hash[:12]}...)")

    return all_ok


if __name__ == "__main__":
    success = fetch_ieee123()
    if not success:
        print("Failed to acquire all IEEE 123-bus feeder files with verified checksums.")
        sys.exit(1)
    print("IEEE 123-bus test feeder acquired and verified successfully.")
