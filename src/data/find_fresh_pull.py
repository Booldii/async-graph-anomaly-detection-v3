"""
Shared helper for locating the latest fresh BigQuery pull (data/raw/eth_fresh_<start>_<end>.parquet,
produced by bq_fetch_day.py). Created so neither script needs its date hardcoded or updated by hand when a newer pull lands.
"""

import re
from pathlib import Path

FRESH_FILENAME_RE = re.compile(r"eth_fresh_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")


def find_latest_fresh_path(raw_dir: Path) -> Path:
    """Picks the eth_fresh_<start>_<end>.parquet file with the latest end date."""
    candidates = [
        (match.group(2), path)
        for path in raw_dir.glob("eth_fresh_*.parquet")
        if (match := FRESH_FILENAME_RE.search(path.name))
    ]
    if not candidates:
        raise FileNotFoundError(f"no eth_fresh_<start>_<end>.parquet files found in {raw_dir}")
    return max(candidates, key=lambda c: c[0])[1]
