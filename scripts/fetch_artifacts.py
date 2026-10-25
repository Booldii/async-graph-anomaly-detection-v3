"""
One-time bootstrap step for anyone who clones this repo fresh: downloads the large, local-only
artifacts the demo stack needs (the TGN checkpoint, the historical graph, and the replay sample)
from Hugging Face Hub into their expected paths under data/.

Idempotent: skips any file that's already present locally - safe to re-run any time.
"""

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_ID = os.environ.get("HF_REPO_ID", "booldii/onchain-risk-tgn")

# (local destination, filename as stored in the HF repo) - same basename on both sides here,
# kept separate in case the two ever need to diverge.
ARTIFACTS = [
    (PROJECT_ROOT / "data" / "processed" / "tgn_checkpoint.pt", "tgn_checkpoint.pt"),
    (PROJECT_ROOT / "data" / "processed" / "eth_fresh_temporal_graph.pt", "eth_fresh_temporal_graph.pt"),
    (PROJECT_ROOT / "data" / "interim" / "eth_fresh_sample.parquet", "eth_fresh_sample.parquet"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_demo_artifacts")


def main() -> None:
    for local_path, hf_filename in ARTIFACTS:
        if local_path.exists():
            logger.info("already present, skipping: %s", local_path)
            continue
        logger.info("downloading %s from %s ...", hf_filename, HF_REPO_ID)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=HF_REPO_ID, filename=hf_filename, local_dir=local_path.parent)
        logger.info("saved to %s", local_path)


if __name__ == "__main__":
    main()