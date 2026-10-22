"""
Alternative to chain_listener.py: publishes through the same Redpanda topic, so
inference_consumer.py can't tell the difference between a live
and a replayed event. Replays data/interim/eth_fresh_sample.parquet (the same 14-day sample
used to build/train the TGN in training stage) at a sped-up pace, preserving each event's relative
on-chain timing - so the bursty, most events arrive within the same ~12s block pattern that
justified using a broker in the first place is preserved.

Why this exists at all: Replaying known, already-fetched data gives a deterministic, repeatable sequence of
events instead of relying solely on live-streaming data.

Scope: no filtering (mint/burn, contracts, etc) - this replays the raw sample exactly as a live
feed would have delivered it; any such filtering is a downstream (inference_consumer) decision,
consistent with chain_listener.py also not filtering.

Known gap vs. chain_listener.py's rows: eth_fresh_sample.parquet has no logIndex/blockNumber
(src/data/bq_fetch_day.py's FETCH_SQL never selected them) - these two fields are simply absent
from replayed rows rather than fabricated.
"""

import argparse
import asyncio
import json
import logging

import pandas as pd
from aiokafka import AIOKafkaProducer
from common import KAFKA_BOOTSTRAP_SERVERS, PROJECT_ROOT, TOPIC_TRANSFERS

SAMPLE_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_sample.parquet"
# Full 14-day sample replayed this much faster than it actually happened - ~10 real minutes
# for the whole window, slow enough that bursts/gaps are still visible as they go by.
DEFAULT_SPEEDUP = 2000.0
ROW_FIELDS = [
    "hash", "from", "to", "value", "timeStamp", "contractAddress", "tokenDecimal", "gasPrice", "gasUsed",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replay_producer")


async def run(speedup: float, limit: int | None) -> None:
    df = pd.read_parquet(SAMPLE_PATH).sort_values("timeStamp").reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    span_minutes = (df["timeStamp"].iloc[-1] - df["timeStamp"].iloc[0]) / speedup / 60
    logger.info("replaying %d events at %gx speed (~%.1f real minutes)", len(df), speedup, span_minutes)

    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()
    try:
        prev_ts: int | None = None
        for i, row in enumerate(df.to_dict("records")):
            if prev_ts is not None:
                gap = (row["timeStamp"] - prev_ts) / speedup
                if gap > 0:
                    await asyncio.sleep(gap)
            prev_ts = row["timeStamp"]

            payload = {field: row[field] for field in ROW_FIELDS}
            await producer.send_and_wait(TOPIC_TRANSFERS, json.dumps(payload).encode("utf-8"))

            if i % 1000 == 0:
                logger.info("published %d/%d", i, len(df))
    finally:
        await producer.stop()
    logger.info("replay finished")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speedup", type=float, default=DEFAULT_SPEEDUP,
        help=f"replay speed multiplier relative to real on-chain pacing (default: {DEFAULT_SPEEDUP}x)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only replay the first N events")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.speedup, args.limit))
    except KeyboardInterrupt:
        logger.info("stopped")


if __name__ == "__main__":
    main()