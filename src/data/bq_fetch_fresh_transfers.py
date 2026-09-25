"""
Fetches fresh ERC-20 TOKEN TRANSFERS from BigQuery, joined with TOKENS
(decimals, symbol) and TRANSACTIONS (gas) via inner join, to match the standard row
schema used by build_features.py - the same shape as the historical BybitML dataset
- so the existing pipeline runs on it unchanged.

INNER JOIN means: transfers of tokens absent from TOKENS (unrecognized/unregistered
contracts) or whose parent transaction is missing from TRANSACTIONS are dropped, not
kept with nulls.

Default window: the last 2 full weeks (14 days) ending on the last full UTC day, matching
the historical dataset's 14-day span.

Safety: defaults to a DRY RUN (bytes estimate only - nothing executed).
The real pull requires the explicit --execute flag, and streams results in
chunks straight to a local parquet file.
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery, bigquery_storage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"

DATASET = "bigquery-public-data.crypto_ethereum"
DEFAULT_WINDOW_DAYS = 14
# Most ERC-20 tokens use 18 decimals. Used only as a fallback when tokens.decimals is NULL
DEFAULT_DECIMALS_FALLBACK = 18

FETCH_SQL = f"""
SELECT
  tt.transaction_hash AS `hash`,
  tt.from_address AS `from`,
  tt.to_address AS `to`,
  SAFE_DIVIDE(
    SAFE_CAST(tt.value AS FLOAT64),
    POW(10, COALESCE(SAFE_CAST(tok.decimals AS INT64), {DEFAULT_DECIMALS_FALLBACK}))
  ) AS value,
  UNIX_SECONDS(tt.block_timestamp) AS timeStamp,
  tt.token_address AS contractAddress,
  tok.symbol AS tokenSymbol,
  COALESCE(SAFE_CAST(tok.decimals AS INT64), {DEFAULT_DECIMALS_FALLBACK}) AS tokenDecimal,
  tx.gas_price AS gasPrice,
  tx.receipt_gas_used AS gasUsed,
  -- ETH - matches how gasFee behaves in the historical BybitML dataset
  CAST(
    (CAST(tx.gas_price AS BIGNUMERIC) * CAST(tx.receipt_gas_used AS BIGNUMERIC)) / 1e18
    AS FLOAT64
   )AS gasFee
FROM `{DATASET}.token_transfers` AS tt
INNER JOIN `{DATASET}.tokens` AS tok
  ON tt.token_address = tok.address
INNER JOIN `{DATASET}.transactions` AS tx
  ON tt.transaction_hash = tx.`hash`
  AND tx.block_timestamp >= @range_start AND tx.block_timestamp < @range_end
WHERE tt.block_timestamp >= @range_start AND tt.block_timestamp < @range_end
"""


def last_full_utc_day() -> date:
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def range_bounds(end_day: date, window_days: int) -> tuple[datetime, datetime, date]:
    """[start_day, end_day] inclusive, window_days calendar days total"""
    start_day = end_day - timedelta(days=window_days - 1)
    start = datetime.combine(start_day, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_day, time.min, tzinfo=timezone.utc) + timedelta(days=1)
    return start, end, start_day


def build_job_config(start: datetime, end: datetime, *, dry_run: bool) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        dry_run=dry_run,
        use_query_cache=not dry_run,
        query_parameters=[
            bigquery.ScalarQueryParameter("range_start", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("range_end", "TIMESTAMP", end),
        ],
    )


def estimate_cost(client: bigquery.Client, start: datetime, end: datetime) -> None:
    job_config = build_job_config(start, end, dry_run=True)
    bytes_processed = client.query(FETCH_SQL, job_config=job_config).total_bytes_processed
    gb = bytes_processed / 1e9
    print(f"[dry-run] query generates about {gb:8.2f} GB...")


def run_fetch(client: bigquery.Client, start: datetime, end: datetime, output_path: Path) -> None:
    """Streams the query result to parquet in chunks - never holds the full results in memory at once"""
    job_config = build_job_config(start, end, dry_run=False)
    result = client.query(FETCH_SQL, job_config=job_config).result()

    bqstorage_client = bigquery_storage.BigQueryReadClient(credentials=client._credentials)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    try:
        for chunk in result.to_dataframe_iterable(bqstorage_client=bqstorage_client):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"{total_rows:,} rows written so far...", end="\r")
    finally:
        if writer is not None:
            writer.close()
    print(f"\nsaved {total_rows:,} rows to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="GCP project ID")
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=None,
        help="last UTC day of the window, YYYY-MM-DD (default: last full UTC day)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"window size in calendar days (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run the query and bill the GCP account. Without this flag, the "
        "script only estimates cost (dry run) and exits.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output parquet path (default: data/raw/eth_fresh_<start>_<end>.parquet)",
    )
    args = parser.parse_args()

    end_day = args.end_date or last_full_utc_day()
    start, end, start_day = range_bounds(end_day, args.days)
    client = bigquery.Client(project=args.project)

    print(f"window: {start_day} to {end_day} (UTC, {args.days} days)")
    estimate_cost(client, start, end)

    if not args.execute:
        print("\nDry run completed. Run with --execute to actually fetch and save the data.")
        return

    output_path = args.output or (OUTPUT_DIR / f"eth_fresh_{start_day}_{end_day}.parquet")
    run_fetch(client, start, end, output_path)


if __name__ == "__main__":
    main()