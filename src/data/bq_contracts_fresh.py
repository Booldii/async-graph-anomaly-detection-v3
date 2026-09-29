"""
Checks which addresses from the fresh BigQuery pull (data/raw/eth_fresh_*.parquet) are
deployed smart contracts, analogous to bq_contracts.py for the historical ByBit dataset.

Why a separate script instead of reusing bq_contracts.py: that script passes the address
list inline as an ArrayQueryParameter. That works for the historical dataset's ~59k addresses,
but the fresh pull has above 8M distinct addresses - a literal array that size would blow
past BigQuery's request-size limit (~10MB) long before it got anywhere near billing.
The fix is a JOIN against a scratch table instead of an inline array,
which is a different query shape, not just a different input path.

Free optimization: any address that appears as contractAddress in the transfer data is
already, by definition, a token contract - no BigQuery lookup needed for those. Only
addresses seen exclusively as from/to actually need checking.
"""

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from find_fresh_pull import find_latest_fresh_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_address_contract_flags.parquet"

CONTRACTS_TABLE = "bigquery-public-data.crypto_ethereum.contracts"
SCRATCH_TABLE_EXPIRATION_HOURS = 1

NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}

JOIN_QUERY = f"""
SELECT ua.address
FROM `{{scratch_table}}` AS ua
JOIN `{CONTRACTS_TABLE}` AS c
ON ua.address = c.address
"""


def estimate_cost(client: bigquery.Client) -> None:
    """Dry-runs a full scan of contracts.address - the dominant cost component of the real JOIN"""
    job_config = bigquery.QueryJobConfig(dry_run=True)
    query_job = client.query(f"SELECT address FROM `{CONTRACTS_TABLE}`", job_config=job_config)
    bytes_processed = query_job.total_bytes_processed
    gb = bytes_processed / 1e9
    print(f"[dry-run, proxy] scanning {CONTRACTS_TABLE}.address: {gb:.2f} GB")
    print("[dry-run] this is a stand-in for the real JOIN's cost")

def load_address_universe(fresh_path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Returns: addresses needing a BigQuery lookup, addresses already known"""
    df = pd.read_parquet(fresh_path, columns=["from", "to", "contractAddress"])
    known_contracts = set(df["contractAddress"].unique())
    all_addresses = set(df["from"].unique()) | set(df["to"].unique())
    to_query = all_addresses - known_contracts - NULL_ADDRESSES
    return pd.DataFrame({"address": sorted(to_query)}), known_contracts


def upload_scratch_table(client: bigquery.Client, project: str, dataset: str, table: str, addresses: pd.DataFrame) -> str:
    dataset_ref = bigquery.DatasetReference(project, dataset)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

    table_ref = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField("address", "STRING")],
        write_disposition="WRITE_TRUNCATE",
    )
    load_job = client.load_table_from_dataframe(addresses, table_ref, job_config=job_config)
    load_job.result()

    bq_table = client.get_table(table_ref)
    bq_table.expires = pd.Timestamp.now('UTC') + pd.Timedelta(hours=SCRATCH_TABLE_EXPIRATION_HOURS)
    client.update_table(bq_table, ["expires"])
    return table_ref


def run_join(client: bigquery.Client, table_ref: str) -> set[str]:
    job_config = bigquery.QueryJobConfig(dry_run=False, use_query_cache=False)
    query = JOIN_QUERY.format(scratch_table=table_ref)
    result = client.query(query, job_config=job_config).result()
    return {row["address"] for row in result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="GCP project ID")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run the JOIN. Without this flag, the script only estimates cost and exits.",
    )
    parser.add_argument("--scratch-dataset", default="scratch", help="BQ dataset for the temporary address table")
    parser.add_argument("--scratch-table", default="address_universe_fresh", help="name of the temporary address table")
    args = parser.parse_args()

    fresh_path = find_latest_fresh_path(RAW_DATA_DIR)
    print(f"using latest fresh pull: {fresh_path.name}")

    client = bigquery.Client(project=args.project)
    estimate_cost(client)

    if not args.execute:
        print("\nDry run completed. Run with --execute to actually run the JOIN and save results.")
        return

    to_query, known_contracts = load_address_universe(fresh_path)
    print(f"addresses known to be contracts for free (via contractAddress): {len(known_contracts):,}")
    print(f"addresses requiring a BigQuery lookup: {len(to_query):,}")

    table_ref = upload_scratch_table(client, client.project, args.scratch_dataset, args.scratch_table, to_query)
    try:
        matched_contracts = run_join(client, table_ref)
    finally:
        client.delete_table(table_ref, not_found_ok=True)

    queried_result = to_query.copy()
    queried_result["is_contract"] = queried_result["address"].isin(matched_contracts)
    known_result = pd.DataFrame({"address": sorted(known_contracts), "is_contract": True})
    result = pd.concat([queried_result, known_result], ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(result):,} addresses to: {OUTPUT_PATH}")
    print(result["is_contract"].value_counts())


if __name__ == "__main__":
    main()