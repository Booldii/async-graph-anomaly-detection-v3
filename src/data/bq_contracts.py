"""
A BigQuery client to check which addresses from the historical ByBit dataset are
deployed smart contracts. It uses the public table, which contains addresses
of all contracts ever deployed on Ethereum.

By default, the script performs only a dry run (estimating the number of
processed GBs, without actually running the query and without charging
the billing account). The actual query requires an explicit --execute flag.
"""

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bybit_ec_all14days_Tx.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "bybit_address_contract_flags.parquet"

CONTRACTS_TABLE = "bigquery-public-data.crypto_ethereum.contracts"

QUERY = f"""
SELECT address
FROM `{CONTRACTS_TABLE}`
WHERE address IN UNNEST(@addresses)
"""

def load_address_universe() -> list[str]:
    df = pd.read_parquet(DATA_PATH, columns=["from", "to"])
    addresses = set(df["from"].unique()) | set(df["to"].unique())
    return sorted(addresses)


def build_job_config(addresses: list[str], *, dry_run: bool) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("addresses", "STRING", addresses)],
        dry_run=dry_run,
        use_query_cache=not dry_run,
    )

def estimate_cost(client: bigquery.Client, addresses: list[str]) -> None:
    job_config = build_job_config(addresses, dry_run=True)
    query_job = client.query(QUERY, job_config=job_config)
    bytes_processed = query_job.total_bytes_processed

    print(f"[dry-run] addresses to check: {len(addresses)}")
    print(f"[dry-run] to process: {bytes_processed / 1e9:.2f} GB")


def run_query(client: bigquery.Client, addresses: list[str]) -> pd.DataFrame:
    job_config = build_job_config(addresses, dry_run=False)
    query_job = client.query(QUERY, job_config=job_config)
    contract_addresses = {row["address"] for row in query_job.result()}

    result = pd.DataFrame({"address": addresses})
    result["is_contract"] = result["address"].isin(contract_addresses)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None, help="GCP project ID")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the query and bill the gcp account",
    )
    args = parser.parse_args()

    addresses = load_address_universe()
    client = bigquery.Client(project=args.project)

    estimate_cost(client, addresses)

    if not args.execute:
        print("\nDry run completed. Run with the --execute flag to actually execute the query")
        return

    result = run_query(client, addresses)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(result)} addresses -> {OUTPUT_PATH}")
    print(result["is_contract"].value_counts())


if __name__ == "__main__":
    main()