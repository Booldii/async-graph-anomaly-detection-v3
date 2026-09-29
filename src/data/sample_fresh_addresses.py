"""
Reduces the fresh BigQuery pull down to a small, address-stratified MVP sample for soft labeling.

Pipeline: dedup on full-row identity first (see notebooks/02_fresh_pool_sanity_check.ipynb).
Then sample a fraction of EOA addresses - contracts excluded up front via the fresh contract lookup.

Stratified by activity (out_degree + in_degree), with equal count per quantile bucket.
This method of sampling guarantees representation across the whole activity range.

1-hop neighbor inclusion: once addresses are sampled, every row touching a sampled address is kept (from or to).

Why DuckDB for the heavy steps (dedup, degree, 1-hop expansion) instead of plain pandas:
DuckDB's engine can spill to disk under memory pressure instead of requiring 2x the dataset size in RAM at once.
Plain pandas has no automatic disk spilling. This is the safeguard agaist an OOM issue.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd

from find_fresh_pull import find_latest_fresh_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FRESH_CONTRACT_FLAGS_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_address_contract_flags.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_sample.parquet"

NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}

DEFAULT_FRACTION = 0.01
DEFAULT_STRATA = 4
DEFAULT_SEED = 42

def load_and_dedup(con: duckdb.DuckDBPyConnection, fresh_path: Path) -> None:
    """Creates the deduplicated data as a DuckDB table - materialized once and reused by later queries"""
    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{fresh_path}')").fetchone()[0]
    con.execute(f"CREATE TABLE fresh_deduped AS SELECT DISTINCT * FROM read_parquet('{fresh_path}')")
    n_deduped = con.execute("SELECT COUNT(*) FROM fresh_deduped").fetchone()[0]
    print(f"deduplicated: {n_rows:,} -> {n_deduped:,} rows ({n_rows - n_deduped:,} dropped)")


def compute_eoa_activity(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """out_degree + in_degree per address, excluding contracts and null/burn addresses -
    the activity measure used to stratify the sample."""
    contract_flags = pd.read_parquet(FRESH_CONTRACT_FLAGS_PATH)
    con.register("contract_flags", contract_flags)

    activity = con.execute(
        f"""
        WITH graph AS (
            SELECT "from", "to" FROM fresh_deduped
            WHERE "from" NOT IN {tuple(NULL_ADDRESSES)} AND "to" NOT IN {tuple(NULL_ADDRESSES)}
        ), degree AS (
            SELECT address, COUNT(*) AS degree FROM (
                SELECT "from" AS address FROM graph
                UNION ALL
                SELECT "to" AS address FROM graph
            ) GROUP BY address
        )
        SELECT d.address, d.degree
        FROM degree d
        LEFT JOIN contract_flags cf ON cf.address = d.address
        WHERE COALESCE(cf.is_contract, false) = false
        """
    ).df()
    return activity.set_index("address")["degree"]


def sample_addresses_stratified(activity: pd.Series, fraction: float, n_strata: int, seed: int) -> set[str]:
    """Equal-count sampling per activity quantile bucket"""
    strata = pd.qcut(activity, q=n_strata, labels=False, duplicates="drop")
    actual_n_strata = strata.nunique()
    n_per_stratum = max(1, int(len(activity) * fraction / actual_n_strata))

    sampled = []
    for stratum_id in sorted(strata.unique()):
        pool = pd.Series(activity.index[strata == stratum_id])
        n = min(n_per_stratum, len(pool))
        sampled.append(pool.sample(n=n, random_state=seed))
    return set(pd.concat(sampled))


def expand_to_neighbors(con: duckdb.DuckDBPyConnection, sampled_addresses: set[str]) -> pd.DataFrame:
    """1-hop: keeps every row touching a sampled address on either side"""
    con.register("sampled", pd.DataFrame({"address": list(sampled_addresses)}))
    return con.execute(
        """
        SELECT fd.* FROM fresh_deduped fd
        WHERE fd."from" IN (SELECT address FROM sampled) OR fd."to" IN (SELECT address FROM sampled)
        """
    ).df()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION, help="fraction of EOA addresses to sample")
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA, help="number of activity quantile buckets")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    args = parser.parse_args()

    fresh_path = find_latest_fresh_path(RAW_DATA_DIR)
    print(f"using latest fresh pull: {fresh_path.name}")

    con = duckdb.connect()
    load_and_dedup(con, fresh_path)

    activity = compute_eoa_activity(con)
    print(f"{len(activity):,} EOA addresses eligible for sampling")

    sampled_addresses = sample_addresses_stratified(activity, args.fraction, args.strata, args.seed)
    print(f"sampled {len(sampled_addresses):,} addresses ({len(sampled_addresses) / len(activity):.2%} of eligible EOAs)")

    result = expand_to_neighbors(con, sampled_addresses)
    print(f"expanded to {len(result):,} rows (1-hop neighborhood)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
