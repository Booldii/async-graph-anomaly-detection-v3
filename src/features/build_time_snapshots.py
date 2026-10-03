"""
Builds point-in-time cumulative feature snapshots for the sampled fresh addresses, one per
calendar day in the fetch window.

At each daily cutoff, computes address features from the full cumulative history up to that
point - preserving feature constructor's "background = everything known so far"
reference for value normalization - then keeps only the addresses that were
actually active on that specific day.
"""

from pathlib import Path

import pandas as pd

from build_features import build_address_features_for_scoring

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_sample.parquet"
CONTRACT_FLAGS_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_address_contract_flags.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_sample_features_snapshots.parquet"

NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}

WINDOW_DAYS = 14

def build_snapshots(df: pd.DataFrame, contract_flags: pd.Series) -> pd.DataFrame:
    window_start = pd.Timestamp(df["timeStamp"].min(), unit="s", tz="UTC").floor("D")

    snapshots = []
    for day in range(1, WINDOW_DAYS + 1):
        day_start = window_start + pd.Timedelta(days=day - 1)
        cutoff = window_start + pd.Timedelta(days=day)

        df_cumulative = df[df["timeStamp"] < cutoff.timestamp()]
        if df_cumulative.empty:
            continue

        today = df[(df["timeStamp"] >= day_start.timestamp()) & (df["timeStamp"] < cutoff.timestamp())]
        today_addresses = (set(today["from"]) | set(today["to"])) - NULL_ADDRESSES
        if not today_addresses:
            continue

        features = build_address_features_for_scoring(df_cumulative, contract_flags)
        features = features.loc[features.index.isin(today_addresses)].copy()
        features["snapshot_day"] = day
        print(f"day {day}: {len(today_addresses):,} active addresses -> {len(features):,} scored (EOA only)")
        snapshots.append(features)

    return pd.concat(snapshots)


def main() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    contract_flags = pd.read_parquet(CONTRACT_FLAGS_PATH).set_index("address")["is_contract"]

    result = build_snapshots(df, contract_flags)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH)

    print(f"\n{len(result):,} (address, day) snapshot rows across {result['snapshot_day'].nunique()} days")
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()