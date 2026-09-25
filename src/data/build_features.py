"""
Builds a feature table per EOA address for XGBoost training.

Input: Token transfers DataFrame in standard schema (hash/from/to/value/
timeStamp/tokenSymbol/contractAddress/gasPrice/gasUsed/gasFee/from_label/to_label)

Output: one row per EOA address (contracts and null/burn addresses excluded), "label" column (0/1) -
NaN addresses miss the training dataset because we don't we "ground truth" for them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "bybit_ec_all14days_Tx.parquet"
CONTRACT_FLAGS_PATH = PROJECT_ROOT / "data" / "interim" / "bybit_address_contract_flags.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_address_features.parquet"

NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


def load_raw_transfers(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def split_graph_and_mint_burn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits rows into a graph (real relationships between addresses) and mint/burn transfers"""
    is_mint_or_burn = df["from"].isin(NULL_ADDRESSES) | df["to"].isin(NULL_ADDRESSES)
    return df.loc[~is_mint_or_burn].copy(), df.loc[is_mint_or_burn].copy()


def build_address_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized address -> labeled (0/1) table"""
    occurrences = pd.concat(
        [
            df[["from", "from_label"]].rename(columns={"from": "address", "from_label": "label"}),
            df[["to", "to_label"]].rename(columns={"to": "address", "to_label": "label"}),
        ],
        ignore_index=True,
    ).dropna(subset=["label"])

    labels = occurrences.drop_duplicates(subset="address").reset_index(drop=True)
    return labels[~labels["address"].isin(NULL_ADDRESSES)].reset_index(drop=True)


def add_value_zscore(df_graph: pd.DataFrame, labeled_addresses: set[str]) -> pd.DataFrame:
    """Robust z-score (median/MAD) of log(value) separately per contractAddress.
    Decided to this technique because classic z-score is prone to masking: outliers
    inflate the std used to measure them. Logarithm is still necessary:
    without it, MAD on the raw scale is degenerate (dominated by 'dust'
    transfers), so even moderately large transfers explode into absurd z-scores.

    The reference (median/MAD) is calculated from the background - rows NOT touching any
    labeled address - not from the entire population. For dominant tokens 90%+ of all transactions
    in this dataset touch a labeled address. By calculating the reference on the whole population, we would measure
    "how unusual relative to typical hack activity", not relative to normal traffic - which is
    the exact opposite reference of what a fresh pull from BQ will have (where background will probably dominate).
    Contracts without any background get a fallback: a reference calculated on the entire population."""
    df_graph = df_graph.copy()
    df_graph["log_value"] = np.log1p(df_graph["value"])

    is_background = ~(
        df_graph["from"].isin(labeled_addresses) | df_graph["to"].isin(labeled_addresses)
    )
    background = df_graph[is_background]

    def _median_mad(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        median = frame.groupby("contractAddress")["log_value"].median()
        mad = ( # MAD = median( |x_i - median(x)| )
            frame["log_value"]
            .sub(frame.groupby("contractAddress")["log_value"].transform("median"))
            .abs()
            .groupby(frame["contractAddress"])
            .median()
            * 1.4826
        )
        return median, mad

    bg_median, bg_mad = _median_mad(background)
    all_median, all_mad = _median_mad(df_graph)

    reference_median = bg_median.combine_first(all_median)
    reference_mad = bg_mad.combine_first(all_mad)

    token_median = df_graph["contractAddress"].map(reference_median)
    token_mad = df_graph["contractAddress"].map(reference_mad)
    df_graph["value_zscore"] = np.where(
        token_mad > 0, (df_graph["log_value"] - token_median) / token_mad, 0.0
    )
    return df_graph


def _safe_set(x: object) -> set:
    return x if isinstance(x, set) else set()


def _entropy_normalized(counts: pd.DataFrame, group_col: str, key_col: str) -> pd.Series:
    """Normalized entropy of the count distribution within group_col"""
    total = counts.groupby(group_col)["count"].transform("sum")
    share = counts["count"] / total
    entropy_term = -share * np.log(share.clip(lower=1e-12))
    raw_entropy = counts.assign(entropy_term=entropy_term).groupby(group_col)["entropy_term"].sum()
    n_keys = counts.groupby(group_col)[key_col].nunique()
    return (raw_entropy / np.log(n_keys.clip(lower=2))).fillna(0)


def _is_round_value(values: np.ndarray, sig: int = 2) -> np.ndarray:
    """AML heuristic: does the amount have a suspiciously low number of significant digits"""
    positive = values > 0
    magnitude = np.zeros_like(values)
    magnitude[positive] = 10 ** np.floor(np.log10(values[positive]))
    scaled = np.divide(values, magnitude, out=np.zeros_like(values), where=positive)
    reconstructed = np.round(scaled, sig - 1) * magnitude
    return positive & (np.abs(values - reconstructed) < 1e-6)


def build_base_features(sent: pd.DataFrame, received: pd.DataFrame) -> pd.DataFrame:
    """Degree, z-score value, gas, token diversity, activity timespan."""
    out_degree = sent.groupby("address").size().rename("out_degree")
    in_degree = received.groupby("address").size().rename("in_degree")

    value_sent = sent.groupby("address")["value_zscore"].agg(
        value_sent_zscore_mean="mean",
        value_sent_zscore_median="median",
        value_sent_zscore_std="std",
        value_sent_zscore_max="max",
    )
    value_received = received.groupby("address")["value_zscore"].agg(
        value_received_zscore_mean="mean",
        value_received_zscore_median="median",
        value_received_zscore_std="std",
        value_received_zscore_max="max",
    )

    # gas is a transaction attribute (hash), not a transfer attribute - dedup before averaging,
    # otherwise a transaction with multiple transfers would overrepresent its gas in the average
    sent_tx_level = sent.drop_duplicates(subset=["address", "hash"])
    gas = sent_tx_level.groupby("address")[["gasPrice", "gasUsed", "gasFee"]].mean()
    gas = gas.rename(
        columns={"gasPrice": "gas_price_mean", "gasUsed": "gas_used_mean", "gasFee": "gas_fee_mean"}
    )

    # keying by contractAddress, not tokenSymbol - symbol collisions between contracts
    tokens_sent = sent.groupby("address")["contractAddress"].nunique().rename("tokens_sent")
    tokens_received = received.groupby("address")["contractAddress"].nunique().rename("tokens_received")

    activity = (
        pd.concat(
            [sent.groupby("address")["timeStamp"].agg(["min", "max"]),
             received.groupby("address")["timeStamp"].agg(["min", "max"])]
        )
        .groupby("address")
        .agg({"min": "min", "max": "max"})
    )
    # metadata for time split (not a model feature)
    first_seen_timestamp = activity["min"].rename("first_seen_timestamp")

    features = pd.concat(
        [out_degree, in_degree, value_sent, value_received, gas,
         tokens_sent, tokens_received, first_seen_timestamp],
        axis=1,
    )
    features[["out_degree", "in_degree", "tokens_sent", "tokens_received"]] = (
        features[["out_degree", "in_degree", "tokens_sent", "tokens_received"]].fillna(0)
    )
    features["total_degree"] = features["out_degree"] + features["in_degree"]
    features["unique_tokens"] = features["tokens_sent"] + features["tokens_received"]
    features["value_flow_gap_zscore"] = (
        features["value_sent_zscore_mean"] - features["value_received_zscore_mean"]
    )
    return features


def build_mint_burn_features(mint_burn_rows: pd.DataFrame) -> pd.DataFrame:
    """Mint/burn involvement as a behavioral feature"""
    mint_events = mint_burn_rows[mint_burn_rows["from"].isin(NULL_ADDRESSES)]
    burn_events = mint_burn_rows[mint_burn_rows["to"].isin(NULL_ADDRESSES)]

    mint_count = mint_events.groupby("to").size().rename("mint_count")
    burn_count = burn_events.groupby("from").size().rename("burn_count")

    features = pd.concat([mint_count, burn_count], axis=1).fillna(0)
    features["has_mint_activity"] = features["mint_count"] > 0
    features["has_burn_activity"] = features["burn_count"] > 0
    return features


def build_neighbor_sets(sent: pd.DataFrame, received: pd.DataFrame) -> pd.Series:
    """Set of all counterparties per address - basis for reciprocity and later
    the share of contract counterparties / counterparties with a known label."""
    out_neighbors = sent.groupby("address")["to"].apply(set)
    in_neighbors = received.groupby("address")["from"].apply(set)
    index = out_neighbors.index.union(in_neighbors.index)
    out_neighbors = out_neighbors.reindex(index)
    in_neighbors = in_neighbors.reindex(index)
    return pd.Series(
        [_safe_set(o) | _safe_set(i) for o, i in zip(out_neighbors, in_neighbors)],
        index=index,
        name="all_neighbors",
    ), pd.Series(
        [_safe_set(o) & _safe_set(i) for o, i in zip(out_neighbors, in_neighbors)],
        index=index,
        name="reciprocal_neighbors",
    )


def build_relational_features(
    sent: pd.DataFrame, all_neighbors: pd.Series, reciprocal_neighbors: pd.Series
) -> pd.DataFrame:
    """reciprocity_rate - bidirectional relationships and out_counterparty_entropy
    (concentration of sent value among counterparties - smurfing intentionally disperses it)."""
    reciprocity_rate = pd.Series(
        [len(r) / len(a) if len(a) > 0 else 0.0 for r, a in zip(reciprocal_neighbors, all_neighbors)],
        index=all_neighbors.index,
        name="reciprocity_rate",
    )

    cp_value = sent.groupby(["address", "to"])["value"].sum().rename("count").reset_index()
    out_counterparty_entropy = _entropy_normalized(cp_value, "address", "to").rename(
        "out_counterparty_entropy"
    )

    return pd.concat([reciprocity_rate, out_counterparty_entropy], axis=1)


def build_shape_and_temporal_features(sent: pd.DataFrame, received: pd.DataFrame) -> pd.DataFrame:
    """Value roundness (AML heuristic), number of active days, activity hour entropy."""
    round_value_share = (
        sent.assign(is_round=_is_round_value(sent["value"].to_numpy()))
        .groupby("address")["is_round"]
        .mean()
        .rename("round_value_share")
    )

    sent_days = pd.to_datetime(sent["timeStamp"], unit="s").dt.floor("D")
    received_days = pd.to_datetime(received["timeStamp"], unit="s").dt.floor("D")
    all_days = pd.concat(
        [pd.DataFrame({"address": sent["address"], "day": sent_days}),
         pd.DataFrame({"address": received["address"], "day": received_days})]
    )
    days_active = all_days.groupby("address")["day"].nunique().rename("days_active")

    sent_hours = pd.to_datetime(sent["timeStamp"], unit="s").dt.hour
    received_hours = pd.to_datetime(received["timeStamp"], unit="s").dt.hour
    all_hours = pd.concat(
        [pd.DataFrame({"address": sent["address"], "hour": sent_hours}),
         pd.DataFrame({"address": received["address"], "hour": received_hours})]
    )
    hour_counts = all_hours.groupby(["address", "hour"]).size().rename("count").reset_index()
    hour_of_day_entropy = _entropy_normalized(hour_counts, "address", "hour").rename(
        "hour_of_day_entropy"
    )

    features = pd.concat([round_value_share, days_active, hour_of_day_entropy], axis=1)
    features["round_value_share"] = features["round_value_share"].fillna(0)
    return features


def build_self_loop_features(df_graph: pd.DataFrame) -> pd.DataFrame:
    """self_loop_count: transfers from an address to itself - a cheap wash-trading indicator"""
    self_loop_rows = df_graph[df_graph["from"] == df_graph["to"]]
    self_loop_count = self_loop_rows.groupby("from").size().rename("self_loop_count")
    return self_loop_count.to_frame()


def build_contract_features(
    address_index: pd.Index,
    contract_flags: pd.Series,
    all_neighbors: pd.Series,
) -> pd.DataFrame:
    """is_contract + share of counterparties that are contracts"""
    contract_address_set = set(contract_flags[contract_flags].index)
    neighbors_aligned = all_neighbors.reindex(address_index)

    def share(neighbors: set, reference: set) -> float:
        return len(neighbors & reference) / len(neighbors) if neighbors else 0.0

    features = pd.DataFrame(index=address_index)
    features["is_contract"] = contract_flags.reindex(address_index)
    features["neighbor_contract_share"] = [
        share(_safe_set(n), contract_address_set) for n in neighbors_aligned
    ]
    return features


def build_address_features(df: pd.DataFrame, contract_flags: pd.Series) -> pd.DataFrame:
    """Orchestrates all feature groups. Returns a training set: one row per EOA address with
    a confirmed label 0/1 - contracts, null/burn addresses, and unlabeled addrsses are excluded."""
    df_graph, mint_burn_rows = split_graph_and_mint_burn(df)
    labels = build_address_labels(df)
    df_graph = add_value_zscore(df_graph, set(labels["address"]))

    sent = df_graph.rename(columns={"from": "address"})
    received = df_graph.rename(columns={"to": "address"})

    all_neighbors, reciprocal_neighbors = build_neighbor_sets(sent, received)

    features = build_base_features(sent, received)
    features = features.join(build_mint_burn_features(mint_burn_rows), how="left")
    features[["mint_count", "burn_count"]] = features[["mint_count", "burn_count"]].fillna(0)
    # astype(bool) after fillna: join with NaN results in object dtype which XGBoost doesn't accept
    features["has_mint_activity"] = features["has_mint_activity"].fillna(False).astype(bool)
    features["has_burn_activity"] = features["has_burn_activity"].fillna(False).astype(bool)

    features = features.join(
        build_relational_features(sent, all_neighbors, reciprocal_neighbors), how="left"
    )
    features = features.join(build_shape_and_temporal_features(sent, received), how="left")
    features["tx_per_active_day"] = features["total_degree"] / features["days_active"]
    # days_active only exists to derive the rate above - dropped so it can't be mistaken for a usable feature
    features = features.drop(columns=["days_active"])

    features = features.join(build_self_loop_features(df_graph), how="left")
    features["self_loop_count"] = features["self_loop_count"].fillna(0)

    features = features.join(
        build_contract_features(features.index, contract_flags, all_neighbors),
        how="left",
    )

    # contracts and null/burn addresses are not the scoring target - is_contract
    # unresolved by default is treated as an EOA. Used only to filter, then dropped -
    # it's constant for every remaining row, so it can't be mistaken for a usable feature.
    features["is_contract"] = features["is_contract"].fillna(False).astype(bool)
    features = features[~features["is_contract"]].drop(columns=["is_contract"])

    features = features.join(labels.set_index("address")["label"], how="inner")
    return features


def main() -> None:
    df = load_raw_transfers(RAW_DATA_PATH)
    contract_flags = pd.read_parquet(CONTRACT_FLAGS_PATH).set_index("address")["is_contract"]

    features = build_address_features(df, contract_flags)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH)

    print(f"labeled EOAs addresses in the training set: {len(features)}")
    print(features["label"].value_counts())
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()