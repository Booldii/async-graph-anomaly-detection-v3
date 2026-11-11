"""
Stateless helpers shared by the batch feature engine (build_features.py) and the incremental
one (incremental_feature_engine.py) - kept in one place so both compute the exact same math
for every feature that doesn't actually need a different, streaming-friendly implementation
(see incremental_feature_engine.py's docstring for which ones do).
"""

import numpy as np
import pandas as pd

NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


def entropy_normalized(counts: pd.DataFrame, group_col: str, key_col: str) -> pd.Series:
    """Normalized entropy of the count distribution within group_col"""
    total = counts.groupby(group_col)["count"].transform("sum")
    share = counts["count"] / total
    entropy_term = -share * np.log(share.clip(lower=1e-12))
    raw_entropy = counts.assign(entropy_term=entropy_term).groupby(group_col)["entropy_term"].sum()
    n_keys = counts.groupby(group_col)[key_col].nunique()
    return (raw_entropy / np.log(n_keys.clip(lower=2))).fillna(0)


def entropy_normalized_from_counts(counts: dict) -> float:
    """Same formula as entropy_normalized, for a single group's already-aggregated
    {key: count} dict - used by the incremental engine, which keeps running count dicts
    per address instead of a global long-form DataFrame."""
    if not counts:
        return 0.0
    values = np.array(list(counts.values()), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    share = values / total
    raw_entropy = float(-(share * np.log(np.clip(share, 1e-12, None))).sum())
    n_keys = max(len(counts), 2)
    return raw_entropy / np.log(n_keys)


def is_round_value(values: np.ndarray, sig: int = 2) -> np.ndarray:
    """AML heuristic: does the amount have a suspiciously low number of significant digits"""
    positive = values > 0
    magnitude = np.zeros_like(values)
    magnitude[positive] = 10 ** np.floor(np.log10(values[positive]))
    scaled = np.divide(values, magnitude, out=np.zeros_like(values), where=positive)
    reconstructed = np.round(scaled, sig - 1) * magnitude
    return positive & (np.abs(values - reconstructed) < 1e-6)


def is_round_value_scalar(value: float, sig: int = 2) -> bool:
    """Single-value form of is_round_value, for the incremental engine's per-event updates."""
    if value <= 0:
        return False
    magnitude = 10 ** np.floor(np.log10(value))
    scaled = value / magnitude
    reconstructed = round(scaled, sig - 1) * magnitude
    return abs(value - reconstructed) < 1e-6