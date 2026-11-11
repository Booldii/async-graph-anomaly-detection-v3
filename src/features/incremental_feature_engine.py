"""
Incremental, per-transfer-event feature engine - the streaming counterpart to
build_time_snapshots.py's daily batch recompute. Emits one feature row per sender event,
replacing the (address, snapshot_day) cadence with (address, event_row_id)

Every feature family in build_features.py reduces to counters/sums/sets/count-tables that
update exactly in O(1) amortized per event - see common.py for the stateless math (entropy,
round-value) shared with the batch engine, and AddressState below. The one feature with no
exact incremental update rule over an ever-growing history is value_zscore's per-token
median/MAD.

TokenState resolves this with a bounded recent window (plain deque, exact median/MAD via
numpy), rather than a t-digest sketch over the full unbounded history - a  pivot from earlier plan,
made after empirically testing t-digest against this actual data. Two things broke the t-digest approach
specifically:
- real token value distributions here are strongly non-stationary - a residual digest fed with
 |x - median| computed against a periodically-refreshed median permanently mixes residuals taken
 against very different past medians, which is not anapproximation artifact but a real bug under drift
- these distributions are also heavily zero-inflated/skewed, where IQR-based MAD proxies diverge
 from true MAD by >1000%. A bounded window sidesteps both: it avoids the O(N^2) blowup just as well
 as a sketch would, computes median/MAD exactly, and is a better conceptual fit for a live risk signal
 anyway - "unusual relative to recent behavior", rather than "relative to all of history since this
 token's first ever transfer".

Processing is a single pass over the full time-sorted sample (mint/burn rows included, so
mint/burn counters stay causally correct relative to interleaved real transfers - batch's
day-level cutoffs can't see within-day order at all, so this is strictly finer, not just
faster). Contract addresses never get their own AddressState - only their flag is consulted,
to decide whether to update sender/receiver-side state and whether to emit.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import NULL_ADDRESSES, entropy_normalized_from_counts, is_round_value_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_sample.parquet"
CONTRACT_FLAGS_PATH = PROJECT_ROOT / "data" / "interim" / "eth_fresh_address_contract_flags.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_sample_features_stream.parquet"

# Both provisional, like ALERT_THRESHOLD/MIN_TRANSFERS_FOR_ALERT in inference_consumer.py.
# WINDOW_SIZE: how many of a token's most recent log-values the median/MAD reference is
# computed from - bounded so cost per refresh is O(WINDOW_SIZE), not O(history-so-far).
# REFRESH_EVERY_N_EVENTS: how often (in new background points) the cached median/MAD is
# recomputed from the window - smaller means less staleness, at the cost of more recomputations.
WINDOW_SIZE = 1000
REFRESH_EVERY_N_EVENTS = 50


@dataclass(slots=True)
class TokenState:
    """Per-contractAddress bounded-window state for value_zscore's median/MAD reference"""

    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    events_since_refresh: int = 0
    cached_median: float | None = None
    cached_mad: float | None = None

    def update_and_score(self, log_value: float) -> float:
        zscore = self._zscore(log_value)
        self.window.append(log_value)
        self.events_since_refresh += 1
        if self.events_since_refresh >= REFRESH_EVERY_N_EVENTS:
            self._refresh()
        return zscore

    def _refresh(self) -> None:
        arr = np.fromiter(self.window, dtype=np.float64)
        median = float(np.median(arr))
        self.cached_median = median
        self.cached_mad = float(np.median(np.abs(arr - median)) * 1.4826)
        self.events_since_refresh = 0

    def _zscore(self, log_value: float) -> float:
        if self.cached_median is None or not self.cached_mad:
            return 0.0  # cold-start, before the first refresh - matches add_value_zscore's own
            # fallback-to-zero for tokens whose MAD is 0.
        return (log_value - self.cached_median) / self.cached_mad


@dataclass(slots=True)
class AddressState:
    out_degree: int = 0
    in_degree: int = 0
    sent_zscores: list[float] = field(default_factory=list)
    received_zscores: list[float] = field(default_factory=list)
    seen_tx_hashes: set = field(default_factory=set)
    gas_price_sum: float = 0.0
    gas_used_sum: float = 0.0
    gas_fee_sum: float = 0.0
    gas_tx_count: int = 0
    tokens_sent: set = field(default_factory=set)
    tokens_received: set = field(default_factory=set)
    out_neighbors: set = field(default_factory=set)
    in_neighbors: set = field(default_factory=set)
    counterparty_value: dict = field(default_factory=dict)  # to_address -> cumulative value sent
    hour_counts: dict = field(default_factory=dict)  # hour (0-23) -> touch count, sent + received
    days_active: set = field(default_factory=set)  # UTC calendar-day indices (timestamp // 86400)
    round_count: int = 0
    sent_count: int = 0
    mint_count: int = 0
    burn_count: int = 0
    has_mint_activity: bool = False
    has_burn_activity: bool = False
    self_loop_count: int = 0


def _zscore_stats(values: list[float]) -> tuple[float, float, float, float]:
    """mean/median/std/max of an address's own sent-or-received z-scores so far - NaN for an empty
    list and ddof=1 (pandas std default, itself NaN for a single-element list)."""
    if not values:
        return math.nan, math.nan, math.nan, math.nan
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else math.nan
    return float(arr.mean()), float(np.median(arr)), std, float(arr.max())


def _emit_row(address: str, s: AddressState, contract_set: set, event_row_id: int) -> dict:
    sent_mean, sent_median, sent_std, sent_max = _zscore_stats(s.sent_zscores)
    recv_mean, recv_median, recv_std, recv_max = _zscore_stats(s.received_zscores)

    all_neighbors = s.out_neighbors | s.in_neighbors
    reciprocal = s.out_neighbors & s.in_neighbors
    reciprocity_rate = len(reciprocal) / len(all_neighbors) if all_neighbors else 0.0
    contract_neighbors = sum(1 for n in all_neighbors if n in contract_set)
    neighbor_contract_share = contract_neighbors / len(all_neighbors) if all_neighbors else 0.0

    total_degree = s.out_degree + s.in_degree
    return {
        "address": address,
        "event_row_id": event_row_id,
        "value_sent_zscore_mean": sent_mean,
        "value_sent_zscore_median": sent_median,
        "value_sent_zscore_std": sent_std,
        "value_sent_zscore_max": sent_max,
        "value_received_zscore_mean": recv_mean,
        "value_received_zscore_median": recv_median,
        "value_received_zscore_std": recv_std,
        "value_received_zscore_max": recv_max,
        "value_flow_gap_zscore": sent_mean - recv_mean,
        "gas_price_mean": s.gas_price_sum / s.gas_tx_count,
        "gas_used_mean": s.gas_used_sum / s.gas_tx_count,
        "gas_fee_mean": s.gas_fee_sum / s.gas_tx_count,
        "tx_per_active_day": total_degree / len(s.days_active),
        "hour_of_day_entropy": entropy_normalized_from_counts(s.hour_counts),
        "round_value_share": s.round_count / s.sent_count,
        "mint_count": float(s.mint_count),
        "burn_count": float(s.burn_count),
        "has_mint_activity": s.has_mint_activity,
        "has_burn_activity": s.has_burn_activity,
        "out_degree": float(s.out_degree),
        "in_degree": float(s.in_degree),
        "total_degree": float(total_degree),
        "tokens_sent": float(len(s.tokens_sent)),
        "tokens_received": float(len(s.tokens_received)),
        "unique_tokens": float(len(s.tokens_sent) + len(s.tokens_received)),
        "reciprocity_rate": reciprocity_rate,
        "out_counterparty_entropy": entropy_normalized_from_counts(s.counterparty_value),
        "self_loop_count": float(s.self_loop_count),
        "neighbor_contract_share": neighbor_contract_share,
    }


def _update_mint_burn(
    from_addr: str, to_addr: str, addr_states: dict[str, AddressState], contract_set: set
) -> None:
    """Mint/burn rows never enter the graph and never update value_zscore/degree/neighbor/gas state -
    they only touch the two mint/burn counters, exactly like build_mint_burn_features in the batch engine."""
    if from_addr in NULL_ADDRESSES and to_addr not in NULL_ADDRESSES and to_addr not in contract_set:
        s = addr_states.setdefault(to_addr, AddressState())
        s.mint_count += 1
        s.has_mint_activity = True
    if to_addr in NULL_ADDRESSES and from_addr not in NULL_ADDRESSES and from_addr not in contract_set:
        s = addr_states.setdefault(from_addr, AddressState())
        s.burn_count += 1
        s.has_burn_activity = True


def process_sample(df: pd.DataFrame, contract_set: set) -> pd.DataFrame:
    """Single causal pass over the full time-sorted sample. event_row_id is assigned on the
    full, unfiltered sort (df's index at this point) so it stays stable regardless of which
    rows later turn out to be mint/burn - build_temporal_graph.py's load_edges() must assign it
    the same way, before its own mint/burn filtering, for the label join to line up."""
    addr_states: dict[str, AddressState] = {}
    token_states: dict[str, TokenState] = {}
    rows_out: list[dict] = []

    log_values = np.log1p(df["value"].to_numpy(dtype=np.float64))
    day_idx = df["timeStamp"].to_numpy() // 86400
    hours = pd.to_datetime(df["timeStamp"], unit="s", utc=True).dt.hour.to_numpy()

    columns = zip(
        df.index.to_numpy(),
        df["from"].to_numpy(),
        df["to"].to_numpy(),
        df["value"].to_numpy(dtype=np.float64),
        df["contractAddress"].to_numpy(),
        df["hash"].to_numpy(),
        df["gasPrice"].to_numpy(dtype=np.float64),
        df["gasUsed"].to_numpy(dtype=np.float64),
        df["gasFee"].to_numpy(dtype=np.float64),
        log_values,
        day_idx,
        hours,
    )

    for event_row_id, from_addr, to_addr, value, token, tx_hash, gas_price, gas_used, gas_fee, log_value, day, hour in columns:
        if from_addr in NULL_ADDRESSES or to_addr in NULL_ADDRESSES:
            _update_mint_burn(from_addr, to_addr, addr_states, contract_set)
            continue

        token_state = token_states.setdefault(token, TokenState())
        zscore = token_state.update_and_score(log_value)

        from_is_contract = from_addr in contract_set
        to_is_contract = to_addr in contract_set

        if not from_is_contract:
            s = addr_states.setdefault(from_addr, AddressState())
            s.out_degree += 1
            s.sent_zscores.append(zscore)
            if tx_hash not in s.seen_tx_hashes:
                s.seen_tx_hashes.add(tx_hash)
                s.gas_price_sum += gas_price
                s.gas_used_sum += gas_used
                s.gas_fee_sum += gas_fee
                s.gas_tx_count += 1
            s.tokens_sent.add(token)
            s.out_neighbors.add(to_addr)
            s.counterparty_value[to_addr] = s.counterparty_value.get(to_addr, 0.0) + value
            s.hour_counts[hour] = s.hour_counts.get(hour, 0) + 1
            s.days_active.add(day)
            if is_round_value_scalar(value):
                s.round_count += 1
            s.sent_count += 1
            if from_addr == to_addr:
                s.self_loop_count += 1

        if not to_is_contract:
            r = addr_states.setdefault(to_addr, AddressState())
            r.in_degree += 1
            r.received_zscores.append(zscore)
            r.tokens_received.add(token)
            r.in_neighbors.add(from_addr)
            r.hour_counts[hour] = r.hour_counts.get(hour, 0) + 1
            r.days_active.add(day)

        if not from_is_contract:
            rows_out.append(_emit_row(from_addr, addr_states[from_addr], contract_set, event_row_id))

    # indexed by address, like build_address_features_for_scoring's output -
    # generate_soft_labels.py relies on this.
    return pd.DataFrame(rows_out).set_index("address")


def main() -> None:
    df = pd.read_parquet(SAMPLE_PATH)
    df = df.sort_values("timeStamp", kind="stable").reset_index(drop=True)

    contract_flags = pd.read_parquet(CONTRACT_FLAGS_PATH).set_index("address")["is_contract"]
    contract_set = set(contract_flags[contract_flags].index)

    result = process_sample(df, contract_set)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH)

    print(f"{len(result):,} (address, event) rows across {result.index.nunique():,} addresses")
    print(f"saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()