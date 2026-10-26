"""
Consumes normalized transfer events from the Redpanda topic and maintains a
live TGN risk score per address. Per event: forward pass through the Stage 2 checkpoint's
memory -> gnn -> RiskHead, delta vs the previous score in Redis, alert if the delta exceeds
ALERT_THRESHOLD, write the new score back to Redis.

State ownership: the TGN memory tensor lives in this process, as one big tensor - exactly like
Stage 2 training. Redis only holds (1) the last risk score per address, and (2) a periodic backup snapshot of the in-process memory/neighbor/edge-feature state. The learned weights never
change at inference time, so they always load fresh from the Stage 2 checkpoint regardless of
whether a snapshot is used for the mutable runtime state.

Cold-start: the checkpoint only has memory slots for the ~224k addresses seen during previous Stages.
COLD_START_RESERVED_SLOTS reserves extra, zero-initialized slots beyond that for addresses never seen before.
"""

import asyncio
import io
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from aiokafka import AIOKafkaConsumer
from redis import asyncio as aioredis
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory

from common import KAFKA_BOOTSTRAP_SERVERS, REDIS_URL, TOPIC_TRANSFERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# train_tgn.py lives in a sibling package (src/models/), when this script runs as
# `uv run python src/serving/inference_consumer.py` - added explicitly to reuse its model
# classes/constants directly rather than duplicating them.
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models"))
from train_tgn import (  # noqa: E402
    CHECKPOINT_PATH,
    EMBEDDING_DIM,
    MEMORY_DIM,
    NEIGHBOR_SIZE,
    TIME_DIM,
    GraphAttentionEmbedding,
    RiskHead,
)

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_temporal_graph.pt"

# log(value), log(gasUsed), log(gasPrice) - matches build_temporal_graph.py's msg vector
MSG_DIM = 3

COLD_START_RESERVED_SLOTS = 50_000

# Same exclusion as build_temporal_graph.py - the model never saw these as graph nodes during training
NULL_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}

# Provisional - next step is to tune this empirically against the actual delta distribution from a replay run.
ALERT_THRESHOLD = 0.3

# An address needs this many recorded transfers since this process started before it's
# allowed to trigger an alert - below that, its score reflects a mostly cold/under-warmed memory,
# so deltas are dominated by cold-start noise rather than real risk signal. Provisional, same as
# ALERT_THRESHOLD above.
MIN_TRANSFERS_FOR_ALERT = 5

REDIS_SCORE_KEY_PREFIX = "tgn:score:"
REDIS_SNAPSHOT_KEY = "tgn:snapshot"
REDIS_ALERTS_KEY = "tgn:alerts"
# Bounds the list - without a cap it would grow for as long as the process runs
ALERTS_LIST_MAX_LEN = 1000
# every 2000 events: frequent enough that a crash loses little ground
SNAPSHOT_EVERY_N_EVENTS = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inference_consumer")


class EdgeFeatureStore:
    """Mirrors build_temporal_graph.py's data.t/data.msg, indexed by the same e_id scheme
    LastNeighborLoader.insert() assigns. During Stage 2 training, insert() was called once per
    event in dataset order, so e_id i is row i of the historical graph's t/msg tensors - loaded
    here as the initial contents. New edges seen during live/replay serving get the next
    sequential e_id and are appended, so a neighbor lookup's e_id always resolves correctly
    whether it points at a training-time edge or one seen during this process's lifetime.

    Pre-allocated, doubling-capacity growable array. Grows unboundedly for the life of one
    process - fine at this scale (tens of bytes/edge), not engineered for a 24/7 service that
    would need to prune edges no node's recent-neighbor window still references.
    """

    def __init__(self, historical_t: torch.Tensor, historical_msg: torch.Tensor) -> None:
        n = historical_t.size(0)
        capacity = max(n * 2, 1024)
        self._t = torch.zeros(capacity, dtype=historical_t.dtype, device=historical_t.device)
        self._msg = torch.zeros(
            capacity, historical_msg.size(-1), dtype=historical_msg.dtype, device=historical_msg.device
        )
        self._t[:n] = historical_t
        self._msg[:n] = historical_msg
        self._len = n

    def append(self, t: torch.Tensor, msg: torch.Tensor) -> None:
        n_new = t.size(0)
        if self._len + n_new > self._t.size(0):
            self._grow(self._len + n_new)
        self._t[self._len : self._len + n_new] = t
        self._msg[self._len : self._len + n_new] = msg
        self._len += n_new

    def _grow(self, min_capacity: int) -> None:
        new_capacity = max(min_capacity, self._t.size(0) * 2)
        new_t = torch.zeros(new_capacity, dtype=self._t.dtype, device=self._t.device)
        new_msg = torch.zeros(new_capacity, self._msg.size(-1), dtype=self._msg.dtype, device=self._msg.device)
        new_t[: self._len] = self._t[: self._len]
        new_msg[: self._len] = self._msg[: self._len]
        self._t, self._msg = new_t, new_msg

    def lookup(self, e_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._t[e_id], self._msg[e_id]

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Everything accumulated so far (historical prefix + whatever was appended during this
        process's lifetime) - the same shape __init__ expects, so a restore just re-seeds a new
        instance from this instead of from the Stage 2 training graph."""
        return self._t[: self._len].clone(), self._msg[: self._len].clone()


class InferenceState:
    """Bundles the mutable runtime state carried across every event - everything here either
    is the model's memory of the world, or is needed to correctly index into it."""

    def __init__(
        self,
        memory: TGNMemory,
        gnn: GraphAttentionEmbedding,
        risk_head: RiskHead,
        neighbor_loader: LastNeighborLoader,
        edge_store: EdgeFeatureStore,
        node_index: dict[str, int],
        next_cold_start_id: int,
        total_capacity: int,
        device: torch.device,
        transfer_count: dict[str, int] | None = None,
    ) -> None:
        self.memory = memory
        self.gnn = gnn
        self.risk_head = risk_head
        self.neighbor_loader = neighbor_loader
        self.edge_store = edge_store
        self.node_index = node_index
        self.next_cold_start_id = next_cold_start_id
        self.total_capacity = total_capacity
        self.device = device
        self.assoc = torch.empty(total_capacity, dtype=torch.long, device=device)
        # address (lowercase) -> number of events seen as sender or receiver since this process
        # started. Persisted across restarts via the Redis snapshot.
        self.transfer_count: dict[str, int] = transfer_count if transfer_count is not None else {}


def load_inference_state(device: torch.device, snapshot: dict | None) -> InferenceState:
    """snapshot, when given (from try_restore_snapshot), is a prior run's full runtime state -
    restored 1:1 at the same total_capacity it was saved at. Without one, cold-boots from the Stage 2
    checkpoint as before."""
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    if snapshot is not None:
        expected_total_capacity = len(checkpoint["node_index"]) + COLD_START_RESERVED_SLOTS
        if snapshot["memory"].size(0) != expected_total_capacity:
            logger.warning(
                "Redis snapshot is incompatible with the current checkpoint (snapshot sized "
                "for %d slots, current checkpoint needs %d) - checkpoint was likely "
                "retrained/replaced since this snapshot was taken. Discarding snapshot, "
                "cold-booting from checkpoint instead.",
                snapshot["memory"].size(0), expected_total_capacity,
            )
            snapshot = None

    if snapshot is not None:
        node_index: dict[str, int] = dict(snapshot["node_index"])
        next_cold_start_id = snapshot["next_cold_start_id"]
        # Always derived from the original Stage 2 checkpoint's node count - the reserved-slots
        # buffer size is a fixed constant, stable across restarts regardless of how many addresses
        # got cold-started since.
        total_capacity = len(checkpoint["node_index"]) + COLD_START_RESERVED_SLOTS
    else:
        node_index = dict(checkpoint["node_index"])
        next_cold_start_id = len(node_index)
        total_capacity = len(node_index) + COLD_START_RESERVED_SLOTS

    memory = TGNMemory(
        total_capacity,
        MSG_DIM,
        MEMORY_DIM,
        TIME_DIM,
        message_module=IdentityMessage(MSG_DIM, MEMORY_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)
    # TGNMemory overrides train()/eval(): the first train->eval transition flushes the message
    # store into memory via an extra GRU pass. A freshly constructed module defaults to
    # self.training=True, so calling eval() after loading real memory values below would silently
    # overwrite them with that flush's output. Call eval() now, while the message store is still
    # empty and the memory buffer is still all zeros - the flush is then a harmless no-op,
    # immediately superseded by the real values assigned below.
    memory.eval()

    if snapshot is not None:
        # GRU/time_enc weights still come from the checkpoint (static) - only memory/last_update
        # content is the snapshot's, and it's already sized to total_capacity - direct assign.
        memory.load_state_dict({k: v for k, v in checkpoint["memory"].items() if k not in ("memory", "last_update", "_assoc")}, strict=False)
        with torch.no_grad():
            memory.memory[:] = snapshot["memory"].to(device)
            memory.last_update[:] = snapshot["last_update"].to(device)
    else:
        # memory/last_update/_assoc are sized by num_nodes, so they can't be loaded directly
        # into a module constructed with a larger (expanded) num_nodes - load everything else
        # (GRU/time_enc weights, fixed-size regardless of num_nodes) normally, then manually
        # copy the known addresses' memory/last_update into the first original_num_nodes rows,
        # leaving the reserved cold-start rows at their fresh zero-initialized default.
        original_num_nodes = len(node_index)
        mem_state = dict(checkpoint["memory"])
        saved_memory = mem_state.pop("memory")
        saved_last_update = mem_state.pop("last_update")
        mem_state.pop("_assoc", None)
        memory.load_state_dict(mem_state, strict=False)
        with torch.no_grad():
            memory.memory[:original_num_nodes] = saved_memory.to(device)
            memory.last_update[:original_num_nodes] = saved_last_update.to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEMORY_DIM, out_channels=EMBEDDING_DIM, msg_dim=MSG_DIM, time_enc=memory.time_enc
    ).to(device)
    gnn.load_state_dict(checkpoint["gnn"])

    risk_head = RiskHead(in_channels=EMBEDDING_DIM).to(device)
    risk_head.load_state_dict(checkpoint["risk_head"])

    gnn.eval()
    risk_head.eval()

    neighbor_loader = LastNeighborLoader(total_capacity, size=NEIGHBOR_SIZE, device=device)
    if snapshot is not None:
        neighbor_loader.neighbors[:] = snapshot["neighbors"].to(device)
        neighbor_loader.e_id[:] = snapshot["e_id"].to(device)
        neighbor_loader.cur_e_id = snapshot["cur_e_id"]
        edge_store = EdgeFeatureStore(snapshot["edge_t"].to(device), snapshot["edge_msg"].to(device))
        logger.info(
            "restored runtime state from Redis snapshot: %d addresses (incl. cold-started), "
            "%d historical+live edges",
            len(node_index), snapshot["edge_t"].size(0),
        )
    else:
        nl_state = checkpoint["neighbor_loader"]
        neighbor_loader.neighbors[:original_num_nodes] = nl_state["neighbors"].to(device)
        neighbor_loader.e_id[:original_num_nodes] = nl_state["e_id"].to(device)
        neighbor_loader.cur_e_id = nl_state["cur_e_id"]

        graph_checkpoint = torch.load(GRAPH_PATH, map_location=device, weights_only=False)
        historical_data = graph_checkpoint["data"]
        edge_store = EdgeFeatureStore(historical_data.t.to(device), historical_data.msg.to(device))
        logger.info(
            "loaded checkpoint cold (no Redis snapshot found): %d known addresses, %d reserved "
            "cold-start slots, %d historical edges",
            original_num_nodes, COLD_START_RESERVED_SLOTS, historical_data.t.size(0),
        )

    transfer_count = dict(snapshot.get("transfer_count", {})) if snapshot is not None else {}

    return InferenceState(
        memory, gnn, risk_head, neighbor_loader, edge_store, node_index,
        next_cold_start_id=next_cold_start_id, total_capacity=total_capacity, device=device,
        transfer_count=transfer_count,
    )


def process_event(state: InferenceState, row: dict) -> tuple[str, float, bool] | None:
    """Pure tensor computation - mutates state in place (memory, neighbor_loader,
    edge_store, node_index, next_cold_start_id, transfer_count). Returns (sender_address,
    new_risk_score, alert_eligible), or None if the event was skipped (mint/burn, or the
    cold-start buffer is exhausted)."""
    # chain_listener.py emits EIP-55 checksummed addresses. On the other hand: node_index was
    # built from BigQuery's lowercase-hex convention. Normalize both to lowercase so a live
    # address that is already known doesn't get spuriously cold-started on a case mismatch.
    from_addr = row["from"].lower()
    to_addr = row["to"].lower()

    if from_addr in NULL_ADDRESSES or to_addr in NULL_ADDRESSES:
        return None

    for addr in (from_addr, to_addr):
        if addr not in state.node_index:
            if state.next_cold_start_id >= state.total_capacity:
                logger.error(
                    "cold-start buffer exhausted (%d slots) - dropping new address %s",
                    state.total_capacity, addr,
                )
                continue
            state.node_index[addr] = state.next_cold_start_id
            state.next_cold_start_id += 1

    if from_addr not in state.node_index or to_addr not in state.node_index:
        return None  # buffer was exhausted above for one of the two addresses

    for addr in (from_addr, to_addr):
        state.transfer_count[addr] = state.transfer_count.get(addr, 0) + 1
    eligible = state.transfer_count[from_addr] >= MIN_TRANSFERS_FOR_ALERT

    device = state.device
    src = torch.tensor([state.node_index[from_addr]], dtype=torch.long, device=device)
    dst = torch.tensor([state.node_index[to_addr]], dtype=torch.long, device=device)
    t = torch.tensor([row["timeStamp"]], dtype=torch.long, device=device)
    msg = torch.tensor(
        [[math.log1p(row["value"]), math.log1p(row["gasUsed"]), math.log1p(row["gasPrice"])]],
        dtype=torch.float,
        device=device,
    )

    n_id = torch.cat([src, dst]).unique()
    n_id, edge_index, e_id = state.neighbor_loader(n_id)
    state.assoc[n_id] = torch.arange(n_id.size(0), device=device)

    with torch.no_grad():
        z, last_update = state.memory(n_id)
        edge_t, edge_msg = state.edge_store.lookup(e_id)
        z = state.gnn(z, last_update, edge_index, edge_t, edge_msg)
        pred = state.risk_head(z[state.assoc[src]])
        new_score = float(pred.item())

        state.memory.update_state(src, dst, t, msg)
        state.neighbor_loader.insert(src, dst)
    state.edge_store.append(t, msg)

    return from_addr, new_score, eligible


async def handle_score(redis_client: aioredis.Redis, address: str, new_score: float, eligible: bool) -> None:
    """Delta/alert against Redis's last-known score, then persists the new one. An alert both
    logs and pushes a record to the tgn:alerts list - the backend half of the alert panel"""
    key = REDIS_SCORE_KEY_PREFIX + address
    prev_raw = await redis_client.get(key)
    if prev_raw is not None:
        prev_score = float(prev_raw)
        delta = new_score - prev_score
        if eligible and delta > ALERT_THRESHOLD:
            logger.warning(
                "ALERT address=%s score=%.4f prev=%.4f delta=%+.4f", address, new_score, prev_score, delta
            )
            record = json.dumps(
                {
                    "address": address,
                    "score": new_score,
                    "prev_score": prev_score,
                    "delta": delta,
                    "timestamp": time.time(),
                }
            )
            # LPUSH (newest first) + LTRIM to cap the list - one round trip each, both cheap locally.
            # Not pipelined together since this already only runs on the rare alert-firing path
            await redis_client.lpush(REDIS_ALERTS_KEY, record)
            await redis_client.ltrim(REDIS_ALERTS_KEY, 0, ALERTS_LIST_MAX_LEN - 1)
    await redis_client.set(key, str(new_score))


async def snapshot_to_redis(redis_binary: aioredis.Redis, state: InferenceState) -> None:
    """Serializes everything try_restore_snapshot() needs to reconstruct an identical InferenceState.
    Runs on the main loop - same reasoning as handle_score: a local Redis write is fast enough that
    the simplicity/correctness of awaiting it beats shaving a few hundred ms off one event every
    SNAPSHOT_EVERY_N_EVENTS events."""
    edge_t, edge_msg = state.edge_store.snapshot()
    buffer = io.BytesIO()
    torch.save(
        {
            "memory": state.memory.memory.cpu(),
            "last_update": state.memory.last_update.cpu(),
            "node_index": state.node_index,
            "next_cold_start_id": state.next_cold_start_id,
            "transfer_count": state.transfer_count,
            "neighbors": state.neighbor_loader.neighbors.cpu(),
            "e_id": state.neighbor_loader.e_id.cpu(),
            "cur_e_id": state.neighbor_loader.cur_e_id,
            "edge_t": edge_t.cpu(),
            "edge_msg": edge_msg.cpu(),
        },
        buffer,
    )
    await redis_binary.set(REDIS_SNAPSHOT_KEY, buffer.getvalue())
    logger.info("snapshotted runtime state to Redis (%d bytes)", buffer.tell())


async def try_restore_snapshot(redis_binary: aioredis.Redis) -> dict | None:
    blob = await redis_binary.get(REDIS_SNAPSHOT_KEY)
    if blob is None:
        return None
    return torch.load(io.BytesIO(blob), weights_only=False)


async def consume_loop(state: InferenceState, redis_client: aioredis.Redis, redis_binary: aioredis.Redis) -> None:
    consumer = AIOKafkaConsumer(
        TOPIC_TRANSFERS, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, group_id="inference-consumer",
        # Only matters the very first time this consumer group ever starts - "earliest" means it
        # reads everything already in the topic rather than only what arrives after it subscribes.
        # Removes the startup-order race against the producer entirely, instead of trying to
        # synchronize container start timing in docker-compose.
        auto_offset_reset="earliest",
    )
    await consumer.start()
    logger.info("consuming from topic %r...", TOPIC_TRANSFERS)
    events_processed = 0
    try:
        async for msg in consumer:
            row = json.loads(msg.value.decode())
            result = process_event(state, row)
            if result is not None:
                address, new_score, eligible = result
                await handle_score(redis_client, address, new_score, eligible)
            events_processed += 1
            if events_processed % SNAPSHOT_EVERY_N_EVENTS == 0:
                await snapshot_to_redis(redis_binary, state)
    finally:
        await consumer.stop()
        if events_processed > 0:
            # Covers a deliberate stop: a restart right after should always
            # resume from the freshest possible state.
            await snapshot_to_redis(redis_binary, state)
            logger.info("final snapshot saved on shutdown")


async def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    redis_binary = aioredis.from_url(REDIS_URL, decode_responses=False)
    try:
        snapshot = await try_restore_snapshot(redis_binary)
        state = load_inference_state(device, snapshot)
        await consume_loop(state, redis_client, redis_binary)
    finally:
        await redis_client.aclose()
        await redis_binary.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("stopped")