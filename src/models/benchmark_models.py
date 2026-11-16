"""
Benchmarks one or more saved TGN checkpoints against the stream graph's validation split,
measuring the inference-time cost of each: disk size, parameter count, peak VRAM, and
throughput/latency over a single-event-at-a-time replay of val_data.
"""

import argparse
import gc
import time
from pathlib import Path

import torch
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory

from train_tgn import GRAPH_MODES, GraphAttentionEmbedding, RiskHead

COLUMN_WIDTHS = {"File": 50}
DEFAULT_COLUMN_WIDTH = 16


def inspect_arch_from_state(state: dict) -> dict:
    """Reconstructs architecture dimensions directly from the saved tensors' shapes."""
    return {
        "memory_dim": state["memory"]["memory"].shape[1],
        "embedding_dim": state["risk_head"]["lin1.weight"].shape[0],
        "neighbor_size": state["neighbor_loader"]["neighbors"].shape[1],
    }


@torch.no_grad()
def benchmark_checkpoint(ckpt_path: Path, data, num_nodes: int, device: torch.device) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    disk_size_mb = ckpt_path.stat().st_size / (1024**2)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = inspect_arch_from_state(state)

    msg_dim = data.msg.size(-1)
    time_dim = 100

    memory = TGNMemory(
        num_nodes, msg_dim, arch["memory_dim"], time_dim,
        message_module=IdentityMessage(msg_dim, arch["memory_dim"], time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=arch["memory_dim"], out_channels=arch["embedding_dim"],
        msg_dim=msg_dim, time_enc=memory.time_enc, dropout=0.0,
    ).to(device)

    risk_head = RiskHead(in_channels=arch["embedding_dim"]).to(device)
    neighbor_loader = LastNeighborLoader(num_nodes, size=arch["neighbor_size"], device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    memory.load_state_dict(state["memory"])
    gnn.load_state_dict(state["gnn"])
    risk_head.load_state_dict(state["risk_head"])

    memory.eval()
    gnn.eval()
    risk_head.eval()

    params_count = sum(
        p.numel() for m in (memory, gnn, risk_head) for p in m.parameters()
    )
    val_data = data[data.val_mask]
    loader = TemporalDataLoader(val_data, batch_size=100)

    def step(batch):
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        _ = risk_head(z[assoc[src]])
        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    # GPU warmup
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        step(batch.to(device))

    # Actual timing measurement
    memory.reset_state()
    neighbor_loader.reset_state()
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    total_events = 0
    for batch in loader:
        batch = batch.to(device)
        step(batch)
        total_events += batch.src.size(0)

    torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - t_start

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    throughput = total_events / elapsed_sec
    latency_ms = (elapsed_sec / len(loader)) * 1000

    return {
        "File": ckpt_path.name,
        "Mem/Nbr dim": f"{arch['memory_dim']} / {arch['neighbor_size']}",
        "Params": f"{params_count:,}",
        "Disk [MB]": f"{disk_size_mb:.2f}",
        "Peak VRAM [MB]": f"{peak_vram_mb:.1f}",
        "Time [s]": f"{elapsed_sec:.3f}",
        "Batch [ms]": f"{latency_ms:.2f}",
        "Events/s": f"{throughput:,.0f}",
    }


def print_results_table(results: list[dict]) -> None:
    headers = list(results[0].keys())
    col_widths = [COLUMN_WIDTHS.get(h, DEFAULT_COLUMN_WIDTH) for h in headers]
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)

    print("\n" + row_fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    for r in results:
        print(row_fmt.format(*(str(v) for v in r.values())))


def main():
    parser = argparse.ArgumentParser(description="Benchmarks saved TGN checkpoints against each other.")
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", required=True,
        help="paths to the .pt checkpoint files to compare",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_ckpt = torch.load(GRAPH_MODES["stream"]["graph"], weights_only=False)
    data = graph_ckpt["data"].to(device)
    num_nodes = len(graph_ckpt["node_index"])

    results = [benchmark_checkpoint(p, data, num_nodes, device) for p in args.checkpoints]
    print_results_table(results)


if __name__ == "__main__":
    main()