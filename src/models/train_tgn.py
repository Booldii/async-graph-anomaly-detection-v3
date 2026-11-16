"""
Trains a Temporal Graph Network (TGN) on the sampled fresh transfer graph
(build_temporal_graph.py's output), using the per-event soft labels as the regression target.

Architecture: PyTorch Geometric's reference TGN (memory module with a GRU-based updater + a
graph-attention embedding module over a bounded recent-neighbor window), adapted for node-level
risk regression - a small MLP head maps a node's embedding to a risk score in [0,1], trained
with MSE against consensus_score.

Decisions made regarding architecture of training:
- Loss: unweighted MSE - decided MVP scope
- Memory is .detach()-ed after every batch's optimizer step - bounds the autograd graph to one
  batch's depth (this is what prevents the OOM that would otherwise hit mid-epoch).
- Only src-node predictions are supervised; events whose src has no label (contract, or
  otherwise missing) still update memory/neighbor state, but contribute no loss.
- Train = days 1-11, validation = days 12-14, continuing the same memory/neighbor state from train
  into validation within one epoch - mirrors how this would actually run in production:
  validation is "the future" relative to training, not a separate population.

run_training() is the single training/eval routine, shared by this script's own CLI and by
tune_tgn.py's Optuna sweep. Keeping this one function shared means the final, promoted retrain
after a sweep goes through the exact same code path as the existing reference runs.
"""

import argparse
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.nn import Linear
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TGNMemory,
    TimeEncoder,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --graph stream trains on incremental_feature_engine.py's per-transfer soft labels
# actually serves from - the two are kept side by side for the val_spearman comparison
GRAPH_MODES = {
    "daily": {
        "graph": PROJECT_ROOT / "data" / "processed" / "eth_fresh_temporal_graph.pt",
        "checkpoint": PROJECT_ROOT / "data" / "processed" / "tgn_checkpoint.pt",
    },
    "stream": {
        "graph": PROJECT_ROOT / "data" / "processed" / "eth_fresh_temporal_graph_stream.pt",
        "checkpoint": PROJECT_ROOT / "data" / "processed" / "tgn_checkpoint_stream_final.pt",
    },
}

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "tgn_risk_scoring"

# Defaults for the manual CLI path (python train_tgn.py --graph ...) - a sweep (tune_tgn.py)
# overrides any subset of these via run_training's hparams dict. memory_dim/neighbor_size are
# also the two levers with by far the biggest VRAM impact.
# Hardcoded hparams that gave best val results in Optuna's sweep:
DEFAULT_HPARAMS = {
    "memory_dim": 80,
    "time_dim": 100,
    "embedding_dim": 80,
    "neighbor_size": 13,
    "batch_size": 100,
    "learning_rate": 4.332e-4,
    "min_lr": 1e-5,
    "dropout": 0.2675,
}
NUM_EPOCHS = 19


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder, dropout: float
    ):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=dropout, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class RiskHead(torch.nn.Module):
    """Node-level regression head - replaces the usual link-predictor decoder since our target
    is a per-address risk score, rather than edge existence."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.lin1 = Linear(in_channels, in_channels)
        self.lin2 = Linear(in_channels, 1)

    def forward(self, z):
        h = self.lin1(z).relu()
        return torch.sigmoid(self.lin2(h)).squeeze(-1)


def run_training(
    graph_mode: str,
    epochs: int,
    hparams: dict | None = None,
    trial: "optuna.Trial | None" = None,
    mlflow_experiment: str = MLFLOW_EXPERIMENT_NAME,
    checkpoint_path: Path | None = None,
) -> dict:
    """Builds the model, trains for `epochs`, saves the checkpoint, and returns
    {"val_spearman", "val_mse", "checkpoint_path"}"""
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    paths = GRAPH_MODES[graph_mode]
    save_path = checkpoint_path or paths["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    checkpoint = torch.load(paths["graph"], weights_only=False)
    data = checkpoint["data"].to(device)
    num_nodes = len(checkpoint["node_index"])
    msg_dim = data.msg.size(-1)

    train_data = data[data.train_mask]
    val_data = data[data.val_mask]
    train_loader = TemporalDataLoader(train_data, batch_size=hp["batch_size"])
    val_loader = TemporalDataLoader(val_data, batch_size=hp["batch_size"])

    memory = TGNMemory(
        num_nodes,
        msg_dim,
        hp["memory_dim"],
        hp["time_dim"],
        message_module=IdentityMessage(msg_dim, hp["memory_dim"], hp["time_dim"]),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=hp["memory_dim"],
        out_channels=hp["embedding_dim"],
        msg_dim=msg_dim,
        time_enc=memory.time_enc,
        dropout=hp["dropout"],
    ).to(device)

    risk_head = RiskHead(in_channels=hp["embedding_dim"]).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(risk_head.parameters()), lr=hp["learning_rate"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=hp["min_lr"]
    )

    neighbor_loader = LastNeighborLoader(num_nodes, size=hp["neighbor_size"], device=device)
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def forward_batch(batch):
        src, dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        pred = risk_head(z[assoc[src]])

        memory.update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)
        return pred

    def train_epoch() -> float:
        memory.train()
        gnn.train()
        risk_head.train()
        memory.reset_state()
        neighbor_loader.reset_state()

        total_loss, total_events = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = forward_batch(batch)

            mask = batch.y_mask
            if mask.any():
                loss = F.mse_loss(pred[mask], batch.y[mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * int(mask.sum())
            memory.detach()
            total_events += int(mask.sum())

        return total_loss / max(total_events, 1)

    @torch.no_grad()
    def evaluate(loader) -> tuple[float, float]:
        memory.eval()
        gnn.eval()
        risk_head.eval()

        preds, targets = [], []
        for batch in loader:
            batch = batch.to(device)
            pred = forward_batch(batch)
            mask = batch.y_mask
            if mask.any():
                preds.append(pred[mask].cpu())
                targets.append(batch.y[mask].cpu())

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        mse = float(F.mse_loss(preds, targets))
        rho = float(spearmanr(preds.numpy(), targets.numpy()).correlation)
        return mse, rho

    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(mlflow_experiment) is None:
        mlflow.create_experiment(
            mlflow_experiment, artifact_location=(MLFLOW_TRACKING_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(mlflow_experiment)

    def save_checkpoint() -> None:
        torch.save(
            {
                "memory": memory.state_dict(),
                "gnn": gnn.state_dict(),
                "risk_head": risk_head.state_dict(),
                "node_index": checkpoint["node_index"],
                "neighbor_loader": {
                    "neighbors": neighbor_loader.neighbors,
                    "e_id": neighbor_loader.e_id,
                    "cur_e_id": neighbor_loader.cur_e_id,
                },
            },
            save_path,
        )

    val_mse, val_rho = float("nan"), float("nan")
    best_val_rho, best_val_mse = float("-inf"), float("nan")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "label_mode": graph_mode,
                "num_epochs": epochs,
                "loss": "unweighted_mse",
                "num_nodes": num_nodes,
                "train_events": train_data.src.size(0),
                "val_events": val_data.src.size(0),
                **hp,
            }
        )

        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            train_loss = train_epoch()
            val_mse, val_rho = evaluate(val_loader)
            scheduler.step()
            print(
                f"epoch {epoch:02d} lr={current_lr:.6f} train_mse={train_loss:.6f}  "
                f"val_mse={val_mse:.6f}  val_spearman={val_rho:.6f}"
            )
            mlflow.log_metric("lr", current_lr, step=epoch)
            mlflow.log_metric("train_mse", train_loss, step=epoch)
            mlflow.log_metric("val_mse", val_mse, step=epoch)
            mlflow.log_metric("val_spearman", val_rho, step=epoch)

            # Checkpoint on improvement, not just at the end - a later epoch that overfits
            # must not silently clobber the best weights seen so far.
            if val_rho > best_val_rho:
                best_val_rho, best_val_mse = val_rho, val_mse
                save_checkpoint()
                print(f"  -> new best val_spearman={val_rho:.6f}, saved checkpoint to: {save_path}")

            if trial is not None:
                import optuna

                trial.report(val_rho, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mlflow.log_artifact(str(save_path))

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM allocated: {peak_gb:.2f} GB")

    return {"val_spearman": best_val_rho, "val_mse": best_val_mse, "checkpoint_path": str(save_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", choices=["daily", "stream"], default="stream")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    run_training(graph_mode=args.graph, epochs=args.epochs)


if __name__ == "__main__":
    main()