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
"""

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
GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_temporal_graph.pt"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "processed" / "tgn_checkpoint.pt"

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "tgn_risk_scoring"

MEMORY_DIM = 100
TIME_DIM = 100
EMBEDDING_DIM = 100
NEIGHBOR_SIZE = 10
BATCH_SIZE = 200
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(GRAPH_PATH, weights_only=False)
    data = checkpoint["data"].to(device)
    num_nodes = len(checkpoint["node_index"])
    msg_dim = data.msg.size(-1)

    train_data = data[data.train_mask]
    val_data = data[data.val_mask]
    train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
    val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)

    memory = TGNMemory(
        num_nodes,
        msg_dim,
        MEMORY_DIM,
        TIME_DIM,
        message_module=IdentityMessage(msg_dim, MEMORY_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=MEMORY_DIM, out_channels=EMBEDDING_DIM, msg_dim=msg_dim, time_enc=memory.time_enc
    ).to(device)

    risk_head = RiskHead(in_channels=EMBEDDING_DIM).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters()) | set(risk_head.parameters()), lr=LEARNING_RATE
    )

    neighbor_loader = LastNeighborLoader(num_nodes, size=NEIGHBOR_SIZE, device=device)
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
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=(MLFLOW_TRACKING_DIR / "artifacts").as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "memory_dim": MEMORY_DIM,
                "time_dim": TIME_DIM,
                "embedding_dim": EMBEDDING_DIM,
                "neighbor_size": NEIGHBOR_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "loss": "unweighted_mse",
                "num_nodes": num_nodes,
                "train_events": train_data.src.size(0),
                "val_events": val_data.src.size(0),
            }
        )

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch()
            val_mse, val_rho = evaluate(val_loader)
            print(
                f"epoch {epoch:02d}  train_mse={train_loss:.6f}  "
                f"val_mse={val_mse:.6f}  val_spearman={val_rho:.6f}"
            )
            mlflow.log_metric("train_mse", train_loss, step=epoch)
            mlflow.log_metric("val_mse", val_mse, step=epoch)
            mlflow.log_metric("val_spearman", val_rho, step=epoch)

        torch.save(
            {
                "memory": memory.state_dict(),
                "gnn": gnn.state_dict(),
                "risk_head": risk_head.state_dict(),
                "node_index": checkpoint["node_index"],
            },
            CHECKPOINT_PATH,
        )
        mlflow.log_artifact(str(CHECKPOINT_PATH))
        print(f"\nsaved checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()