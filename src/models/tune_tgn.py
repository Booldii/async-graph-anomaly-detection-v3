"""
Optuna hyperparameter sweep for train_tgn.py's run_training():

- Searches memory_dim/embedding_dim, neighbor_size, learning_rate, batch_size, dropout.
  heads stays fixed at 2 (hardcoded in GraphAttentionEmbedding) - tuning it would need
  embedding_dim % heads == 0 validation coupled to the memory_dim choices below, for a
  parameter that isn't among the highest-leverage ones here.
- memory_dim/neighbor_size bounds are set from this project's own measured VRAM scaling on its
  dev GPU (8GB): ~11.5KB/node.
- Runs under its own MLflow experiment (MLFLOW_EXPERIMENT_NAME_HPO), so the two hand-picked
  reference runs there never get buried under dozens/hundreds of trial runs. Reuses run_training()'s
  own existing MLflow logging (just pointed at a different experiment name).
"""

import argparse
import gc
from pathlib import Path

import optuna
import torch

from train_tgn import PROJECT_ROOT, run_training

MLFLOW_EXPERIMENT_NAME_HPO = "tgn_risk_scoring_hpo"
OPTUNA_STORAGE = f"sqlite:///{PROJECT_ROOT / 'mlruns' / 'optuna_tgn.db'}"
SWEEP_EPOCHS = 5
SWEEP_GRAPH_MODE = "stream"  # the fraction=0.02 graph prepared for this sweep


def objective(trial: optuna.Trial) -> float:
    memory_dim = trial.suggest_categorical("memory_dim", [64, 80, 100, 112, 128])
    neighbor_size = trial.suggest_int("neighbor_size", 5, 15)

    # The only genuinely risky corner at this graph's node count -
    # prune up front, before spending any GPU time on it.
    if memory_dim >= 112 and neighbor_size >= 13:
        raise optuna.TrialPruned()

    hparams = {
        "memory_dim": memory_dim,
        "embedding_dim": memory_dim,  # TGN requires these equal
        "neighbor_size": neighbor_size,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [100, 200, 400]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }

    try:
        result = run_training(
            graph_mode=SWEEP_GRAPH_MODE,
            epochs=SWEEP_EPOCHS,
            hparams=hparams,
            trial=trial,
            mlflow_experiment=MLFLOW_EXPERIMENT_NAME_HPO,
        )
        return result["val_spearman"]
    except optuna.TrialPruned:
        raise
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"trial failed ({type(e).__name__}): {e}")
        raise optuna.TrialPruned()
    finally:
        # gc.collect() before empty_cache(): autograd graphs holding circular references
        # mean CUDA memory can stay attached to Python objects until a GC pass runs -
        # empty_cache() alone doesn't reliably release it, as with the earlier near-OOM run.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--study-name", default="tgn_stream_sweep")
    parser.add_argument("--n-trials", type=int, default=None)
    args = parser.parse_args()

    Path(OPTUNA_STORAGE.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5),
    )
    study.optimize(
        objective,
        timeout=int(args.timeout_hours * 3600),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    print(f"\n{len(study.trials)} trials completed")
    print(f"best val_spearman: {study.best_value:.6f}")
    print(f"best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()