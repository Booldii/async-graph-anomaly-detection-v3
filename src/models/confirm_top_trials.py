"""
Re-trains a hand-picked set of Optuna trials (see tune_tgn.py) for a larger epoch budget than
the sweep itself used, to confirm the sweep's 5-epoch ranking still holds further out - the
sweep's pruning only looks at early epochs, so a config that plateaus fast could outrank one
that's still improving past epoch 5.

Trial numbers come from the sweep's own study storage, so the hparams tested here are exactly
what that trial actually used, including the memory_dim == embedding_dim coupling tune_tgn.py's
objective() enforces (trial.params only stores "memory_dim" - embedding_dim is reconstructed from
it below, the same as objective() does.

Logs to the main tgn_risk_scoring MLflow experiment - unlike the sweep's dozens of trials,
a handful of hand-picked confirmation runs are exactly the kind of candidate-for-promotion run.
"""

import argparse

import optuna

from train_tgn import GRAPH_MODES, run_training
from tune_tgn import OPTUNA_STORAGE, SWEEP_GRAPH_MODE


def hparams_from_trial(trial: optuna.trial.FrozenTrial) -> dict:
    p = trial.params
    return {
        "memory_dim": p["memory_dim"],
        "embedding_dim": p["memory_dim"],
        "neighbor_size": p["neighbor_size"],
        "learning_rate": p["learning_rate"],
        "batch_size": p["batch_size"],
        "dropout": p["dropout"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, nargs="+", required=True, help="Optuna trial numbers to re-run")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--study-name", default="tgn_stream_sweep")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=OPTUNA_STORAGE)
    by_number = {t.number: t for t in study.trials}

    results = []
    for n in args.trials:
        if n not in by_number:
            print(f"trial {n}: not found in study '{args.study_name}' - skipping")
            continue
        trial = by_number[n]
        hparams = hparams_from_trial(trial)
        base_checkpoint = GRAPH_MODES[SWEEP_GRAPH_MODE]["checkpoint"]
        checkpoint_path = base_checkpoint.with_stem(f"{base_checkpoint.stem}_trial{n}_final")
        print(f"\n=== trial {n} (sweep value={trial.value}, state={trial.state.name}) - {args.epochs} epochs ===")
        print(f"hparams: {hparams}")

        result = run_training(
            graph_mode=SWEEP_GRAPH_MODE, epochs=args.epochs, hparams=hparams, checkpoint_path=checkpoint_path
        )
        results.append({"trial": n, "hparams": hparams, **result})

    if not results:
        print("no valid trial numbers given")
        return

    results.sort(key=lambda r: r["val_spearman"], reverse=True)
    print(f"\n{'trial':>6}  {'val_spearman':>12}  {'val_mse':>10}  checkpoint")
    for r in results:
        print(f"{r['trial']:>6}  {r['val_spearman']:>12.6f}  {r['val_mse']:>10.6f}  {r['checkpoint_path']}")
    print(f"\nwinner: trial {results[0]['trial']} - hparams: {results[0]['hparams']}")


if __name__ == "__main__":
    main()