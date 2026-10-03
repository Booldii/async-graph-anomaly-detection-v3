"""
Applies the final production ensemble (trained on the full labeled dataset) to the point-in-time
feature snapshots, producing soft risk labels.

Consensus = unweighted mean of the 4 family models' predicted probabilities. Divergence = standard
deviation across families - the same disagreement-as-uncertainty signal established during
training's diagnostics.

These are an ordinal risk ranking, rather than a calibrated probability: the historical training set's
26% fraud rate doesn't reflect real Ethereum traffic, so P(fraud)=0.7 has no literal
real-world meaning here - only relative ordering between addresses/snapshots is meaningful.
"""

from pathlib import Path

import mlflow
import pandas as pd

from train_xgboost_ensemble import FEATURE_FAMILIES, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_sample_features_snapshots.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "eth_fresh_soft_labels.parquet"


def find_final_model_run() -> str:
    runs = mlflow.search_runs(
        experiment_names=[MLFLOW_EXPERIMENT_NAME],
        filter_string="tags.final_model = 'true'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise RuntimeError(
            f"no run tagged final_model=true found in experiment '{MLFLOW_EXPERIMENT_NAME}' - "
            "run train_xgboost_ensemble.py --full-data first"
        )
    return runs.iloc[0]["run_id"]


def load_family_models(run_id: str) -> dict:
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    logged = mlflow.search_logged_models(experiment_ids=[experiment.experiment_id])
    logged = logged[logged["source_run_id"] == run_id]

    models = {}
    for family in FEATURE_FAMILIES:
        name = f"xgb_family_{family}"
        match = logged[logged["name"] == name]
        if match.empty:
            raise RuntimeError(f"no logged model named '{name}' found for run {run_id}")
        model_id = match.iloc[0]["model_id"]
        models[family] = mlflow.xgboost.load_model(f"models:/{model_id}")
    return models


def score(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    probs = pd.DataFrame(index=df.index)
    for family, feats in FEATURE_FAMILIES.items():
        probs[family] = models[family].predict_proba(df[feats])[:, 1]

    result = probs.copy()
    result["consensus_score"] = probs.mean(axis=1)
    result["divergence"] = probs.std(axis=1)
    result["snapshot_day"] = df["snapshot_day"]
    return result


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR / 'mlflow.db'}")

    run_id = find_final_model_run()
    print(f"using final_model run: {run_id}")
    models = load_family_models(run_id)

    df = pd.read_parquet(SNAPSHOTS_PATH)
    result = score(df, models)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH)

    print(f"\n{len(result):,} (address, day) soft labels")
    print(result["consensus_score"].describe())
    print(f"\nsaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
