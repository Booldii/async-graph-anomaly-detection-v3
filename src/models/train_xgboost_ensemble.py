"""
Trains an ensemble of XGBoost models per feature family and verifies the assumptions behind this approach.

Why a separate model per family (value / gas / behavioral / structural): a single model
relied almost entirely on the value family (85% mean|SHAP|), so the soft label would
effectively just be a rule of "how extreme are the address's values," which the TGN could trivially reproduce.
Each family acts here as an independent "labeling function" - the consensus is a simple unweighted
average of probabilities. Weighting by AUC would hand dominance back to the value family). The divergence between
families serves as a measure of soft label uncertainty.

Without scale_pos_weight: with 26% fraud, the imbalance is mild, and weighting distorts
probabilities (the model acts as if frauds were 50/50), which ruins the averaging.
We check calibration after the fact (Brier and ECE). Exact calibration will be added only if families
differ significantly.

Time-based split: based on first_seen_timestamp - training on the earlier part of the window,
validation on the later part. Limitation: features in build_features.py are calculated
over the entire 14-day window.
"""

import re
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "bybit_address_features.parquet"

MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = "xgboost_family_ensemble"

NON_FEATURE_COLS = {"label", "first_seen_timestamp"}
TRAIN_QUANTILE_CUTOFF = 0.75

FEATURE_FAMILIES: dict[str, list[str]] = {
    "value": [
        "value_sent_zscore_mean", "value_sent_zscore_median", "value_sent_zscore_std",
        "value_sent_zscore_max", "value_received_zscore_mean", "value_received_zscore_median",
        "value_received_zscore_std", "value_received_zscore_max", "value_flow_gap_zscore",
    ],
    "gas": ["gas_price_mean", "gas_used_mean", "gas_fee_mean"],
    "behavioral": [
        "tx_per_active_day", "hour_of_day_entropy", "round_value_share", "mint_count",
        "burn_count", "has_mint_activity", "has_burn_activity",
    ],
    "structural": [
        "out_degree", "in_degree", "total_degree", "tokens_sent", "tokens_received",
        "unique_tokens", "reciprocity_rate", "out_counterparty_entropy", "self_loop_count",
        "neighbor_contract_share",
    ],
}

# acceptance gate thresholds
MAX_FAMILY_LOO_SHARE = 0.5


def mlflow_key(name: str) -> str:
    """MLflow param keys only allow alphanumerics, _-.: / - sanitize anything built from dynamic strings"""
    return re.sub(r"[^a-zA-Z0-9_\-.: /]", "_", name)


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def check_families_cover_features(df: pd.DataFrame) -> None:
    """Each model feature must belong to exactly one family."""
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    assigned = [f for feats in FEATURE_FAMILIES.values() for f in feats]
    missing = set(feature_cols) - set(assigned)
    extra = set(assigned) - set(feature_cols)
    duplicated = {f for f in assigned if assigned.count(f) > 1}
    if missing or extra or duplicated:
        raise ValueError(f"inconsistent feature families: missing={missing}, extra={extra}, duplicates={duplicated}")


def time_based_split(
    df: pd.DataFrame, quantile: float = TRAIN_QUANTILE_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training: addresses first seen before the cutoff point (quantile of
    first_seen_timestamp), validation: after it"""
    cutoff = df["first_seen_timestamp"].quantile(quantile)
    return df[df["first_seen_timestamp"] < cutoff], df[df["first_seen_timestamp"] >= cutoff]


MODEL_PARAMS = {
    "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
    "eval_metric": "aucpr", "random_state": 42,
}


def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)
    return model


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """ECE: the average (weighted by sample size) difference between the predicted and observed
    fraud rate in equal probability bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def score_summary(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    return {
        "AUC-ROC": roc_auc_score(y, p),
        "AUC-PR": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "ECE": expected_calibration_error(y.to_numpy(), p),
        "in_between(0.1-0.9)": float(((p > 0.1) & (p < 0.9)).mean()),
    }


def family_shap_report(model: xgb.XGBClassifier, X: pd.DataFrame, top_n: int = 3) -> pd.Series:
    """Share of mean|SHAP| within the family"""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    share = pd.Series(np.abs(contribs).mean(axis=0), index=X.columns)
    return (share / share.sum()).sort_values(ascending=False).head(top_n)


def main() -> None:
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_TRACKING_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=(MLFLOW_TRACKING_DIR / "artifacts").as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_features()
    check_families_cover_features(df)
    train_df, val_df = time_based_split(df)
    y_train, y_val = train_df["label"], val_df["label"]
    print(
        f"training: {len(train_df)} addresses (fraud {y_train.mean():.1%})\n"
        f"validation: {len(val_df)} addresses (fraud {y_val.mean():.1%})\n"
    )

    with mlflow.start_run():
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("train_quantile_cutoff", TRAIN_QUANTILE_CUTOFF)
        mlflow.log_param("max_family_loo_share", MAX_FAMILY_LOO_SHARE)
        for family, feats in FEATURE_FAMILIES.items():
            mlflow.log_param(f"features_{family}", feats)
        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_val", len(val_df))
        mlflow.log_metric("fraud_rate_train", y_train.mean())
        mlflow.log_metric("fraud_rate_val", y_val.mean())

        models: dict[str, xgb.XGBClassifier] = {}
        probs = pd.DataFrame(index=val_df.index)
        for family, feats in FEATURE_FAMILIES.items():
            models[family] = train_model(train_df[feats], y_train)
            probs[family] = models[family].predict_proba(val_df[feats])[:, 1]
            mlflow.xgboost.log_model(models[family], name=f"xgb_family_{family}")

        all_feats = [f for feats in FEATURE_FAMILIES.values() for f in feats]
        reference = train_model(train_df[all_feats], y_train)
        reference_p = reference.predict_proba(val_df[all_feats])[:, 1]

        consensus = probs.mean(axis=1).to_numpy()
        summary = {f"family:{f}": score_summary(y_val, probs[f].to_numpy()) for f in probs}
        summary["CONSENSUS (average)"] = score_summary(y_val, consensus)
        summary["reference (1 model)"] = score_summary(y_val, reference_p)
        print("[1.] Quality and calibration (validation, without scale_pos_weight)")
        summary_df = pd.DataFrame(summary).T
        print(summary_df.round(4).to_string())
        for row_name, row in summary_df.iterrows():
            metric_prefix = row_name.split(":")[-1].split(" ")[0].lower()
            for metric_name, value in row.items():
                mlflow.log_metric(mlflow_key(f"{metric_prefix}_{metric_name}"), value)

        print("\n[2.] Consensus dependence on families")
        loo = {}
        for family in probs:
            without = probs.drop(columns=family).mean(axis=1).to_numpy()
            loo[family] = {
                "AUC-ROC without family": roc_auc_score(y_val, without),
                "avg|soft label change|": float(np.abs(consensus - without).mean()),
            }
        loo_df = pd.DataFrame(loo).T
        loo_df["change_share"] = loo_df["avg|soft label change|"] / loo_df["avg|soft label change|"].sum()
        print(loo_df.round(4).to_string())
        max_family = loo_df["change_share"].idxmax()
        max_share = float(loo_df["change_share"].max())
        gate_passed = max_share <= MAX_FAMILY_LOO_SHARE
        for family, row in loo_df.iterrows():
            mlflow.log_metric(f"loo_{family}_change_share", row["change_share"])
        mlflow.log_metric("loo_max_change_share", max_share)
        mlflow.log_param("loo_max_family", max_family)
        mlflow.log_metric("gate_passed", float(gate_passed))
        print(
            f"gate: no family > {MAX_FAMILY_LOO_SHARE:.0%} of change share -> "
            f"{'OK' if gate_passed else 'FAILS'} (max: {max_family}={max_share:.1%})"
        )

        print("\n[3.] Family divergence as uncertainty (consensus error at 0.5 threshold)")
        spread = probs.max(axis=1) - probs.min(axis=1)
        wrong = ((consensus >= 0.5).astype(int) != y_val.to_numpy())
        buckets = pd.cut(spread, [-0.001, 0.1, 0.5, 1.0], labels=["agreed(<0.1)", "moderate(0.1-0.5)", "conflicting(>0.5)"])
        print(
            pd.DataFrame({"n": spread.groupby(buckets, observed=False).size(),
                          "consensus_error": pd.Series(wrong, index=spread.index).groupby(buckets, observed=False).mean()})
            .round(4).to_string()
        )

        print("\n[4.] SHAP concentration within family (top-3 features, share of mean|SHAP|)")
        for family, feats in FEATURE_FAMILIES.items():
            top = family_shap_report(models[family], val_df[feats])
            print(f"{family}: " + ", ".join(f"{k}={v:.2f}" for k, v in top.items()))
            for feat_name, share in top.items():
                mlflow.log_metric(f"shap_{family}_{feat_name}", share)

        print(f"\nmodels + metrics logged to MLflow: {MLFLOW_TRACKING_DIR} (experiment: {MLFLOW_EXPERIMENT_NAME})")

        if not gate_passed:
            # hard gate: a silent no-op if this script is ever called from automation
            raise RuntimeError(
                f"acceptance gate failed: family '{max_family}' accounts for "
                f"{max_share:.1%} of consensus sensitivity (> {MAX_FAMILY_LOO_SHARE:.0%} "
                "threshold) - one family dominates the ensemble, defeating its purpose"
            )

if __name__ == "__main__":
    main()