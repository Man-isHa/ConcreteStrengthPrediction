"""Fit and save the two FINAL models exactly as specified in the reports.

No model selection, tuning or experimentation happens here. Both configurations are
frozen copies of what the studies already chose:

  RHA  pipeline/out_le60/best_model.txt      XGBoost, 3 inputs, monotone
  RCF  pipeline_rcf/out_le60/best_model.txt  HistGradientBoosting, 18 inputs, monotone

The research pipelines only ever saved metrics and predictions, never a fitted model,
so this script exists purely to persist the already-chosen models for inference.
Training data is the same <=60 MPa table the reports used.

Run once (RHA needs xgboost; on this machine that means the WSL venv):
    ~/rha_venv/bin/python inference/build_final_models.py
"""
import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- RHA
RHA_FEATURES = ["replacement_pct", "age_days", "ctrl_strength"]
RHA_PARAMS = dict(max_depth=4, learning_rate=0.03, n_estimators=300, min_child_weight=5,
                  reg_lambda=3.0, subsample=0.8, colsample_bytree=0.9,
                  random_state=42, n_jobs=-1)
RHA_MONOTONE = {"replacement_pct": 0, "age_days": 1, "ctrl_strength": 1}
RHA_SCORES = {"r2": 0.346, "r2_sd": 0.249, "rmse": 10.86, "mae": 8.18}

# --------------------------------------------------------------------------- RCF
RCF_NOMINAL = ["activation_method", "processing_method", "mineralogy", "material_family"]
RCF_FEATURES = (["replacement_pct", "age_days", "ctrl_strength"] + RCF_NOMINAL
                + ["pozzolanic_reactivity", "d10", "d50", "d90", "mean_particle_um",
                   "specific_surface", "bet_surface", "blaine", "specific_gravity",
                   "bulk_density", "water_absorption"])
RCF_PARAMS = dict(max_depth=2, learning_rate=0.03, max_iter=300, l2_regularization=3.0,
                  min_samples_leaf=20, random_state=42)
RCF_MONOTONE = ["age_days", "ctrl_strength"]
RCF_SCORES = {"r2": 0.308, "r2_sd": 0.187, "rmse": 11.02, "mae": 8.47}


def add_control(df):
    """Each paper's own 0% control strength at the SAME curing age; blends only."""
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"], as_index=False)["strength_mpa"]
            .mean().rename(columns={"strength_mpa": "ctrl_strength"}))
    merged = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return merged[merged.replacement_pct > 0].reset_index(drop=True)


def onehot(df, cols):
    """One-hot with NaN kept as its own level - mirrors pipeline_rcf/common.py."""
    parts = [pd.get_dummies(df[c], prefix=c, dummy_na=True).astype(float) for c in cols]
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def meta(name, target, features, params, monotone, columns, n_rows, n_papers,
         scores, source, extra=None):
    return {
        "name": name, "created": str(date.today()), "target": target,
        "input_features": features, "model_params": params,
        "monotone_constraints": monotone, "encoded_columns": columns,
        "training_rows": int(n_rows), "training_papers": int(n_papers),
        "reported_performance_grouped_holdout": scores,
        "config_source": source,
        "note": ("Frozen copy of the model selected in the report; refitted once on the "
                 "same <=60 MPa training table purely to persist an artifact."),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, **(extra or {})},
    }


def build_rha():
    from xgboost import XGBRegressor
    import xgboost
    df = pd.read_csv(ROOT / "pipeline" / "out_le60" / "train_features.csv")
    d = add_control(df)
    X, y = d[RHA_FEATURES], d.strength_mpa.values
    model = XGBRegressor(**RHA_PARAMS,
                         monotone_constraints=tuple(RHA_MONOTONE[c] for c in RHA_FEATURES))
    model.fit(X, y)
    joblib.dump(model, MODELS / "rha_strength_model.joblib")
    model.save_model(MODELS / "rha_strength_model.json")   # portable, version-proof
    (MODELS / "rha_strength_model_metadata.json").write_text(json.dumps(
        meta("RHA blended-concrete compressive strength (<=60 MPa)", "strength_mpa",
             RHA_FEATURES, RHA_PARAMS, RHA_MONOTONE, RHA_FEATURES, len(d),
             d.paper_id.nunique(), RHA_SCORES, "pipeline/out_le60/best_model.txt",
             {"xgboost": xgboost.__version__}), indent=2), encoding="utf-8")
    p = model.predict(X)
    print(f"RHA : {len(d)} rows / {d.paper_id.nunique()} papers, {len(RHA_FEATURES)} inputs "
          f"-> in-sample R2 {r2_score(y, p):.3f}, MAE {mean_absolute_error(y, p):.2f} "
          f"(held-out performance from the report: R2 {RHA_SCORES['r2']}, MAE {RHA_SCORES['mae']})")


def build_rcf():
    df = pd.read_csv(ROOT / "pipeline_rcf" / "out_le60" / "train_features.csv")
    d = add_control(df[df.age_known].reset_index(drop=True))
    numeric = [c for c in RCF_FEATURES if c not in RCF_NOMINAL]
    X = pd.concat([d[numeric], onehot(d, RCF_NOMINAL)], axis=1)
    y = d.strength_mpa.values
    model = HistGradientBoostingRegressor(
        **RCF_PARAMS,
        monotonic_cst=[1 if c in RCF_MONOTONE else 0 for c in X.columns])
    model.fit(X, y)
    joblib.dump(model, MODELS / "rcf_strength_model.joblib")
    (MODELS / "rcf_strength_model_metadata.json").write_text(json.dumps(
        meta("RCF blended-concrete compressive strength (<=60 MPa)", "strength_mpa",
             RCF_FEATURES, RCF_PARAMS, RCF_MONOTONE, list(X.columns), len(d),
             d.paper_id.nunique(), RCF_SCORES, "pipeline_rcf/out_le60/best_model.txt",
             {"joblib": joblib.__version__}), indent=2), encoding="utf-8")
    p = model.predict(X)
    print(f"RCF : {len(d)} rows / {d.paper_id.nunique()} papers, {X.shape[1]} encoded columns "
          f"-> in-sample R2 {r2_score(y, p):.3f}, MAE {mean_absolute_error(y, p):.2f} "
          f"(held-out performance from the report: R2 {RCF_SCORES['r2']}, MAE {RCF_SCORES['mae']})")


if __name__ == "__main__":
    build_rcf()
    build_rha()
    print(f"\nsaved to {MODELS}")
