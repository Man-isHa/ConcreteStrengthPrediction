"""
Step 13 - Part III ML half: the Part 1 pipeline re-run with every point above
60 MPa removed. Reuses 06_ablation / 07_best_model / 08_report_figs unchanged by
pointing their module-level OUT at pipeline/out_le60/.

  Study 0  like-for-like: same paper splits, train on all points vs <=60 points,
           both scored only on held-out <=60 points (the fair test of whether the
           >60 points were hurting the predictor).
  S1-S5    06_ablation.main() on the <=60 data (same studies/feature sets as Part 1).
  Final    07_best_model.main() on the <=60 data.
  Direct   tuned XGBoost trained/tested on the LLM benchmark's <=60 split.

Run (WSL venv, xgboost required): ~/rha_venv/bin/python pipeline/13_ml_le60.py
"""
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
SRC = HERE / "out"
DST = HERE / "out_le60"
LLM = HERE.parent / "datasets" / "llm_benchmark_rha" / "filtered60"
CUT = 60.0
N_REPEATS = 40
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
PART1_BEST = dict(max_depth=2, learning_rate=0.03, n_estimators=300, min_child_weight=3,
                  reg_lambda=1.0, subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare():
    DST.mkdir(exist_ok=True)
    (DST / "figs").mkdir(exist_ok=True)
    tf = pd.read_csv(SRC / "train_features.csv")
    tl = pd.read_csv(SRC / "train_long.csv")
    tf[tf.strength_mpa <= CUT].to_csv(DST / "train_features.csv", index=False)
    tl[tl.strength_mpa <= CUT].to_csv(DST / "train_long.csv", index=False)
    for f in ["papers_wide.csv", "paper_level.csv"]:
        shutil.copy(SRC / f, DST / f)
    kept = tf[tf.strength_mpa <= CUT]
    print(f"rows {len(tf)} -> {len(kept)}; papers {tf.paper_id.nunique()} -> {kept.paper_id.nunique()}")
    return tf


def add_ctrl(df):
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"])["strength_mpa"]
            .mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return d[d.replacement_pct > 0].copy()


def study0(tf, abl):
    """Train on all vs <=60 points; test only on <=60 points of held-out papers."""
    feats_a = ["replacement_pct", "age_days"] + abl.CHEM + abl.FINE + abl.PROC + abl.ORDINAL
    framings = {
        "A: absolute, all features": (tf, feats_a, dict(n_estimators=400, max_depth=3,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, random_state=42, n_jobs=-1), None),
        "C: blend | control, tuned": (add_ctrl(tf), BASE, PART1_BEST, (0, 1, 1)),
    }
    rows = []
    for name, (d, feats, params, mono) in framings.items():
        d = d.reset_index(drop=True)
        X, y, g = d[feats], d.strength_mpa.values, d.paper_id.values
        onehot = abl.onehot(d) if name.startswith("A") else None
        if onehot is not None:
            X = pd.concat([X, onehot], axis=1)
        res = {"all points": [], "<=60 only": []}
        for s in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(X, y, g))
            te = te[y[te] <= CUT]
            if len(te) < 5:
                continue
            for arm, trn in [("all points", tr), ("<=60 only", tr[y[tr] <= CUT])]:
                kw = dict(params)
                if mono:
                    kw["monotone_constraints"] = mono
                m = XGBRegressor(**kw).fit(X.iloc[trn], y[trn])
                p = m.predict(X.iloc[te])
                res[arm].append((r2_score(y[te], p), np.sqrt(mean_squared_error(y[te], p)),
                                 mean_absolute_error(y[te], p)))
        for arm, v in res.items():
            v = np.array(v)
            rows.append(dict(framing=name, train_on=arm, r2_mean=v[:, 0].mean(), r2_std=v[:, 0].std(),
                             rmse_mean=v[:, 1].mean(), mae_mean=v[:, 2].mean(), repeats=len(v)))
    out = pd.DataFrame(rows)
    out.to_csv(DST / "study0_like_for_like.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def direct_llm_split():
    """Tuned XGBoost on the exact <=60 MPa train/test split the LLM sees."""
    rows = []
    for cond in ["with", "no"]:
        tr = pd.read_csv(LLM / f"train_{cond}.csv")
        te = pd.read_csv(LLM / f"test_{cond}.csv")
        gt = pd.read_csv(LLM / f"gt_{cond}.csv")
        te = te.merge(gt[["test_case_id", "replacement_pct", "age_days", "true_strength_mpa", "strength_band"]],
                      on=["test_case_id", "replacement_pct", "age_days"])
        if cond == "with":
            feats, mono = ["replacement_pct", "age_days", "control_strength_28d_mpa"], (0, 1, 1)
        else:
            num = tr.select_dtypes("number").columns
            feats = ["replacement_pct", "age_days"] + [c for c in num if c not in (
                "replacement_pct", "age_days", "compressive_strength_mpa",
                "replacement_min_pct", "replacement_max_pct")]
            mono = None
        kw = dict(PART1_BEST)
        if mono:
            kw["monotone_constraints"] = mono
        m = XGBRegressor(**kw).fit(tr[feats], tr.compressive_strength_mpa)
        te["pred_mpa"] = m.predict(te[feats])
        te["cond"] = cond
        rows.append(te[["cond", "test_case_id", "paper_id", "replacement_pct", "age_days",
                        "true_strength_mpa", "strength_band", "pred_mpa"]])
        y, p = te.true_strength_mpa, te.pred_mpa
        print(f"direct {cond}: n={len(te)} features={feats} R2={r2_score(y, p):.3f} "
              f"RMSE={np.sqrt(mean_squared_error(y, p)):.2f} MAE={mean_absolute_error(y, p):.2f} "
              f"bias={(p - y).mean():+.2f}")
    pd.concat(rows).to_csv(DST / "direct_llm_split_predictions.csv", index=False)


def main():
    tf = prepare()
    abl = load("abl", HERE / "06_ablation.py")
    abl.OUT = DST

    print("\n=== Study 0: like-for-like ===")
    study0(tf, abl)

    print("\n=== Studies 1-5 (06_ablation on <=60 data) ===")
    abl.main()

    print("\n=== Final tuned model (07_best_model on <=60 data) ===")
    best = load("best", HERE / "07_best_model.py")
    best.OUT = DST
    best.main()

    figs = load("figs", HERE / "08_report_figs.py")
    figs.OUT, figs.FIG = DST, DST / "figs"
    figs.barh_study("S3_features", "Study 3 (≤60 MPa) — feature set (framing C, XGBoost)", "abl_features.png")
    figs.fig_pred_vs_actual()
    figs.fig_residuals()
    figs.fig_importance()

    print("\n=== Direct comparison on the LLM benchmark split ===")
    direct_llm_split()


if __name__ == "__main__":
    main()
