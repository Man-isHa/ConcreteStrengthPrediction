"""
Step 17 - XGBoost per-split scores on the same 40 grouped hold-outs used for gpt-oss in
16_llm_grouped_cv.py (<=60 MPa, framing C), for a paired split-by-split comparison.

  xgb_final : tuned final model of Part III (base features, monotone constraints)
  xgb_all   : all features (06_ablation "all structured" configuration)

Run (WSL venv): ~/rha_venv/bin/python pipeline/17_xgb_grouped_cv_per_split.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_llm_cv"
spec = importlib.util.spec_from_file_location("abl", HERE / "06_ablation.py")
abl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abl)

BASE = ["replacement_pct", "age_days", "ctrl_strength"]
FINAL = dict(max_depth=4, learning_rate=0.03, n_estimators=300, min_child_weight=5, reg_lambda=3.0,
             subsample=0.8, colsample_bytree=0.9, random_state=42, n_jobs=-1,
             monotone_constraints=(0, 1, 1))
ALL = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=3, reg_lambda=1.0, random_state=42, n_jobs=-1)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(HERE / "out_le60" / "train_features.csv")
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"])["strength_mpa"]
            .mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    d = d[d.replacement_pct > 0].reset_index(drop=True)
    num = BASE + abl.CHEM + abl.FINE + abl.PROC + abl.ORDINAL
    X_all = pd.concat([d[num], abl.onehot(d)], axis=1)
    y, g = d.strength_mpa.values, d.paper_id.values
    rows = []
    for s in range(40):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(d, groups=g))
        for name, X, params in [("xgb_final", d[BASE], FINAL), ("xgb_all", X_all, ALL)]:
            p = XGBRegressor(**params).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            rows.append(dict(model=name, split=s, r2=r2_score(y[te], p),
                             rmse=float(np.sqrt(mean_squared_error(y[te], p))),
                             mae=mean_absolute_error(y[te], p),
                             test_papers=",".join(f"P{int(x):03d}" for x in sorted(set(g[te])))))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "xgb_per_split.csv", index=False)
    print(res.groupby("model")[["r2", "rmse", "mae"]].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
