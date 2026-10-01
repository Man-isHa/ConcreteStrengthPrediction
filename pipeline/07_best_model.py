"""
Step 7 - Final tuned model (best config from the ablation) + clean OOF outputs
for the report.

Winning recipe (see out/ablation.csv):
  framing C (predict RHA-blend strength given the paper's control strength),
  features = [replacement_pct, age_days, ctrl_strength], XGBoost with
  monotone constraints (strength non-decreasing in age & control strength).

Outputs:
  out/best_model.txt        final metrics + chosen hyper-parameters
  out/oof_predictions.csv    5-fold GroupKFold out-of-fold predictions (for plots)
  out/feature_importance.csv importances of the final model
  out/optimum_predictions.csv predicted vs reported optimum % per paper

Run: python pipeline/07_best_model.py
"""
from pathlib import Path
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
N_REPEATS = 40
RNG = 42
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
MONO = {"replacement_pct": 0, "age_days": 1, "ctrl_strength": 1}


def load_blend():
    df = pd.read_csv(OUT / "train_features.csv")
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"])["strength_mpa"].mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return df, d[d.replacement_pct > 0].copy()


def make_xgb(params):
    from xgboost import XGBRegressor
    return XGBRegressor(
        monotone_constraints=tuple(MONO[c] for c in BASE),
        random_state=RNG, n_jobs=-1, subsample=0.8, colsample_bytree=0.9, **params)


def repeated_cv(d, params):
    X, y, g = d[BASE], d["strength_mpa"].values, d["paper_id"].values
    r2s, rmses, maes = [], [], []
    for s in range(N_REPEATS):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(X, y, g))
        m = make_xgb(params)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
    return np.mean(r2s), np.std(r2s), np.mean(rmses), np.mean(maes)


def tune(d):
    grid = dict(max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
                n_estimators=[300, 500], min_child_weight=[3, 5, 8], reg_lambda=[1.0, 3.0])
    best, best_r2 = None, -np.inf
    log = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        r2, sd, rmse, mae = repeated_cv(d, params)
        log.append((params, r2, sd, rmse, mae))
        if r2 > best_r2:
            best, best_r2 = params, r2
    log.sort(key=lambda x: -x[1])
    return best, log


def oof_predictions(d, params):
    X, y, g = d[BASE], d["strength_mpa"].values, d["paper_id"].values
    gkf = GroupKFold(min(5, d.paper_id.nunique()))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, g):
        m = make_xgb(params)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    out = d[["paper_id", "replacement_pct", "age_days", "ctrl_strength", "strength_mpa"]].copy()
    out["pred"] = oof
    return out


def optimum(d, params):
    """Predicted optimum % = argmax of best model at 28 d, held-out per paper."""
    paper = pd.read_csv(OUT / "paper_level.csv").set_index("paper_id")
    X, y, g = d[BASE], d["strength_mpa"].values, d["paper_id"].values
    grid = np.arange(0, 41, 2.5)
    gkf = GroupKFold(min(5, d.paper_id.nunique()))
    preds = {}
    for tr, te in gkf.split(X, y, g):
        m = make_xgb(params)
        m.fit(X.iloc[tr], y[tr])
        for pid in np.unique(g[te]):
            rows = d[d.paper_id == pid]
            c28 = rows[rows.age_days == 28]["ctrl_strength"]
            if c28.empty:
                continue
            cand = pd.DataFrame({"replacement_pct": grid, "age_days": 28.0,
                                 "ctrl_strength": float(c28.iloc[0])})[BASE]
            preds[pid] = float(grid[int(np.argmax(m.predict(cand)))])
    ps = pd.Series(preds, name="pred_optimum")
    res = pd.DataFrame({"pred_optimum": ps})
    res["reported_optimum"] = paper["reported_optimum_pct"].reindex(res.index)
    res["empirical_optimum"] = paper["empirical_optimum_pct"].reindex(res.index)
    return res.reset_index().rename(columns={"index": "paper_id"})


def main():
    _, d = load_blend()
    lines = [f"Blend rows: {len(d)}  papers: {d.paper_id.nunique()}",
             f"Features: {BASE}  monotone: {MONO}"]

    best, log = tune(d)
    r2, sd, rmse, mae = repeated_cv(d, best)
    lines += [f"\nBest hyper-parameters: {best}",
              f"Repeated CV ({N_REPEATS}x, 30% papers held out):",
              f"  R2 = {r2:.3f} +/- {sd:.3f}",
              f"  RMSE = {rmse:.2f} MPa   MAE = {mae:.2f} MPa"]
    lines.append("\nTop-5 hyper-parameter settings:")
    for params, r, s, rm, ma in log[:5]:
        lines.append(f"  R2={r:.3f}+/-{s:.3f} RMSE={rm:.2f} MAE={ma:.2f}  {params}")

    oof = oof_predictions(d, best)
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    lines.append(f"\nGroupKFold OOF: R2={r2_score(oof.strength_mpa, oof.pred):.3f}  "
                 f"RMSE={np.sqrt(mean_squared_error(oof.strength_mpa, oof.pred)):.2f}")

    # feature importance from a full-data fit
    m = make_xgb(best); m.fit(d[BASE], d["strength_mpa"].values)
    fi = pd.DataFrame({"feature": BASE, "importance": m.feature_importances_})
    fi.to_csv(OUT / "feature_importance.csv", index=False)

    opt = optimum(d, best)
    opt.to_csv(OUT / "optimum_predictions.csv", index=False)
    mask = opt.reported_optimum.notna()
    mae_opt = mean_absolute_error(opt.reported_optimum[mask], opt.pred_optimum[mask])
    base_mae = mean_absolute_error(opt.reported_optimum[mask],
                                   np.full(mask.sum(), opt.reported_optimum[mask].mean()))
    lines += [f"\nOptimal replacement (argmax of best model):",
              f"  papers={mask.sum()}  MAE={mae_opt:.2f} pts  (mean-baseline {base_mae:.2f})"]

    txt = "\n".join(lines)
    (OUT / "best_model.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
