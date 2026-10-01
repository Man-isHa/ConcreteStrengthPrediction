"""
Step 7 - combine the ablation winners, tune, and produce the final artifacts.

Stage 1 searches the architecture space the ablation opened up: model family x
feature set x whether duplicate mixes are averaged in training x monotone
constraints. Stage 2 tunes the winner's hyper-parameters. Both stages score on
the same repeated grouped hold-out used everywhere else.

Run: python pipeline_rcf/07_best_model.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE, CHEM, FINE, NOMINAL, N_REPEATS, ORDINAL,  # noqa: E402
                    OUT, PROC, RNG, add_control, build_matrix,
                    evaluate_train_variant)

MONO = ["age_days", "ctrl_strength"]
FEATURE_SETS = {
    "base": BASE,
    "base+cat": BASE + NOMINAL + ORDINAL,
    "base+cat+fine": BASE + NOMINAL + ORDINAL + FINE,
    "base+cat+chem": BASE + NOMINAL + ORDINAL + CHEM,
    "all structured": BASE + CHEM + FINE + PROC + ORDINAL + NOMINAL,
}
lines = []


def log(s=""):
    print(s)
    lines.append(s)


def load():
    df = pd.read_csv(OUT / "train_features.csv")
    d = add_control(df[df.age_known].reset_index(drop=True)).copy()
    d["graph_derived"] = d["graph_derived"].astype(float)
    return d


def make_tuned(kind, params, monotone_cols, order):
    """Hyper-parameterised versions of the two gradient-boosted candidates."""
    if kind == "histgb":
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(random_state=RNG, **params)
        if monotone_cols:
            kw["monotonic_cst"] = [1 if c in monotone_cols else 0 for c in order]
        return HistGradientBoostingRegressor(**kw)
    if kind == "xgb":
        from xgboost import XGBRegressor
        kw = dict(random_state=RNG, n_jobs=-1, subsample=0.8,
                  colsample_bytree=0.9, **params)
        if monotone_cols:
            kw["monotone_constraints"] = tuple(
                1 if c in monotone_cols else 0 for c in order)
        return XGBRegressor(**kw)
    raise ValueError(kind)


def repeated_cv(d, feature_cols, kind, params, *, collapse, monotone):
    """Repeated grouped hold-out for one fully-specified configuration."""
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import GroupShuffleSplit

    X = build_matrix(d, feature_cols)
    y = d["strength_mpa"].values
    groups = d["paper_id"].values
    order = list(X.columns)
    mono = MONO if monotone else None

    r2s, rmses, maes = [], [], []
    for seed in range(N_REPEATS):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                      .split(X, y, groups))
        tr_rows = d.iloc[tr]
        if collapse:
            keys = ["paper_id", "replacement_pct", "age_days"]
            num = tr_rows.select_dtypes("number").columns.tolist()
            nom = [c for c in NOMINAL if c in tr_rows.columns]
            agg = tr_rows.groupby(keys, as_index=False)[num].mean()
            if nom:
                first = tr_rows.groupby(keys, as_index=False)[nom].first()
                agg = agg.merge(first, on=keys, how="left")
            tr_rows = agg
        Xtr = build_matrix(tr_rows, feature_cols).reindex(columns=order,
                                                          fill_value=0.0)
        try:
            m = make_tuned(kind, params, mono, order)
            m.fit(Xtr, tr_rows["strength_mpa"].values)
            p = m.predict(X.iloc[te])
        except Exception:
            continue
        r2s.append(r2_score(y[te], p))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
        maes.append(float(mean_absolute_error(y[te], p)))

    if not r2s:
        return None
    return (float(np.mean(r2s)), float(np.std(r2s)),
            float(np.mean(rmses)), float(np.mean(maes)), len(r2s))


DEFAULTS = {"histgb": dict(max_depth=3, learning_rate=0.05, max_iter=400,
                           l2_regularization=1.0),
            "xgb": dict(max_depth=3, learning_rate=0.05, n_estimators=400,
                        min_child_weight=3, reg_lambda=1.0)}

GRIDS = {
    "histgb": dict(max_depth=[2, 3, 4, None], learning_rate=[0.03, 0.05, 0.1],
                   max_iter=[300, 500], l2_regularization=[1.0, 3.0],
                   min_samples_leaf=[10, 20]),
    "xgb": dict(max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
                n_estimators=[300, 500], min_child_weight=[3, 5, 8],
                reg_lambda=[1.0, 3.0]),
}


def stage1(d):
    log("Stage 1 - architecture search "
        "(model x features x duplicate handling x monotone)")
    log(f"{'model':8s} {'features':16s} {'collapse':9s} {'mono':5s} "
        f"{'R2':>16s} {'RMSE':>7s}")
    results = []
    for kind in ("histgb", "xgb"):
        for fname, fcols in FEATURE_SETS.items():
            for collapse in (False, True):
                for monotone in (False, True):
                    res = repeated_cv(d, fcols, kind, DEFAULTS[kind],
                                      collapse=collapse, monotone=monotone)
                    if res is None:
                        continue
                    r2, sd, rmse, mae, n = res
                    results.append(dict(model=kind, features=fname,
                                        collapse=collapse, monotone=monotone,
                                        r2=r2, r2_std=sd, rmse=rmse, mae=mae,
                                        repeats=n))
                    log(f"{kind:8s} {fname:16s} {str(collapse):9s} "
                        f"{str(monotone):5s} {r2:+.3f}+/-{sd:.3f} {rmse:7.2f}")
    res = pd.DataFrame(results).sort_values("r2", ascending=False)
    res.to_csv(OUT / "architecture_search.csv", index=False, encoding="utf-8")
    log()
    log("top 5 architectures:")
    for _, r in res.head(5).iterrows():
        log(f"   {r.model:8s} {r.features:16s} collapse={str(r.collapse):5s} "
            f"mono={str(r.monotone):5s}  R2={r.r2:+.3f}+/-{r.r2_std:.3f}")
    return res.iloc[0]


def stage2(d, best):
    kind = best.model
    fcols = FEATURE_SETS[best.features]
    grid = GRIDS[kind]
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    log()
    log(f"Stage 2 - tuning {kind} over {len(combos)} combinations "
        f"x {N_REPEATS} repeats")

    scored = []
    for values in combos:
        params = dict(zip(keys, values))
        res = repeated_cv(d, fcols, kind, params, collapse=bool(best.collapse),
                          monotone=bool(best.monotone))
        if res is None:
            continue
        scored.append((res[0], res[1], res[2], res[3], params))
    scored.sort(key=lambda t: -t[0])
    log("top 5 hyper-parameter settings:")
    for r2, sd, rmse, mae, p in scored[:5]:
        log(f"   R2={r2:+.3f}+/-{sd:.3f} RMSE={rmse:5.2f} MAE={mae:5.2f}  {p}")
    return scored[0]


def collapse_frame(rows, feature_cols):
    keys = ["paper_id", "replacement_pct", "age_days"]
    num = rows.select_dtypes("number").columns.tolist()
    nom = [c for c in NOMINAL if c in rows.columns]
    agg = rows.groupby(keys, as_index=False)[num].mean()
    if nom:
        agg = agg.merge(rows.groupby(keys, as_index=False)[nom].first(),
                        on=keys, how="left")
    return agg


def final_artifacts(d, best, params):
    """Out-of-fold predictions, importances, and the optimum-level task."""
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    kind, fcols = best.model, FEATURE_SETS[best.features]
    mono = MONO if bool(best.monotone) else None
    X = build_matrix(d, fcols)
    order = list(X.columns)
    y = d["strength_mpa"].values
    groups = d["paper_id"].values
    n_splits = min(5, d.paper_id.nunique())

    oof = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        rows = d.iloc[tr]
        if bool(best.collapse):
            rows = collapse_frame(rows, fcols)
        Xtr = build_matrix(rows, fcols).reindex(columns=order, fill_value=0.0)
        m = make_tuned(kind, params, mono, order)
        m.fit(Xtr, rows["strength_mpa"].values)
        oof[te] = m.predict(X.iloc[te])

    out = d[["paper_id", "replacement_pct", "age_days", "ctrl_strength",
             "strength_mpa"]].copy()
    out["pred"] = oof
    out.to_csv(OUT / "oof_predictions.csv", index=False, encoding="utf-8")
    log()
    log(f"pooled GroupKFold({n_splits}) OOF: R2={r2_score(y, oof):.3f} "
        f"MAE={mean_absolute_error(y, oof):.2f} MPa  "
        f"(optimistic vs the repeated hold-out above)")

    # full-data refit for importances
    rows = collapse_frame(d, fcols) if bool(best.collapse) else d
    Xf = build_matrix(rows, fcols).reindex(columns=order, fill_value=0.0)
    m = make_tuned(kind, params, mono, order)
    m.fit(Xf, rows["strength_mpa"].values)
    if hasattr(m, "feature_importances_"):
        imp = m.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(m, Xf, rows["strength_mpa"].values,
                                     n_repeats=10, random_state=RNG).importances_mean
    (pd.DataFrame({"feature": order, "importance": imp})
       .sort_values("importance", ascending=False)
       .to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8"))
    log()
    log("top feature importances:")
    for f, v in sorted(zip(order, imp), key=lambda t: -t[1])[:6]:
        log(f"   {f:34s} {v:.3f}")

    # ---- optimal replacement level, as the argmax of the predicted 28-d curve
    paper = pd.read_csv(OUT / "paper_level.csv")
    grid = np.arange(0, 51, 2.5)
    preds = []
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        rows = d.iloc[tr]
        if bool(best.collapse):
            rows = collapse_frame(rows, fcols)
        Xtr = build_matrix(rows, fcols).reindex(columns=order, fill_value=0.0)
        m = make_tuned(kind, params, mono, order)
        m.fit(Xtr, rows["strength_mpa"].values)
        for pid in d.iloc[te].paper_id.unique():
            sub = d[d.paper_id == pid]
            proto = sub.iloc[[0]]
            cand = pd.concat([proto] * len(grid), ignore_index=True)
            cand["replacement_pct"] = grid
            cand["age_days"] = 28.0
            cand["ctrl_strength"] = sub.ctrl_strength.median()
            Xc = build_matrix(cand, fcols).reindex(columns=order, fill_value=0.0)
            preds.append(dict(paper_id=pid,
                              pred_optimum=float(grid[np.argmax(m.predict(Xc))])))
    op = pd.DataFrame(preds).merge(
        paper[["paper_id", "reported_optimum_pct", "empirical_optimum_pct"]],
        on="paper_id", how="left")
    op.to_csv(OUT / "optimum_predictions.csv", index=False, encoding="utf-8")

    ok = op.dropna(subset=["reported_optimum_pct"])
    log()
    log(f"optimum task: {len(ok)} held-out papers with a reported optimum")
    if len(ok):
        mae = mean_absolute_error(ok.reported_optimum_pct, ok.pred_optimum)
        base_mae = mean_absolute_error(
            ok.reported_optimum_pct,
            np.full(len(ok), ok.reported_optimum_pct.mean()))
        log(f"   argmax MAE = {mae:.2f} pts vs mean-prior {base_mae:.2f} pts")
        log(f"   predicted optima: {op.pred_optimum.nunique()} distinct values, "
            f"range {op.pred_optimum.min():.1f}-{op.pred_optimum.max():.1f}, "
            f"std {op.pred_optimum.std():.2f}")
        if op.pred_optimum.nunique() == 1:
            log("   WARNING: the argmax predictor is CONSTANT across papers")
        log(f"   reported optima: mean {ok.reported_optimum_pct.mean():.1f}, "
            f"median {ok.reported_optimum_pct.median():.1f}, "
            f"range {ok.reported_optimum_pct.min():.0f}-"
            f"{ok.reported_optimum_pct.max():.0f}, "
            f"std {ok.reported_optimum_pct.std():.2f}")


def main():
    d = load()
    log(f"framing-C modelling set: {len(d)} blend rows over "
        f"{d.paper_id.nunique()} papers")
    log()
    best = stage1(d)
    r2, sd, rmse, mae, params = stage2(d, best)
    log()
    log("FINAL MODEL")
    log(f"   model     : {best.model}")
    log(f"   features  : {best.features} -> {FEATURE_SETS[best.features]}")
    log(f"   collapse  : {bool(best.collapse)} (average duplicate mixes in training)")
    log(f"   monotone  : {bool(best.monotone)} on {MONO}")
    log(f"   params    : {params}")
    log(f"   R2 = {r2:.3f} +/- {sd:.3f}   RMSE = {rmse:.2f}   MAE = {mae:.2f}")

    from sklearn.dummy import DummyRegressor  # noqa: F401
    from common import evaluate
    mb = evaluate(d, BASE, "strength_mpa", "mean")
    log(f"   mean baseline: R2 = {mb['r2_mean']:.3f}  "
        f"RMSE = {mb['rmse_mean']:.2f}  MAE = {mb['mae_mean']:.2f}")

    final_artifacts(d, best, params)
    (OUT / "best_model.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
