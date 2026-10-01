"""
Step 9 - Outlier-removal re-run: drop UHPC-class papers and re-measure the
framing A/B/C comparison + the tuned "best model", BEFORE vs AFTER.

Cutoff: papers whose MAXIMUM reported compressive strength exceeds
CUTOFF_MPA = 120 MPa are dropped in full (every row, every replacement level
and age) -- not a per-row clip. Rationale: exactly two of the 44 papers ever
cross 120 MPa, and both are qualitatively distinct from the rest of the corpus
(next-highest paper tops out at 100.0 MPa):

  paper 17 - titled "...to produce ultra high performance concrete", w/b =
             0.15-0.23 (UHPC-grade vs ~0.35-0.5 for the rest of the corpus),
             climbs smoothly 51 -> 205.5 MPa with age -- internally consistent
             UHPC behaviour, just a different mix-design regime (silica fume /
             superplasticizer / very low w/b) than ordinary RHA concrete.
  paper 36 - reports 125-135 MPa AT 1 DAY (including for its own 0% control),
             not physically plausible for normally-cured concrete; every row
             is flagged is_range=True (graph-read, not a table value) --
             most likely a mis-read axis/unit rather than genuine UHPC.

Both were already implicated in the original report's own diagnosis that
framing A mixes "mortars, normal concrete and UHPC (1-205 MPa)" on one scale.

IMPORTANT ENVIRONMENT NOTE: on this machine, xgboost.dll is currently blocked
by a Windows Application Control policy (confirmed: `from xgboost import
XGBRegressor` fails with XGBoostError / WinError 4551, independent of shell).
The original out/cv_results.txt and out/best_model.txt were produced with
real XGBoost and are NOT reusable as the "before" side of a fair comparison.
So this script re-runs BOTH the unfiltered (44-paper) and filtered (42-paper)
data through the *same* fallback backend (sklearn HistGradientBoostingRegressor,
with monotonic_cst standing in for XGBoost's monotone_constraints) so the only
thing that differs between "before" and "after" is the outlier removal, not
the model. These numbers are therefore internally comparable to each other,
but not directly comparable in absolute terms to the xgboost-based numbers
already in the report.

Outputs: pipeline/out_no_outliers/comparison.txt (+ copies of the filtered
train_features.csv for provenance). Does not touch pipeline/out/.

Run: python pipeline/09_run_no_outliers.py
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SRC = HERE / "out"
DST = HERE / "out_no_outliers"
DST.mkdir(exist_ok=True)

CUTOFF_MPA = 120.0
N_REPEATS = 40
RNG = 42
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
MONO = {"replacement_pct": 0, "age_days": 1, "ctrl_strength": 1}
CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o", "loi",
        "amorphous_pct", "specific_gravity", "mean_particle_um",
        "specific_surface", "bet_surface", "blaine", "wb_ratio",
        "burn_temp_c", "burn_duration", "grind_duration", "pozzolanic_reactivity"]
NOMINAL = ["activation_method", "mineralogy", "processing_method"]

lines = []


def log(s=""):
    print(s)
    lines.append(s)


def xgb_available():
    try:
        from xgboost import XGBRegressor  # noqa
        return True
    except Exception:
        return False


BACKEND = "xgboost" if xgb_available() else "histgb (xgboost.dll blocked by Application Control policy on this machine)"


def encode(df, num_cols):
    X = df[[c for c in num_cols if c in df.columns]].copy()
    for c in NOMINAL:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype("object"), prefix=c, dummy_na=True)
            X = pd.concat([X, d.astype(float)], axis=1)
    return X


def make_model(params, mono_cols=None):
    if mono_cols is None:
        return HistGradientBoostingRegressor(random_state=RNG, **params)
    cst = {c: MONO[c] for c in mono_cols}
    return HistGradientBoostingRegressor(random_state=RNG, monotonic_cst=cst, **params)


def group_cv(df, num_cols, target, label, params, mono_cols=None):
    X = encode(df, num_cols)
    y = df[target].values
    groups = df["paper_id"].values
    r2s, rmses, maes = [], [], []
    for seed in range(N_REPEATS):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                      .split(X, y, groups))
        m = make_model(params, mono_cols)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
    r2, rmse, mae = float(np.mean(r2s)), float(np.mean(rmses)), float(np.mean(maes))
    log(f"[{label}]  n={len(y)} papers={df.paper_id.nunique()} repeats={len(r2s)}")
    log(f"    R2={r2:.3f}+/-{np.std(r2s):.3f}  RMSE={rmse:.2f}+/-{np.std(rmses):.2f}"
        f"  MAE={mae:.2f}")
    return dict(r2=r2, r2_std=float(np.std(r2s)), rmse=rmse, mae=mae)


def add_control(df):
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"])["strength_mpa"].mean().rename("ctrl_strength"))
    return df.merge(ctrl, on=["paper_id", "age_days"], how="inner")


def tune_best(d):
    grid = dict(max_depth=[2, 3, 4], learning_rate=[0.03, 0.05, 0.1],
                max_iter=[100, 200], min_samples_leaf=[5, 10], l2_regularization=[0.0, 1.0])
    best, best_r2, best_metrics = None, -np.inf, None
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), combo))
        X, y, g = d[BASE], d["strength_mpa"].values, d["paper_id"].values
        r2s, rmses, maes = [], [], []
        for s in range(N_REPEATS):
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(X, y, g))
            m = make_model(params, BASE)
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            r2s.append(r2_score(y[te], p))
            rmses.append(np.sqrt(mean_squared_error(y[te], p)))
            maes.append(mean_absolute_error(y[te], p))
        r2 = float(np.mean(r2s))
        if r2 > best_r2:
            best, best_r2 = params, r2
            best_metrics = dict(r2=r2, r2_std=float(np.std(r2s)),
                                 rmse=float(np.mean(rmses)), mae=float(np.mean(maes)))
    return best, best_metrics


def optimal_argmax(d, params):
    X, y, g = d[BASE], d["strength_mpa"].values, d["paper_id"].values
    grid = np.arange(0, 41, 2.5)
    gkf = GroupKFold(min(5, d.paper_id.nunique()))
    preds = {}
    for tr, te in gkf.split(X, y, g):
        m = make_model(params, BASE)
        m.fit(X.iloc[tr], y[tr])
        for pid in np.unique(g[te]):
            rows = d[d.paper_id == pid]
            c28 = rows[rows.age_days == 28]["ctrl_strength"]
            if c28.empty:
                continue
            cand = pd.DataFrame({"replacement_pct": grid, "age_days": 28.0,
                                 "ctrl_strength": float(c28.iloc[0])})[BASE]
            preds[pid] = float(grid[int(np.argmax(m.predict(cand)))])
    paper = pd.read_csv(SRC / "paper_level.csv").set_index("paper_id")
    ps = pd.Series(preds)
    rep = paper["reported_optimum_pct"].reindex(ps.index)
    mask = rep.notna()
    mae = mean_absolute_error(rep[mask], ps[mask])
    bmae = mean_absolute_error(rep[mask], np.full(mask.sum(), rep[mask].mean()))
    log(f"[Optimal% via CS-argmax]  papers={mask.sum()}  MAE={mae:.2f} pts "
        f"(mean-baseline {bmae:.2f})")
    return mae


def run_side(df, tag):
    log("\n" + "=" * 70)
    log(f"{tag}: rows={len(df)} papers={df.paper_id.nunique()} "
        f"strength range {df.strength_mpa.min():.1f}-{df.strength_mpa.max():.1f} MPa")
    log("=" * 70)

    log("\n-- Framing A: absolute CS, all papers, all features --")
    resA = group_cv(df, ["replacement_pct", "age_days"] + CHEM, "strength_mpa",
                     "A", params=dict(max_depth=3, learning_rate=0.05, max_iter=300))

    d = add_control(df)
    dpos = d[d.replacement_pct > 0].copy()
    log(f"\n(papers with a 0% control: {d.paper_id.nunique()}; "
        f"RHA-blend rows for framing C: {len(dpos)})")

    log("\n-- Framing C: tuned best model [replacement, age, control], monotone --")
    best_params, resC = tune_best(dpos)
    log(f"    best HistGB params: {best_params}")
    log(f"[C-tuned]  R2={resC['r2']:.3f}+/-{resC['r2_std']:.3f}  "
        f"RMSE={resC['rmse']:.2f}  MAE={resC['mae']:.2f}")

    opt_mae = optimal_argmax(dpos, best_params)

    return dict(rowsA=len(df), papersA=df.paper_id.nunique(), resA=resA,
                rowsC=len(dpos), papersC=dpos.paper_id.nunique(), resC=resC,
                opt_mae=opt_mae)


def main():
    log(f"Backend for this comparison: {BACKEND}")
    tl = pd.read_csv(SRC / "train_long.csv")
    tf = pd.read_csv(SRC / "train_features.csv")

    paper_max = tl.groupby("paper_id")["strength_mpa"].max()
    drop_ids = sorted(paper_max[paper_max > CUTOFF_MPA].index.tolist())
    keep_max = paper_max.drop(index=drop_ids).max()

    log(f"\nCutoff: drop papers with max reported strength > {CUTOFF_MPA:.0f} MPa (whole paper, all rows).")
    log(f"Papers dropped: {drop_ids}  max strength = "
        f"{ {int(p): float(paper_max[p]) for p in drop_ids} }")
    log(f"Highest max-strength among the papers KEPT: {keep_max:.1f} MPa")

    tf_before = tf.copy()
    tf_after = tf[~tf.paper_id.isin(drop_ids)].copy()
    tf_after.to_csv(DST / "train_features.csv", index=False)

    before = run_side(tf_before, "BEFORE (all 44 papers, unfiltered)")
    after = run_side(tf_after, "AFTER (outlier papers removed)")

    log("\n" + "=" * 70)
    log("SUMMARY  (same HistGB backend on both sides -> isolates the outlier effect)")
    log("=" * 70)
    log(f"{'':32s}{'BEFORE':>16s}{'AFTER':>16s}")
    a_before = f"{before['rowsA']}/{before['papersA']}"
    a_after = f"{after['rowsA']}/{after['papersA']}"
    c_before = f"{before['rowsC']}/{before['papersC']}"
    c_after = f"{after['rowsC']}/{after['papersC']}"
    log(f"{'rows / papers (framing A)':32s}{a_before:>16s}{a_after:>16s}")
    log(f"{'Framing A  R2':32s}{before['resA']['r2']:16.3f}{after['resA']['r2']:16.3f}")
    log(f"{'Framing A  RMSE':32s}{before['resA']['rmse']:16.2f}{after['resA']['rmse']:16.2f}")
    log(f"{'Framing A  MAE':32s}{before['resA']['mae']:16.2f}{after['resA']['mae']:16.2f}")
    log(f"{'rows / papers (framing C)':32s}{c_before:>16s}{c_after:>16s}")
    log(f"{'Framing C  R2 (tuned)':32s}{before['resC']['r2']:16.3f}{after['resC']['r2']:16.3f}")
    log(f"{'Framing C  RMSE (tuned)':32s}{before['resC']['rmse']:16.2f}{after['resC']['rmse']:16.2f}")
    log(f"{'Framing C  MAE (tuned)':32s}{before['resC']['mae']:16.2f}{after['resC']['mae']:16.2f}")
    log(f"{'Optimum-% argmax MAE':32s}{before['opt_mae']:16.2f}{after['opt_mae']:16.2f}")

    (DST / "comparison.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nSaved -> {DST/'comparison.txt'}")


if __name__ == "__main__":
    main()
