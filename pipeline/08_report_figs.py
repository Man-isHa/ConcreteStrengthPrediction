"""
Step 8 - Publication-quality figures for the LaTeX report. -> out/figs/*.png

Reads: ablation.csv, oof_predictions.csv, feature_importance.csv,
       optimum_predictions.csv, train_features.csv, paper_level.csv.

Run: python pipeline/08_report_figs.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)
BLUE = "#2c6fbb"


def barh_study(study, title, fname, fmt="{:.3f}", metric="r2_mean"):
    ab = pd.read_csv(OUT / "ablation.csv")
    sub = ab[ab.study == study].dropna(subset=[metric]).sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(sub) + 1))
    err = sub["r2_std"] if metric == "r2_mean" else None
    ax.barh(range(len(sub)), sub[metric], xerr=err, color=BLUE, ecolor="gray", capsize=3)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub["config"], fontsize=8)
    ax.set_xlabel("R² (mean ± std, repeated grouped hold-out)" if metric == "r2_mean" else metric)
    ax.set_title(title)
    for i, v in enumerate(sub[metric]):
        ax.text(v, i, "  " + fmt.format(v), va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)


def fig_pred_vs_actual():
    oof = pd.read_csv(OUT / "oof_predictions.csv").dropna(subset=["pred"])
    y, p = oof.strength_mpa, oof.pred
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(y, p, c=oof.replacement_pct, cmap="viridis", s=28, alpha=0.8, edgecolor="w", lw=0.3)
    lim = [min(y.min(), p.min()) - 3, max(y.max(), p.max()) + 3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="actual strength (MPa)", ylabel="predicted strength (MPa)",
           title=f"Best model OOF: R²={r2_score(y,p):.2f}, MAE={mean_absolute_error(y,p):.1f} MPa")
    fig.colorbar(sc, label="replacement %")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_actual.png"); plt.close(fig)


def fig_residuals():
    oof = pd.read_csv(OUT / "oof_predictions.csv").dropna(subset=["pred"])
    res = oof.pred - oof.strength_mpa
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.hist(res, bins=25, color=BLUE, alpha=0.85)
    a1.axvline(0, color="k", lw=1); a1.set(xlabel="residual (pred − actual, MPa)", ylabel="count",
                                           title="Residual distribution")
    a2.scatter(oof.pred, res, s=20, alpha=0.6, color=BLUE)
    a2.axhline(0, color="k", lw=1); a2.set(xlabel="predicted (MPa)", ylabel="residual (MPa)",
                                           title="Residuals vs predicted")
    fig.tight_layout(); fig.savefig(FIG / "residuals.png"); plt.close(fig)


def fig_importance():
    fi = pd.read_csv(OUT / "feature_importance.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(fi.feature, fi.importance, color=BLUE)
    ax.set(xlabel="XGBoost gain importance", title="Feature importance (final model)")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)


def fig_optimum():
    op = pd.read_csv(OUT / "optimum_predictions.csv").dropna(subset=["reported_optimum"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(op.reported_optimum, op.pred_optimum, s=40, color=BLUE, alpha=0.8, edgecolor="w")
    lim = [-1, max(op.reported_optimum.max(), op.pred_optimum.max()) + 3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="reported optimum (%)", ylabel="predicted optimum (%)",
           title=f"Optimal replacement: MAE={mean_absolute_error(op.reported_optimum, op.pred_optimum):.1f} pts")
    fig.tight_layout(); fig.savefig(FIG / "optimum_scatter.png"); plt.close(fig)


def fig_cs_curves():
    """Physics checks that survive the paper-level confound.

    Left  : within-paper age trend. A raw cross-paper mean of strength vs age is
            not a usable check here, because the corpus mixes mortars, normal
            concrete and UHPC (1-205 MPa) and the age bins are unevenly
            populated, so the mean moves with mix composition rather than with
            age. Instead each (paper, replacement level) series that contains a
            28-day point is normalised by that point, giving a common anchor;
            we plot the median and IQR across series.
    Right : how often an interior optimum actually exists, over every paper with
            at least three distinct replacement levels at 28 days.
    """
    long = pd.read_csv(OUT / "train_features.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    # ---- left: within-paper age trend, anchored at 28 d ---------------------
    recs = []
    for _, g in long.groupby(["paper_id", "replacement_pct"]):
        ref = g.loc[g.age_days == 28, "strength_mpa"]
        if ref.empty or len(g) < 2:
            continue
        base = ref.mean()
        if base <= 0:
            continue
        for age, s in zip(g.age_days, g.strength_mpa):
            recs.append((age, s / base))
    rel = pd.DataFrame(recs, columns=["age_days", "ratio"])
    grp = rel.groupby("age_days")["ratio"]
    med, n = grp.median(), grp.size()
    q1, q3 = grp.quantile(0.25), grp.quantile(0.75)
    ages = n[n >= 5].index                      # only ages with real support
    a1.fill_between(ages, q1.loc[ages], q3.loc[ages], color=BLUE, alpha=0.20,
                    label="IQR across series")
    a1.plot(ages, med.loc[ages], "o-", color=BLUE, label="median")
    a1.axhline(1.0, color="k", lw=0.8, ls=":")
    a1.set(xscale="log", xlabel="curing age (days)",
           ylabel="strength / same series at 28 d",
           title=f"Within-paper age trend (n={rel.age_days.nunique()} ages)")
    a1.legend(fontsize=8)

    # ---- right: does an interior optimum exist? -----------------------------
    at28 = long[long.age_days == 28]
    agg = (at28.groupby(["paper_id", "replacement_pct"])["strength_mpa"]
           .mean().reset_index())
    counts = {"interior\noptimum": 0, "peak at lowest\nlevel tested": 0,
              "peak at highest\nlevel tested": 0}
    for _, s in agg.groupby("paper_id"):
        s = s.sort_values("replacement_pct")
        if s.replacement_pct.nunique() < 3:
            continue
        v = s.strength_mpa.values
        k = int(np.argmax(v))
        if k == 0:
            counts["peak at lowest\nlevel tested"] += 1
        elif k == len(v) - 1:
            counts["peak at highest\nlevel tested"] += 1
        else:
            counts["interior\noptimum"] += 1
    tot = sum(counts.values())
    labels = list(counts); vals = [counts[k] for k in labels]
    colors = [BLUE, "#c1584b", "#c1584b"]
    a2.bar(range(len(vals)), vals, color=colors)
    a2.set_xticks(range(len(vals))); a2.set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(vals):
        a2.text(i, v + 0.2, f"{v}  ({100*v/tot:.0f}%)", ha="center", fontsize=8)
    a2.set(ylabel="papers", ylim=(0, max(vals) * 1.25),
           title=f"Shape of the 28-day curve ({tot} papers, $\\geq$3 levels)")

    fig.tight_layout(); fig.savefig(FIG / "cs_curves.png"); plt.close(fig)


def fig_sparsity():
    paper = pd.read_csv(OUT / "paper_level.csv")
    long = pd.read_csv(OUT / "train_features.csv")
    yp = paper[paper.paper_id.isin(long.paper_id.unique())]
    feats = [c for c in yp.columns if c not in ("paper_id", "reported_optimum_pct", "empirical_optimum_pct")]
    fill = yp[feats].notna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(fill)), fill.values * 100, color=BLUE)
    ax.set_yticks(range(len(fill))); ax.set_yticklabels(fill.index, fontsize=7)
    ax.set(xlabel="% of yielding papers with a value", title="Feature availability (44 yielding papers)")
    fig.tight_layout(); fig.savefig(FIG / "data_availability.png"); plt.close(fig)


def fig_parse_yield():
    long = pd.read_csv(OUT / "train_long.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(["papers\n(total)", "with CS\ncell", "parsed\n(yielded)"], [79, 70, 44],
           color=["#b0b0b0", "#7aa8d6", BLUE])
    a1.set(ylabel="papers", title="Parse yield: 44/70 CS papers")
    for i, v in enumerate([79, 70, 44]):
        a1.text(i, v + 1, str(v), ha="center")
    vc = long.replacement_pct.value_counts().sort_index()
    a2.bar(vc.index, vc.values, width=1.6, color=BLUE)
    a2.set(xlabel="replacement %", ylabel="data points",
           title=f"{len(long)} CS points by replacement level")
    fig.tight_layout(); fig.savefig(FIG / "parse_yield.png"); plt.close(fig)


def main():
    barh_study("S1_framing", "Study 1 — target framing", "abl_framing.png")
    barh_study("S2_model", "Study 2 — model (framing C, all features)", "abl_model.png")
    barh_study("S3_features", "Study 3 — feature set (framing C, XGBoost)", "abl_features.png")
    barh_study("S5_priors", "Study 5 — physics priors (framing C, base features)", "abl_priors.png")
    barh_study("S4_imputation", "Study 4 — imputation (RMSE, lower=better)",
               "abl_imputation.png", fmt="{:.2f}", metric="rmse_mean")
    fig_pred_vs_actual(); fig_residuals(); fig_importance(); fig_optimum()
    fig_cs_curves(); fig_sparsity(); fig_parse_yield()
    print(f"saved figures to {FIG}")


if __name__ == "__main__":
    main()
