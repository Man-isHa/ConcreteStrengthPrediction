"""
Step 8 - the publication figure set.

Every count that appears in a title or axis label is computed from the pipeline's
own CSVs. The RHA version of this script hard-coded its parse-yield funnel and
its "44 yielding papers" caption, which is exactly how a figure and its prose
drift apart.

Run: python pipeline_rcf/08_report_figs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BLUE, FIG, GREY, LIGHT, OUT, RED, use_plot_style  # noqa: E402

plt = use_plot_style()


# ------------------------------------------------------------------ ablation bars
def barh_study(study, title, fname, fmt="{:.3f}", metric="r2_mean", clip=None):
    """Horizontal bar chart of one ablation study.

    `clip` bounds the axis so that a catastrophically bad configuration (the
    linear and neural baselines reach R2 of -23 and -229) does not compress
    every informative bar into a single pixel. Clipped bars keep their true
    value as a label, marked with an arrow.
    """
    ab = pd.read_csv(OUT / "ablation.csv")
    sub = ab[ab.study == study].dropna(subset=[metric]).sort_values(metric)
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(sub) + 1))
    vals = sub[metric].values
    err = sub["r2_std"].values if metric == "r2_mean" else None

    if clip is not None:
        lo, hi = clip
        drawn = np.clip(vals, lo, hi)
        err = None if err is None else np.clip(err, 0, (hi - lo) / 3)
    else:
        drawn = vals
    ax.barh(range(len(sub)), drawn, xerr=err, color=BLUE, ecolor="gray",
            capsize=3)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["config"], fontsize=8)
    ax.set_xlabel("$R^2$ (mean $\\pm$ std, repeated grouped hold-out)"
                  if metric == "r2_mean" else metric)
    ax.set_title(title)
    if clip is not None:
        ax.set_xlim(clip)
    for i, (v, dv) in enumerate(zip(vals, drawn)):
        off = clip is not None and v != dv
        label = ("  " + ("← " if off else "") + fmt.format(v))
        ax.text(dv, i, label, va="center", fontsize=8,
                ha="left" if dv >= 0 else "left")
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)


# ------------------------------------------------------------------ final model
def fig_pred_vs_actual():
    from sklearn.metrics import mean_absolute_error, r2_score
    oof = pd.read_csv(OUT / "oof_predictions.csv").dropna(subset=["pred"])
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(oof.strength_mpa, oof.pred, c=oof.replacement_pct,
                    cmap="viridis", s=28, alpha=0.8, edgecolor="w", lw=0.3)
    lim = [min(oof.strength_mpa.min(), oof.pred.min()) - 3,
           max(oof.strength_mpa.max(), oof.pred.max()) + 3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("actual strength (MPa)")
    ax.set_ylabel("predicted strength (MPa)")
    ax.set_title(f"Pooled out-of-fold predictions\n"
                 f"$R^2$={r2_score(oof.strength_mpa, oof.pred):.2f}, "
                 f"MAE={mean_absolute_error(oof.strength_mpa, oof.pred):.1f} MPa")
    fig.colorbar(sc, ax=ax, label="replacement %")
    fig.tight_layout(); fig.savefig(FIG / "pred_vs_actual.png"); plt.close(fig)


def fig_residuals():
    oof = pd.read_csv(OUT / "oof_predictions.csv").dropna(subset=["pred"])
    res = oof.pred - oof.strength_mpa
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    a1.hist(res, bins=25, color=BLUE, alpha=0.85)
    a1.axvline(0, color="k", lw=1)
    a1.set_xlabel("residual (predicted - actual, MPa)")
    a1.set_ylabel("count")
    a1.set_title(f"mean {res.mean():+.1f}, median {res.median():+.1f} MPa")
    a2.scatter(oof.pred, res, s=20, alpha=0.6, color=BLUE)
    a2.axhline(0, color="k", lw=1)
    a2.set_xlabel("predicted strength (MPa)")
    a2.set_ylabel("residual (MPa)")
    a2.set_title("residual vs prediction")
    fig.tight_layout(); fig.savefig(FIG / "residuals.png"); plt.close(fig)


def fig_importance():
    imp = pd.read_csv(OUT / "feature_importance.csv").head(12)
    imp = imp.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(imp) + 1.2))
    ax.barh(range(len(imp)), imp.importance, color=BLUE)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp.feature, fontsize=8)
    ax.set_xlabel("gain importance")
    ax.set_title("Final model feature importance")
    fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)


def fig_optimum():
    """Two panels, because the target itself is ambiguous.

    The right panel is the important one: a paper's *reported* optimum and the
    argmax of that same paper's own 28-day table are two different numbers, and
    they disagree by more than the model disagrees with either.
    """
    from sklearn.metrics import mean_absolute_error
    op = pd.read_csv(OUT / "optimum_predictions.csv")
    rep = op.dropna(subset=["reported_optimum_pct"])
    if rep.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.6))

    a1.scatter(rep.reported_optimum_pct, rep.pred_optimum, s=40, alpha=0.8,
               color=BLUE, edgecolor="w")
    lim = [-1, max(rep.reported_optimum_pct.max(), rep.pred_optimum.max()) + 3]
    a1.plot(lim, lim, "k--", lw=1)
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("reported optimum (%)")
    a1.set_ylabel("model argmax optimum (%)")
    mae = mean_absolute_error(rep.reported_optimum_pct, rep.pred_optimum)
    a1.set_title(f"Model vs the reported optimum\n"
                 f"MAE = {mae:.1f} pts ({len(rep)} held-out papers)")

    both = op.dropna(subset=["reported_optimum_pct", "empirical_optimum_pct"])
    a2.scatter(both.reported_optimum_pct, both.empirical_optimum_pct, s=40,
               alpha=0.8, color=RED, edgecolor="w")
    lim2 = [-1, max(both.reported_optimum_pct.max(),
                    both.empirical_optimum_pct.max()) + 3]
    a2.plot(lim2, lim2, "k--", lw=1)
    a2.set_xlim(lim2); a2.set_ylim(lim2)
    a2.set_xlabel("reported optimum (%)")
    a2.set_ylabel("argmax of the paper's own 28-day data (%)")
    mae2 = mean_absolute_error(both.reported_optimum_pct,
                               both.empirical_optimum_pct)
    a2.set_title(f"The two labels disagree\n"
                 f"MAE = {mae2:.1f} pts, r = "
                 f"{both.reported_optimum_pct.corr(both.empirical_optimum_pct):.2f} "
                 f"({len(both)} papers)")
    fig.tight_layout(); fig.savefig(FIG / "optimum_scatter.png"); plt.close(fig)


# ------------------------------------------------------------------ data figures
def fig_cs_curves():
    """Physics checks that survive the paper-level confound.

    A raw cross-paper average of strength against age would track mix
    composition, not age: the parsed corpus spans pastes at 0.6 MPa to UHPC at
    184 MPa, and 0.6-108 MPa among the age-resolved rows used here. Both panels
    are therefore computed *within* a paper.
    """
    long = pd.read_csv(OUT / "train_features.csv")
    aged = long[long.age_known]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    recs = []
    for _, g in aged.groupby(["paper_id", "replacement_pct"]):
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
    ages = n[n >= 5].index
    a1.fill_between(ages, q1.loc[ages], q3.loc[ages], color=BLUE, alpha=0.20,
                    label="IQR across series")
    a1.plot(ages, med.loc[ages], "o-", color=BLUE, label="median")
    a1.axhline(1.0, color="k", lw=0.8, ls=":")
    a1.set_xscale("log")
    a1.set_xlabel("curing age (days, log)")
    a1.set_ylabel("strength / same series' 28-day strength")
    a1.set_title(f"Within-series age trend ({len(rel)} points)")
    a1.legend(fontsize=8)

    d28 = aged[aged.age_days == 28]
    interior = low = high = 0
    for pid, g in d28.groupby("paper_id"):
        s = (g.groupby("replacement_pct", as_index=False)["strength_mpa"].mean()
             .sort_values("replacement_pct"))
        if len(s) < 3:
            continue
        arg = s.replacement_pct.iloc[int(np.argmax(s.strength_mpa.values))]
        if arg == s.replacement_pct.iloc[0]:
            low += 1
        elif arg == s.replacement_pct.iloc[-1]:
            high += 1
        else:
            interior += 1
    tot = interior + low + high
    counts = [interior, low, high]
    labels = ["interior\noptimum", "peak at\nlowest level", "peak at\nhighest level"]
    a2.bar(range(3), counts, color=[BLUE, RED, RED])
    a2.set_xticks(range(3)); a2.set_xticklabels(labels, fontsize=8)
    a2.set_ylabel("papers")
    a2.set_ylim(0, max(counts) * 1.25 if tot else 1)
    for i, c in enumerate(counts):
        a2.text(i, c, f"\n{c}  ({c/tot*100:.0f}%)" if tot else "", ha="center",
                va="bottom", fontsize=8)
    a2.set_title(f"28-day curve shape ({tot} papers with $\\geq$3 levels)")
    fig.tight_layout(); fig.savefig(FIG / "cs_curves.png"); plt.close(fig)


def fig_sparsity():
    feats = pd.read_csv(OUT / "paper_level.csv")
    long = pd.read_csv(OUT / "train_features.csv")
    yielding = feats[feats.paper_id.isin(long.paper_id.unique())]
    cols = [c for c in feats.columns
            if c not in ("paper_id", "reported_optimum_pct",
                         "empirical_optimum_pct")]
    fill = (yielding[cols].notna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(fill)), fill.values, color=BLUE)
    ax.set_yticks(range(len(fill)))
    ax.set_yticklabels(fill.index, fontsize=7)
    ax.set_xlabel("% of yielding papers with a value")
    ax.set_title(f"Feature availability ({len(yielding)} yielding papers)")
    ax.set_xlim(0, 100)
    fig.tight_layout(); fig.savefig(FIG / "data_availability.png"); plt.close(fig)


def fig_parse_yield():
    y = pd.read_csv(OUT / "parse_yield.csv")
    long = pd.read_csv(OUT / "train_features.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    import textwrap
    labels = y.stage.tolist()
    vals = y.n.tolist()
    colors = [GREY, GREY, LIGHT, BLUE, BLUE][:len(vals)]
    a1.bar(range(len(vals)), vals, color=colors)
    a1.set_xticks(range(len(vals)))
    a1.set_xticklabels(["\n".join(textwrap.wrap(l, 14)) for l in labels],
                       fontsize=7)
    a1.set_ylabel("papers")
    a1.set_title("Parse funnel")
    for i, v in enumerate(vals):
        a1.text(i, v, f"\n{v}", ha="center", va="bottom", fontsize=8)

    counts = long.replacement_pct.value_counts().sort_index()
    a2.bar(counts.index, counts.values, width=1.6, color=BLUE)
    a2.set_xlabel("cement replacement (%)")
    a2.set_ylabel("recovered data points")
    a2.set_title(f"{len(long)} points across "
                 f"{long.paper_id.nunique()} papers")
    fig.tight_layout(); fig.savefig(FIG / "parse_yield.png"); plt.close(fig)


def fig_sources():
    """Where the recovered points come from, and how many resolve a curing age."""
    long = pd.read_csv(OUT / "train_features.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    src = long.source.value_counts()
    a1.bar(range(len(src)), src.values, color=[BLUE, LIGHT][:len(src)])
    a1.set_xticks(range(len(src)))
    a1.set_xticklabels(["Sheet1 grids" if s == "sheet1" else "Experimental Matrix"
                        for s in src.index], fontsize=8)
    a1.set_ylabel("data points")
    a1.set_title("Recovered points by source")
    for i, v in enumerate(src.values):
        a1.text(i, v, f"\n{v}", ha="center", va="bottom", fontsize=8)

    tab = long.groupby(["source", "age_known"]).size().unstack(fill_value=0)
    known = tab.get(True, pd.Series(0, index=tab.index))
    unknown = tab.get(False, pd.Series(0, index=tab.index))
    idx = range(len(tab))
    a2.bar(idx, known.values, color=BLUE, label="age resolved")
    a2.bar(idx, unknown.values, bottom=known.values, color=GREY,
           label="age unknown")
    a2.set_xticks(list(idx))
    a2.set_xticklabels(["Sheet1" if s == "sheet1" else "Matrix"
                        for s in tab.index], fontsize=8)
    a2.set_ylabel("data points")
    a2.set_title("Curing age resolution")
    a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "sources.png"); plt.close(fig)


def main():
    barh_study("S1_framing", "Study 1 - target framing", "abl_framing.png",
               clip=(-0.5, 0.55))
    barh_study("S2_model", "Study 2 - model (framing C, all features)",
               "abl_model.png", clip=(-1.0, 0.55))
    barh_study("S3_features", "Study 3 - feature set (framing C, XGBoost)",
               "abl_features.png")
    barh_study("S5_priors", "Study 5 - physics priors (framing C, base features)",
               "abl_priors.png")
    barh_study("S6_data", "Study 6 - training data (identical held-out rows)",
               "abl_data.png")
    barh_study("S4_imputation", "Study 4 - imputation (RMSE, lower = better)",
               "abl_imputation.png", fmt="{:.2f}", metric="rmse_mean",
               clip=(0, 20))
    fig_pred_vs_actual()
    fig_residuals()
    fig_importance()
    fig_optimum()
    fig_cs_curves()
    fig_sparsity()
    fig_parse_yield()
    fig_sources()
    print(f"wrote {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    main()
