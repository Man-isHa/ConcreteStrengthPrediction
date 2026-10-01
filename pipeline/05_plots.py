"""
Step 5 - Sanity / physics plots. Saves PNGs to out/figs/.

  data_availability.png  parse yield & feature fill-rates
  cs_vs_age.png          strength grows with curing age (physics check)
  cs_vs_replacement.png  example per-paper CS-vs-replacement curves (unimodal)

Run: python pipeline/05_plots.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = OUT / "figs"
FIG.mkdir(exist_ok=True)


def main():
    long = pd.read_csv(OUT / "train_features.csv")

    # 1) CS vs age (28-day should exceed 7-day, etc.)
    fig, ax = plt.subplots(figsize=(6, 4))
    ag = long.groupby("age_days")["strength_mpa"].mean()
    ax.plot(ag.index, ag.values, "o-")
    ax.set(xlabel="curing age (days)", ylabel="mean compressive strength (MPa)",
           title="Strength vs curing age (physics sanity)")
    ax.set_xscale("log")
    fig.tight_layout(); fig.savefig(FIG / "cs_vs_age.png", dpi=120); plt.close(fig)

    # 2) Example CS-vs-replacement curves at 28 days for a few papers
    at28 = long[long.age_days == 28]
    counts = at28.groupby("paper_id").size().sort_values(ascending=False)
    examples = counts[counts >= 4].index[:6]
    fig, ax = plt.subplots(figsize=(6, 4))
    for pid in examples:
        s = at28[at28.paper_id == pid].sort_values("replacement_pct")
        ax.plot(s.replacement_pct, s.strength_mpa, "o-", label=f"paper {pid}")
    ax.set(xlabel="RHA replacement (%)", ylabel="28-day strength (MPa)",
           title="CS vs replacement (28 d) - typically unimodal")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "cs_vs_replacement.png", dpi=120); plt.close(fig)

    # 3) Feature fill-rate bar
    paper = pd.read_csv(OUT / "paper_level.csv")
    yp = paper[paper.paper_id.isin(long.paper_id.unique())]
    feats = [c for c in yp.columns if c not in
             ("paper_id", "reported_optimum_pct", "empirical_optimum_pct")]
    fill = yp[feats].notna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.barh(range(len(fill)), fill.values * 100)
    ax.set_yticks(range(len(fill))); ax.set_yticklabels(fill.index, fontsize=7)
    ax.set(xlabel="% of yielding papers with value", title="Feature availability")
    fig.tight_layout(); fig.savefig(FIG / "data_availability.png", dpi=120); plt.close(fig)

    print(f"saved figures to {FIG}")


if __name__ == "__main__":
    main()
