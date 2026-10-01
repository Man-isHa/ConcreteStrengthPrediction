"""
Step 5 - quick EDA sanity plots.

These are diagnostics for the analyst, not report figures; the publication set is
built by 08_report_figs.py.

Run: python pipeline_rcf/05_plots.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BLUE, FIG, OUT, use_plot_style  # noqa: E402

plt = use_plot_style()


def main():
    long = pd.read_csv(OUT / "train_features.csv")
    aged = long[long.age_known]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(aged.age_days, aged.strength_mpa, s=10, alpha=0.4, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlabel("curing age (days, log)")
    ax.set_ylabel("compressive strength (MPa)")
    ax.set_title("Strength vs curing age (all papers pooled)")
    fig.tight_layout(); fig.savefig(FIG / "cs_vs_age.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    d28 = aged[aged.age_days == 28]
    top = (d28.groupby("paper_id").size().sort_values(ascending=False)
           .head(6).index)
    for pid in top:
        s = (d28[d28.paper_id == pid]
             .groupby("replacement_pct", as_index=False)["strength_mpa"].mean()
             .sort_values("replacement_pct"))
        ax.plot(s.replacement_pct, s.strength_mpa, "o-", lw=1.2,
                label=f"paper {pid}")
    ax.set_xlabel("cement replacement (%)")
    ax.set_ylabel("28-day strength (MPa)")
    ax.set_title("28-day strength vs replacement level")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "cs_vs_replacement.png"); plt.close(fig)

    print(f"wrote {FIG / 'cs_vs_age.png'}")
    print(f"wrote {FIG / 'cs_vs_replacement.png'}")


if __name__ == "__main__":
    main()
