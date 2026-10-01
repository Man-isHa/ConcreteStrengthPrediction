"""
Step 10 - Same outlier-removal comparison as 09_run_no_outliers.py, but using
real XGBoost (identical model class/hyper-parameter grid to 04_model.py and
07_best_model.py) instead of the sklearn HistGB fallback.

Why a separate script: on the Windows Python used for the rest of this
project, xgboost.dll is blocked by Windows Smart App Control (an OS-level
security feature; disabling it is a one-way door short of a Windows
reinstall, so it was left alone). This script is meant to be run under the
WSL Ubuntu venv at /tmp/rha_venv, where xgboost imports fine, so we can
report genuine XGBoost numbers instead of a substitute backend.

Cutoff: identical to 09_run_no_outliers.py -- papers whose maximum reported
strength exceeds 120 MPa (papers 17 and 36) are dropped in full.

Re-runs, unmodified, the original pipeline/04_model.py (framing A/B/C
comparison) and pipeline/07_best_model.py (tuned monotone-constrained
XGBoost) against both the unfiltered and filtered tables, by importing those
modules and overriding their OUT path -- no logic is duplicated/reimplemented,
so results are directly comparable to what's already in pipeline/out/.

Run (inside WSL, with xgboost installed):
  /tmp/rha_venv/bin/python pipeline/10_run_no_outliers_xgb.py
"""
import importlib.util
import shutil
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "out"
DST_BEFORE = HERE / "out_xgb_before"
DST_AFTER = HERE / "out_xgb_after"
for d in (DST_BEFORE, DST_AFTER):
    d.mkdir(exist_ok=True)

CUTOFF_MPA = 120.0


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pair(train_features, tag_dir, drop_ids):
    tf = train_features if drop_ids is None else train_features[~train_features.paper_id.isin(drop_ids)]
    tf.to_csv(tag_dir / "train_features.csv", index=False)
    shutil.copy(SRC / "paper_level.csv", tag_dir / "paper_level.csv")
    shutil.copy(SRC / "papers_wide.csv", tag_dir / "papers_wide.csv")

    m04 = load_module(f"m04_{tag_dir.name}", HERE / "04_model.py")
    m04.OUT = tag_dir
    m04.report_lines = []
    m04.main()

    m07 = load_module(f"m07_{tag_dir.name}", HERE / "07_best_model.py")
    m07.OUT = tag_dir
    m07.main()


def main():
    tf = pd.read_csv(SRC / "train_features.csv")
    tl = pd.read_csv(SRC / "train_long.csv")

    paper_max = tl.groupby("paper_id")["strength_mpa"].max()
    drop_ids = sorted(paper_max[paper_max > CUTOFF_MPA].index.tolist())
    print(f"Backend check: ", end="")
    try:
        from xgboost import XGBRegressor  # noqa
        print("xgboost available - using real XGBoost for this run.")
    except Exception as e:
        print(f"xgboost NOT available ({e}); 04_model.py will silently fall back to HistGB.")

    print(f"Papers dropped (max strength > {CUTOFF_MPA:.0f} MPa): {drop_ids}")

    print("\n" + "=" * 70)
    print("BEFORE (all 44 papers, unfiltered) -- pipeline/out_xgb_before/")
    print("=" * 70)
    run_pair(tf, DST_BEFORE, None)

    print("\n" + "=" * 70)
    print("AFTER (UHPC-outlier papers 17 & 36 removed) -- pipeline/out_xgb_after/")
    print("=" * 70)
    run_pair(tf, DST_AFTER, drop_ids)


if __name__ == "__main__":
    main()
