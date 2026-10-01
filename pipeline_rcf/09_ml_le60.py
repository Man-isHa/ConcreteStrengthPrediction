"""
Step 9 - the RCF pipeline re-run on samples <= 60 MPa only.

Every row whose measured strength exceeds 60 MPa is dropped (control rows too), then
06_ablation.py and 07_best_model.py are run unchanged against the filtered table by
pointing their OUT at pipeline_rcf/out_le60/. Nothing in out/ is touched.

Run (WSL venv, xgboost needed): ~/rha_venv/bin/python pipeline_rcf/09_ml_le60.py
"""
import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "out", HERE / "out_le60"
CUT = 60.0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare():
    DST.mkdir(exist_ok=True)
    (DST / "figs").mkdir(exist_ok=True)
    full = pd.read_csv(SRC / "train_features.csv")
    kept = full[full.strength_mpa <= CUT]
    kept.to_csv(DST / "train_features.csv", index=False)
    for f in ["paper_level.csv", "papers_wide.csv"]:
        shutil.copy(SRC / f, DST / f)
    tl = pd.read_csv(SRC / "train_long.csv")
    tl[tl.strength_mpa <= CUT].to_csv(DST / "train_long.csv", index=False)

    aged = kept[kept.age_known]
    ctrl = aged[aged.replacement_pct == 0].groupby(["paper_id", "age_days"]).size()
    blend = aged.merge(ctrl.rename("n").reset_index()[["paper_id", "age_days"]],
                       on=["paper_id", "age_days"])
    blend = blend[blend.replacement_pct > 0]
    print(f"all rows {len(full)} -> {len(kept)} (papers {full.paper_id.nunique()} -> "
          f"{kept.paper_id.nunique()}); dropped {len(full) - len(kept)} rows above {CUT:.0f} MPa "
          f"from {full[full.strength_mpa > CUT].paper_id.nunique()} papers")
    print(f"framing C after filter: {len(blend)} rows / {blend.paper_id.nunique()} papers")


def main():
    prepare()
    common = load("common", HERE / "common.py")
    common.OUT, common.FIG = DST, DST / "figs"

    for name, path in [("abl_le60", HERE / "06_ablation.py"),
                       ("best_le60", HERE / "07_best_model.py")]:
        mod = load(name, path)
        mod.OUT = DST
        print(f"\n{'=' * 70}\n{path.name} on <= {CUT:.0f} MPa data\n{'=' * 70}", flush=True)
        mod.main()


if __name__ == "__main__":
    main()
