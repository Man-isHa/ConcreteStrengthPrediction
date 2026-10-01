"""
Step 4 - baseline cross-validated models over the three target framings.

Establishes the headline result before any ablation: absolute strength across
papers is not learnable, but blend strength given the paper's own control is.

Run: python pipeline_rcf/04_model.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE, BASE_NOCTRL, CHEM, FINE, NOMINAL, N_REPEATS,  # noqa: E402
                    ORDINAL, OUT, PROC, add_control, evaluate)

report_lines = []


def log(s=""):
    print(s)
    report_lines.append(s)


def load():
    """The modelling table: age-resolved rows only.

    Rows whose curing age never resolved are kept in train_features.csv and are
    revisited in Study 6; they cannot be used here because age is a feature.
    """
    df = pd.read_csv(OUT / "train_features.csv")
    return df[df.age_known].reset_index(drop=True)


def show(label, res):
    if res is None:
        log(f"[{label}]  no usable splits")
        return
    log(f"[{label}]  n={res['n_rows']} papers={res['n_papers']} "
        f"repeats={res['repeats']}")
    log(f"    R2={res['r2_mean']:.3f}+/-{res['r2_std']:.3f}  "
        f"RMSE={res['rmse_mean']:.2f}  MAE={res['mae_mean']:.2f}")


def main():
    df = load()
    structured_num = CHEM + FINE + PROC + ORDINAL
    log(f"age-resolved rows: {len(df)} over {df.paper_id.nunique()} papers")
    log(f"strength range   : {df.strength_mpa.min():.1f}-"
        f"{df.strength_mpa.max():.1f} MPa")
    log()

    # ---- framing A: absolute strength, every paper, no control input
    featsA = BASE_NOCTRL + structured_num + NOMINAL
    resA = evaluate(df, featsA, "strength_mpa", "xgb")
    show("A: absolute, all papers", resA)

    # ---- framings B and C need the paper's own 0% control at the same age
    d = add_control(df)
    log()
    log(f"with-control subset: {len(d)} blend rows over "
        f"{d.paper_id.nunique()} papers "
        f"(control rows excluded from evaluation)")
    log(f"    control strength range: {d.ctrl_strength.min():.1f}-"
        f"{d.ctrl_strength.max():.1f} MPa")
    log()

    d = d.copy()
    d["rel_ctrl"] = d.strength_mpa / d.ctrl_strength
    resB = evaluate(d, featsA, "rel_ctrl", "xgb")
    show("B: relative-to-control", resB)

    featsC = BASE + structured_num + NOMINAL
    resC = evaluate(d, featsC, "strength_mpa", "xgb")
    show("C: blend | control, all structured", resC)

    resC1 = evaluate(d, BASE, "strength_mpa", "xgb")
    show("C1: blend | control, base only", resC1)

    resC2 = evaluate(d, BASE + CHEM, "strength_mpa", "xgb")
    show("C2: base + chemistry", resC2)

    log()
    best = max([("A", resA), ("B", resB), ("C", resC), ("C1", resC1),
                ("C2", resC2)],
               key=lambda kv: kv[1]["r2_mean"] if kv[1] else -9e9)
    log(f"best framing by mean R2: {best[0]} "
        f"({best[1]['r2_mean']:.3f})")

    (OUT / "cv_results.txt").write_text("\n".join(report_lines),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()
