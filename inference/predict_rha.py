"""Predict compressive strength of RHA-blended concrete (final model, <= 60 MPa range).

Loads models/rha_strength_model.json (XGBoost) and predicts from user-supplied inputs.
The final RHA model deliberately uses only three inputs - the study found that adding ash
chemistry, fineness or processing columns did not improve predictions on unseen papers.

USAGE
  Single mix:
    python inference/predict_rha.py --replacement_pct 10 --age_days 28 --ctrl_strength 40

  Strength-vs-replacement sweep for one mix design:
    python inference/predict_rha.py --age_days 28 --ctrl_strength 40 --sweep 0:30:2.5

  A batch from CSV (columns: replacement_pct, age_days, ctrl_strength):
    python inference/predict_rha.py --csv my_mixes.csv --out predictions.csv

INPUTS (all three required)
  replacement_pct   % of cement replaced by rice-husk ash (0-100)
  age_days          curing age in days
  ctrl_strength     the SAME mix without RHA (0% replacement), tested at the SAME curing
                    age, in MPa. Cast and test the plain mix first; this is the single
                    most important input.

The model is monotone: predicted strength never falls as curing age or control strength
rise. It was trained on concrete up to 60 MPa; predictions outside roughly 1-60 MPa are
extrapolation and are flagged.

NOTE ON THIS MACHINE: XGBoost cannot load under Windows here (Smart App Control blocks
the unsigned DLL), so run this script inside the WSL venv:
    ~/rha_venv/bin/python inference/predict_rha.py ...
On any machine with a normal xgboost install, plain `python` works.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_JSON = ROOT / "models" / "rha_strength_model.json"
MODEL_PKL = ROOT / "models" / "rha_strength_model.joblib"
META_FILE = ROOT / "models" / "rha_strength_model_metadata.json"

FEATURES = ["replacement_pct", "age_days", "ctrl_strength"]
TRAIN_RANGE = {"strength": (1.0, 60.0), "ctrl_strength": (5.0, 58.0)}


def load():
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        sys.exit(f"xgboost could not be loaded ({exc}).\n"
                 f"On this machine run it under WSL:\n"
                 f"    ~/rha_venv/bin/python inference/predict_rha.py ...")
    model = XGBRegressor()
    if MODEL_JSON.exists():
        model.load_model(MODEL_JSON)          # portable across xgboost versions
    elif MODEL_PKL.exists():
        import joblib
        model = joblib.load(MODEL_PKL)
    else:
        sys.exit(f"model not found in {ROOT / 'models'}\n"
                 f"Run: python inference/build_final_models.py")
    return model, meta


def prepare(df):
    missing = [c for c in FEATURES if c not in df.columns or df[c].isna().any()]
    if missing:
        sys.exit(f"these inputs are required for every row: {', '.join(missing)}")
    return df[FEATURES].apply(pd.to_numeric, errors="coerce")


def warn(df, preds):
    lo, hi = TRAIN_RANGE["ctrl_strength"]
    for i, row in df.reset_index(drop=True).iterrows():
        if not lo <= float(row.ctrl_strength) <= hi:
            print(f"note: row {i + 1} control strength {row.ctrl_strength} MPa is outside the "
                  f"{lo:.0f}-{hi:.0f} MPa range the model was trained on")
    slo, shi = TRAIN_RANGE["strength"]
    for i, p in enumerate(preds):
        if not slo <= p <= shi:
            print(f"note: row {i + 1} prediction {p:.1f} MPa falls outside the "
                  f"{slo:.0f}-{shi:.0f} MPa range the model was trained for")


def main():
    ap = argparse.ArgumentParser(description="Predict RHA-blend compressive strength (MPa).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--csv", help="CSV of input rows (batch mode)")
    ap.add_argument("--out", help="write predictions to this CSV")
    ap.add_argument("--sweep", metavar="START:STOP:STEP",
                    help="sweep replacement %% for one mix, e.g. 0:30:2.5")
    for col in FEATURES:
        ap.add_argument(f"--{col}", type=float)
    args = ap.parse_args()

    model, meta = load()
    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.sweep:
        if args.age_days is None or args.ctrl_strength is None:
            sys.exit("--sweep also needs --age_days and --ctrl_strength")
        start, stop, step = (float(x) for x in args.sweep.split(":"))
        levels = [start + i * step for i in range(int((stop - start) / step) + 1)]
        df = pd.DataFrame({"replacement_pct": levels, "age_days": args.age_days,
                           "ctrl_strength": args.ctrl_strength})
    else:
        given = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("csv", "out", "sweep")}
        if not given:
            ap.print_help()
            sys.exit(0)
        df = pd.DataFrame([given])

    preds = model.predict(prepare(df))
    warn(df, preds)

    result = df.copy()
    result["predicted_strength_mpa"] = [round(float(v), 2) for v in preds]
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} prediction(s) -> {args.out}")
    else:
        print()
        for i, (_, r) in enumerate(result.iterrows(), 1):
            print(f"  {i}. {r.replacement_pct}% RHA, {r.age_days} d, control "
                  f"{r.ctrl_strength} MPa  ->  {r.predicted_strength_mpa} MPa")
        if args.sweep:
            best = result.loc[result.predicted_strength_mpa.idxmax()]
            print(f"\nHighest predicted strength at {best.replacement_pct}% replacement "
                  f"({best.predicted_strength_mpa} MPa). Treat this as indicative only: the "
                  f"study found the optimum-replacement task unreliable on this corpus.")
        print(f"\nModel: {meta['name']}")
        print(f"Expected accuracy on an unseen paper: MAE about "
              f"{meta['reported_performance_grouped_holdout']['mae']} MPa "
              f"(R2 {meta['reported_performance_grouped_holdout']['r2']}).")


if __name__ == "__main__":
    main()
