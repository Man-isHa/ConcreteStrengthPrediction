"""Predict compressive strength of RCF-blended concrete (final model, <= 60 MPa range).

Loads models/rcf_strength_model.joblib and predicts from user-supplied inputs. Does all
preprocessing the saved model expects: selects the 18 input columns, one-hot encodes the
four categorical columns with an explicit "missing" level, and puts the columns in the
exact order the model was fitted with (read from the metadata file, so it can never drift).

USAGE
  Single mix (only the three core inputs are required):
    python inference/predict_rcf.py --replacement_pct 20 --age_days 28 --ctrl_strength 42

  With powder properties (any subset; anything omitted is treated as "not reported"):
    python inference/predict_rcf.py --replacement_pct 20 --age_days 28 --ctrl_strength 42 \
        --material_family powder --processing_method milling --mean_particle_um 15

  A batch from CSV (one row per mix, column names as listed below):
    python inference/predict_rcf.py --csv my_mixes.csv --out predictions.csv

INPUTS
  required : replacement_pct     % of cement replaced by RCF (0-100)
             age_days            curing age in days
             ctrl_strength       the SAME mix without RCF (0% replacement), tested at the
                                 SAME curing age, in MPa. Cast and test the plain mix first.
  optional : activation_method   carbonation | chemical | mechanical | multiple | none | other | thermal
             processing_method   combined | crushing | milling | other | thermal
             mineralogy          calcite | mixed | other | portlandite | quartz
             material_family     aac | cdw | fines | other | paste | powder
             pozzolanic_reactivity   low | medium | high   (or 0 | 1 | 2)
             d10, d50, d90, mean_particle_um   particle sizes, micrometres
             specific_surface, bet_surface, blaine   surface areas
             specific_gravity, bulk_density, water_absorption

The model was trained on concrete up to 60 MPa; predictions outside roughly 1-60 MPa, or
for control strengths far outside 12-60 MPa, are extrapolation and are flagged.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "models" / "rcf_strength_model.joblib"
META_FILE = ROOT / "models" / "rcf_strength_model_metadata.json"

NOMINAL = ["activation_method", "processing_method", "mineralogy", "material_family"]
REQUIRED = ["replacement_pct", "age_days", "ctrl_strength"]
POZZ = {"low": 0.0, "medium": 1.0, "high": 2.0, "0": 0.0, "1": 1.0, "2": 2.0}
TRAIN_RANGE = {"strength": (1.0, 60.0), "ctrl_strength": (12.0, 60.0)}


def load():
    if not MODEL_FILE.exists():
        sys.exit(f"model not found: {MODEL_FILE}\nRun: python inference/build_final_models.py")
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    return joblib.load(MODEL_FILE), meta


def prepare(df, meta):
    """Build the exact matrix the model was fitted on."""
    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns or df[c].isna().any()]
    if missing:
        sys.exit(f"these inputs are required for every row: {', '.join(missing)}")

    if "pozzolanic_reactivity" in df.columns:
        df["pozzolanic_reactivity"] = df["pozzolanic_reactivity"].map(
            lambda v: POZZ.get(str(v).strip().lower(), pd.to_numeric(v, errors="coerce"))
            if pd.notna(v) else pd.NA)

    for col in meta["input_features"]:                 # add anything not supplied
        if col not in df.columns:
            df[col] = pd.NA
    for col in NOMINAL:                                # normalise category text
        df[col] = df[col].map(lambda v: str(v).strip().lower() if pd.notna(v) else pd.NA)

    numeric = [c for c in meta["input_features"] if c not in NOMINAL]
    X = df[numeric].apply(pd.to_numeric, errors="coerce")
    for col in NOMINAL:                                # one-hot, "missing" is its own level
        for level in [c for c in meta["encoded_columns"] if c.startswith(col + "_")]:
            value = level[len(col) + 1:]
            X[level] = ((df[col] == value) if value != "nan" else df[col].isna()).astype(float)
    unknown = {c: sorted(set(df[c].dropna()) -
                         {l[len(c) + 1:] for l in meta["encoded_columns"] if l.startswith(c + "_")})
               for c in NOMINAL}
    for col, vals in unknown.items():
        if vals:
            print(f"note: unrecognised {col} value(s) {vals} treated as 'not reported'")
    return X[meta["encoded_columns"]]


def warn(df, preds):
    lo, hi = TRAIN_RANGE["ctrl_strength"]
    for i, row in df.reset_index(drop=True).iterrows():
        if not lo <= float(row.ctrl_strength) <= hi:
            print(f"note: row {i + 1} control strength {row.ctrl_strength} MPa is outside the "
                  f"{lo:.0f}-{hi:.0f} MPa range the model was trained on")
        if not 0 < float(row.replacement_pct) <= 100:
            print(f"note: row {i + 1} replacement {row.replacement_pct}% is outside 0-100%")
    slo, shi = TRAIN_RANGE["strength"]
    for i, p in enumerate(preds):
        if not slo <= p <= shi:
            print(f"note: row {i + 1} prediction {p:.1f} MPa falls outside the "
                  f"{slo:.0f}-{shi:.0f} MPa range the model was trained for")


def main():
    ap = argparse.ArgumentParser(description="Predict RCF-blend compressive strength (MPa).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--csv", help="CSV of input rows (batch mode)")
    ap.add_argument("--out", help="write predictions to this CSV")
    for col in REQUIRED + ["pozzolanic_reactivity", "d10", "d50", "d90", "mean_particle_um",
                           "specific_surface", "bet_surface", "blaine", "specific_gravity",
                           "bulk_density", "water_absorption"] + NOMINAL:
        ap.add_argument(f"--{col}")
    args = ap.parse_args()

    model, meta = load()
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        given = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("csv", "out")}
        if not given:
            ap.print_help()
            sys.exit(0)
        df = pd.DataFrame([given])

    preds = model.predict(prepare(df, meta))
    warn(df, preds)

    result = df.copy()
    result["predicted_strength_mpa"] = [round(float(v), 2) for v in preds]
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"wrote {len(result)} prediction(s) -> {args.out}")
    else:
        print()
        for i, (_, r) in enumerate(result.iterrows(), 1):
            print(f"  {i}. {r.replacement_pct}% RCF, {r.age_days} d, control "
                  f"{r.ctrl_strength} MPa  ->  {r.predicted_strength_mpa} MPa")
        print(f"\nModel: {meta['name']}")
        print(f"Expected accuracy on an unseen paper: MAE about "
              f"{meta['reported_performance_grouped_holdout']['mae']} MPa "
              f"(R2 {meta['reported_performance_grouped_holdout']['r2']}).")


if __name__ == "__main__":
    main()
