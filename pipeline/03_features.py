"""
Step 3 - Feature engineering: numeric parsing, canonicalized categoricals,
missingness handling, and assembly of the modeling tables.

Outputs:
  out/train_features.csv   long CS table joined with paper-level features
                           (one row per (paper_id, replacement_pct, age_days))
  out/paper_level.csv      per-paper features + reported & empirical optimum %
  out/feature_report.txt   fill-rates, dropped features, canonicalization counts

Imputation note: we deliberately leave NaNs in place. The step-4 models
(XGBoost / HistGradientBoosting) handle missing values natively; imputers
(MICE/KNN) are fit *inside* CV in step 4 only for models that require it.

Run: python pipeline/03_features.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Numeric paper-level features to parse (feature name -> keep).
NUMERIC_FEATS = [
    "SiO2", "CaO", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O",
    "Loss on Ignition (LOI)", "Amorphous Content", "Specific Gravity",
    "Mean Particle Size", "Specific Surface Area", "BET Surface Area",
    "Blaine Fineness", "Water-to-Binder Ratio (w/b)",
    "Thermal Treatment Temperature", "Thermal Treatment Duration",
    "Grinding Duration",
]
CATEG_FEATS = ["Pozzolanic Reactivity", "Activation Method", "XRD Phases",
               "RHA Processing Method"]
OPT_COL = "Optimum Replacement Level of Cement with Rice Husk Ash (RHA)::Extracted Value"

DROP_THRESHOLD = 0.95  # drop features missing in >95% of yielding papers


def val(feat):
    return f"{feat}::Extracted Value"


def parse_numeric_mean(cell):
    """Mean of all numbers in a cell. Handles '94.6 (RHA-A), 93.7 (RHA-B)' etc."""
    if not isinstance(cell, str):
        return np.nan
    nums = [float(x) for x in NUM_RE.findall(cell)]
    # for oxides/props the parenthetical numbers are sample variants -> mean is fine;
    # a bare percentage sign or unit leaves no spurious numbers.
    return float(np.mean(nums)) if nums else np.nan


# --- canonicalizers --------------------------------------------------------
def canon_pozzolanic(s):
    if not isinstance(s, str):
        return np.nan
    t = s.lower()
    if re.search(r"\b(low|poor|weak)\b", t):
        return "low"
    if re.search(r"\b(moderate|medium|intermediate)\b", t):
        return "medium"
    if re.search(r"high|efficient|reactive|active|important", t):
        return "high"
    return np.nan


def canon_activation(s):
    if not isinstance(s, str):
        return np.nan
    t = s.lower()
    thermal = bool(re.search(r"thermal|burn|incinerat|calcin|heat", t))
    mech = bool(re.search(r"mechanic|grind|mill|pulveriz", t))
    if thermal and mech:
        return "both"
    if thermal:
        return "thermal"
    if mech:
        return "mechanical"
    return "none"


def canon_xrd(s):
    if not isinstance(s, str):
        return np.nan
    t = s.lower()
    amorph = bool(re.search(r"amorph|non-crystall|noncrystall|vitreous|glassy", t))
    cryst = bool(re.search(r"quartz|cristobalite|tridymite|coesite|crystall", t))
    if amorph and cryst:
        return "mixed"
    if amorph:
        return "amorphous"
    if cryst:
        return "crystalline"
    return np.nan


def canon_processing(s):
    if not isinstance(s, str):
        return np.nan
    t = s.lower()
    if re.search(r"uncontrolled|open[ -]?(field|air)|heaped|drum", t):
        return "uncontrolled_burn"
    if re.search(r"controlled|furnace|specially designed|reactor|boiler", t):
        return "controlled_burn"
    if re.search(r"burn|incinerat|combust|thermal", t):
        return "burn_other"
    return "other"


CANON = {
    "Pozzolanic Reactivity": ("pozzolanic_reactivity", canon_pozzolanic),
    "Activation Method": ("activation_method", canon_activation),
    "XRD Phases": ("mineralogy", canon_xrd),
    "RHA Processing Method": ("processing_method", canon_processing),
}
POZZ_ORDINAL = {"low": 0, "medium": 1, "high": 2}


def parse_optimum(cell):
    if not isinstance(cell, str):
        return np.nan
    m = NUM_RE.search(cell)
    return float(m.group()) if m else np.nan


def main():
    w = pd.read_csv(OUT / "papers_wide.csv")
    long = pd.read_csv(OUT / "train_long.csv")
    yielding = sorted(long.paper_id.unique())

    # --- build paper-level feature frame ---
    feats = pd.DataFrame({"paper_id": w.paper_id})
    short = {  # tidy column names
        "SiO2": "sio2", "CaO": "cao", "Al2O3": "al2o3", "Fe2O3": "fe2o3",
        "MgO": "mgo", "SO3": "so3", "Na2O": "na2o", "K2O": "k2o",
        "Loss on Ignition (LOI)": "loi", "Amorphous Content": "amorphous_pct",
        "Specific Gravity": "specific_gravity", "Mean Particle Size": "mean_particle_um",
        "Specific Surface Area": "specific_surface", "BET Surface Area": "bet_surface",
        "Blaine Fineness": "blaine", "Water-to-Binder Ratio (w/b)": "wb_ratio",
        "Thermal Treatment Temperature": "burn_temp_c",
        "Thermal Treatment Duration": "burn_duration",
        "Grinding Duration": "grind_duration",
    }
    for f in NUMERIC_FEATS:
        feats[short[f]] = w[val(f)].map(parse_numeric_mean)

    for f, (newname, fn) in CANON.items():
        feats[newname] = w[val(f)].map(fn)
    # ordinal encode pozzolanic reactivity
    feats["pozzolanic_reactivity"] = feats["pozzolanic_reactivity"].map(POZZ_ORDINAL)

    # optimum targets
    feats["reported_optimum_pct"] = w[OPT_COL].map(parse_optimum)
    emp = (long[long.age_days == 28]
           .sort_values("strength_mpa")
           .groupby("paper_id").tail(1)
           .set_index("paper_id")["replacement_pct"])
    feats["empirical_optimum_pct"] = feats["paper_id"].map(emp)

    # --- fill-rate report + drop near-empty features (among yielding papers) ---
    fy = feats[feats.paper_id.isin(yielding)]
    feature_cols = [c for c in feats.columns
                    if c not in ("paper_id", "reported_optimum_pct", "empirical_optimum_pct")]
    fill = fy[feature_cols].notna().mean().sort_values()
    dropped = [c for c in feature_cols if fill[c] < (1 - DROP_THRESHOLD)]
    kept = [c for c in feature_cols if c not in dropped]

    # --- merge to long modeling table ---
    model_tbl = long.merge(feats[["paper_id"] + kept], on="paper_id", how="left")
    model_tbl.to_csv(OUT / "train_features.csv", index=False, encoding="utf-8")
    feats.to_csv(OUT / "paper_level.csv", index=False, encoding="utf-8")

    # --- report ---
    lines = []
    lines.append(f"Yielding papers: {len(yielding)}  |  CS rows: {len(long)}")
    lines.append(f"Modeling table:  {model_tbl.shape}  ->  train_features.csv")
    lines.append(f"\nDropped features (>{int(DROP_THRESHOLD*100)}% missing): {dropped or 'none'}")
    lines.append("\nFill-rate among yielding papers (kept features):")
    for c in kept:
        lines.append(f"  {fill[c]*100:5.1f}%  {c}")
    lines.append("\nCanonical category counts:")
    for _, (newname, _) in CANON.items():
        vc = fy[newname].value_counts(dropna=False).to_dict()
        lines.append(f"  {newname}: {vc}")
    lines.append("\nOptimum targets (yielding papers):")
    lines.append(f"  reported_optimum non-null : {fy['reported_optimum_pct'].notna().sum()}")
    lines.append(f"  empirical_optimum non-null: {fy['empirical_optimum_pct'].notna().sum()}")
    report = "\n".join(lines)
    (OUT / "feature_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
