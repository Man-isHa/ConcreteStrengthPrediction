"""
Step 3 - build the modelling table.

Numeric cells are parsed as the mean of the numbers they contain, but every
feature carries a physical range guard. That guard is not cosmetic: the w/b cell
"RC: 0.41 / C30R: 0.45 / C100R: 0.48" contains 30 and 100 from the mix labels,
and an unguarded mean would report a water/binder ratio of 26.

High-cardinality free text is canonicalised into compact categoricals. No
imputation is applied - the tree models handle NaN natively, and imputers are
only ever fit inside a CV split (Study 4).

Run: python pipeline_rcf/03_features.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CHEM, FINE, NOMINAL, NUM_RE, OUT, ORDINAL, PROC,  # noqa: E402
                    is_null)

DROP_THRESHOLD = 0.95      # drop a feature filled in <5% of yielding papers

# raw Sheet1 feature name -> (short name, low, high)
# The ranges are physical plausibility bounds in the sheet's own units.
NUMERIC_FEATS = {
    "SiO2": ("sio2", 0, 100),
    "CaO": ("cao", 0, 100),
    "Al2O3": ("al2o3", 0, 100),
    "Fe2O3": ("fe2o3", 0, 100),
    "MgO": ("mgo", 0, 100),
    "SO3": ("so3", 0, 100),
    "Na2O": ("na2o", 0, 100),
    "K2O": ("k2o", 0, 100),
    "Loss on Ignition (LOI)": ("loi", 0, 100),
    "Amorphous Content": ("amorphous_pct", 0, 100),
    "d10": ("d10", 0.01, 2000),
    "d50": ("d50", 0.01, 2000),
    "d90": ("d90", 0.01, 2000),
    "Mean Particle Size": ("mean_particle_um", 0.01, 2000),
    "Specific Surface Area": ("specific_surface", 50, 50000),
    "BET Surface Area": ("bet_surface", 50, 50000),
    "Blaine Fineness": ("blaine", 50, 50000),
    "Water Absorption Capacity": ("water_absorption", 0, 30),
    "Specific Gravity": ("specific_gravity", 1.5, 4.0),
    "Bulk Density": ("bulk_density", 200, 3500),
    "Water-to-Binder Ratio (w/b)": ("wb_ratio", 0.15, 1.2),
    "Thermal Treatment Temperature": ("thermal_temp_c", 50, 1600),
    "Thermal Treatment Duration": ("thermal_duration", 0.01, 72),
    "Grinding Duration": ("grind_duration", 0.01, 600),
    "Grinding Speed": ("grind_speed", 10, 2000),
}

# Cells that are prose rather than a measurement. "Disordered phase of jennite
# (in #2, #5, #6)" is an Amorphous Content cell whose only numbers are sample
# labels; requiring a numeric-looking cell keeps those out.
PROSE_GUARD = {"amorphous_pct", "water_absorption"}
NOT_QUANTIFIED = re.compile(r"not\s+(?:quantitativ\w*\s+)?(?:specified|reported|"
                            r"quantified|given)|exact value not", re.I)

# The same column mixes units row by row - Specific Surface Area alone appears as
# m2/kg, m2/g and cm2/g - so values must be converted before any range guard is
# applied, or two thirds of the column silently disappears.
CANON_UNITS = {
    "specific_surface": "surface", "bet_surface": "surface", "blaine": "surface",
    "bulk_density": "density",
    "d10": "length", "d50": "length", "d90": "length",
    "mean_particle_um": "length",
}
UNIT_SCALES = {
    # canonical: surface = m2/kg, density = kg/m3, length = um
    "surface": [("m2/kg", 1.0), ("m2/g", 1000.0), ("cm2/g", 0.1),
                ("cm2/cm3", 1.0)],
    "density": [("kg/m3", 1.0), ("g/cm3", 1000.0), ("kg/dm3", 1000.0),
                ("g/ml", 1000.0)],
    "length": [("um", 1.0), ("nm", 0.001), ("mm", 1000.0)],
}
# Applied when the unit cell is missing or unrecognised: a surface area below
# 100 cannot be m2/kg for a ground powder, so it was reported in m2/g.
MAGNITUDE_FALLBACK = {"surface": (100.0, 1000.0), "density": (10.0, 1000.0)}


def normalise_unit(unit):
    if is_null(unit):
        return ""
    u = str(unit).lower().replace("²", "2").replace("³", "3").replace("^", "")
    u = u.replace("µ", "u").replace("μ", "u")
    return re.sub(r"\s+", "", u)


def unit_scale(short, unit):
    kind = CANON_UNITS.get(short)
    if kind is None:
        return 1.0
    u = normalise_unit(unit)
    for token, mult in UNIT_SCALES[kind]:
        if token in u:
            return mult
    return None          # unrecognised -> caller applies the magnitude fallback


def parse_numeric(cell, unit, short, lo, hi, guard=False):
    """Mean of the in-range numbers in a cell, converted to canonical units."""
    if is_null(cell):
        return np.nan
    s = str(cell)
    if NOT_QUANTIFIED.search(s):
        return np.nan
    if guard and "%" not in s and len(s) > 24:
        return np.nan

    vals = [float(v) for v in NUM_RE.findall(s)]
    if not vals:
        return np.nan

    scale = unit_scale(short, unit)
    kind = CANON_UNITS.get(short)
    if scale is None:
        scale = 1.0
        if kind in MAGNITUDE_FALLBACK:
            thresh, mult = MAGNITUDE_FALLBACK[kind]
            if float(np.median(vals)) < thresh:
                scale = mult
    vals = [v * scale for v in vals]

    vals = [v for v in vals if lo <= v <= hi]
    return float(np.mean(vals)) if vals else np.nan


# ------------------------------------------------------------------ canonicalisers
ACT_PATTERNS = [
    ("thermal", r"thermal|calcin|sinter|burn|heat|dehydrat"),
    ("carbonation", r"carbonat|co\s*2|co₂"),
    ("mechanical", r"mechanic|mill|grind"),
    ("chemical", r"chemical|alkali|nano|activator|na2so4|sodium|gypsum"),
]


def canon_activation(cell):
    """Activation route. Carbonation gets its own level rather than being folded
    into a generic 'other': CO2 curing is a central RCF-specific treatment."""
    if is_null(cell):
        return np.nan
    s = str(cell).lower()
    if re.search(r"\bnone\b|no activation|untreated|not activated|without", s):
        return "none"
    hits = [name for name, pat in ACT_PATTERNS if re.search(pat, s)]
    if not hits:
        return "other"
    return hits[0] if len(hits) == 1 else "multiple"


PROC_PATTERNS = [
    ("thermal", r"thermal|calcin|burn|dry|heat"),
    ("milling", r"mill|grind|pulveris|pulveriz"),
    ("crushing", r"crush|siev|screen|jaw"),
]


def canon_processing(cell):
    if is_null(cell):
        return np.nan
    s = str(cell).lower()
    hits = [name for name, pat in PROC_PATTERNS if re.search(pat, s)]
    if not hits:
        return "other"
    return hits[0] if len(hits) == 1 else "combined"


def canon_pozzolanic(cell):
    """Ordinal 0/1/2. Negative markers are checked first, because phrases like
    'Limited pozzolanic reactivity' contain the positive word too."""
    if is_null(cell):
        return np.nan
    s = str(cell).lower()
    if re.search(r"nonreact|non-react|inert|very low|\bpoor\b|not significant|"
                 r"negligible|unlikely|without pozzolan|no pozzolan", s):
        return 0.0
    if re.search(r"limited|moderate|medium|\blow\b|weak|slight|some ", s):
        return 1.0
    if re.search(r"high|strong|significant|excellent|good|confirmed|positive|"
                 r"reactive|pozzolanic activity|promote", s):
        return 2.0
    return 1.0


MINERAL_PATTERNS = [
    ("calcite", r"calcite|caco3|caco₃|carbonate"),
    ("portlandite", r"portlandite|ca\(oh\)2|ca\(oh\)₂|hydroxide"),
    ("quartz", r"quartz|sio2 phase|silica"),
    ("csh", r"c-s-h|csh|tobermorite|jennite"),
]


def canon_mineralogy(cell):
    if is_null(cell):
        return np.nan
    s = str(cell).lower()
    hits = [name for name, pat in MINERAL_PATTERNS if re.search(pat, s)]
    if not hits:
        return "other"
    return hits[0] if len(hits) == 1 else "mixed"


FAMILY_PATTERNS = [
    ("aac", r"aerated|\baac\b"),
    ("cdw", r"\bcdw\b|c&d|demolition|construction and demolition"),
    ("paste", r"paste"),
    ("fines", r"fines|\brcf\b|\bfrca?\b|fine recycled"),
    ("powder", r"powder|\brcp\b|\brp\b|\bwcp\b"),
]


def canon_family(cell):
    """Collapse the 90 free-text Material Type spellings into a few families."""
    if is_null(cell):
        return np.nan
    s = str(cell).lower()
    for name, pat in FAMILY_PATTERNS:
        if re.search(pat, s):
            return name
    return "other"


def parse_optimum(cell):
    """Reported optimum replacement level, in percentage points.

    Handles 'Below 15', 'Up to 25', '15% - 30%', '10'. When a cell reports one
    optimum per mechanical property, the compressive one is taken. Otherwise the
    median of the in-range numbers is used, which is robust to the cells that
    list a per-mix optimum table.
    """
    if is_null(cell):
        return np.nan
    s = str(cell)
    if re.search(r"compress", s, re.I) and re.search(r"flexur", s, re.I):
        seg = re.split(r"flexur", s, flags=re.I)[0]
        vals = [float(v) for v in NUM_RE.findall(seg)]
    else:
        vals = [float(v) for v in NUM_RE.findall(s)]
    vals = [v for v in vals if 0.0 <= v <= 100.0]
    return float(np.median(vals)) if vals else np.nan


def main():
    wide = pd.read_csv(OUT / "papers_wide.csv")
    long = pd.read_csv(OUT / "train_long.csv")

    feats = pd.DataFrame({"paper_id": wide["paper_id"]})
    for raw, (short, lo, hi) in NUMERIC_FEATS.items():
        col = f"{raw}::Extracted Value"
        if col not in wide.columns:
            print(f"  ! missing column: {col}")
            feats[short] = np.nan
            continue
        guard = short in PROSE_GUARD
        ucol = f"{raw}::Unit"
        units = wide[ucol] if ucol in wide.columns else pd.Series(
            np.nan, index=wide.index)
        feats[short] = [parse_numeric(c, u, short, lo, hi, guard)
                        for c, u in zip(wide[col], units)]

    feats["pozzolanic_reactivity"] = wide[
        "Pozzolanic Reactivity::Extracted Value"].map(canon_pozzolanic)
    feats["activation_method"] = wide[
        "Activation Method::Extracted Value"].map(canon_activation)
    feats["processing_method"] = wide[
        "RCF/RCP Processing Method::Extracted Value"].map(canon_processing)
    feats["mineralogy"] = wide[
        "Mineral Composition::Extracted Value"].map(canon_mineralogy)
    feats["material_family"] = wide[
        "Material Type::Extracted Value"].map(canon_family)
    feats["reported_optimum_pct"] = wide[
        "Optimum Replacement Level of Cement with RCF/RCP::Extracted Value"
    ].map(parse_optimum)

    # empirical optimum: the level with the highest mean 28-day strength
    d28 = long[long.age_days == 28]
    agg = (d28.groupby(["paper_id", "replacement_pct"], as_index=False)
              ["strength_mpa"].mean())
    emp = (agg.sort_values(["paper_id", "strength_mpa"])
              .groupby("paper_id").tail(1)
              .rename(columns={"replacement_pct": "empirical_optimum_pct"})
              [["paper_id", "empirical_optimum_pct"]])
    feats = feats.merge(emp, on="paper_id", how="left")

    yielding = sorted(long.paper_id.unique())
    among = feats[feats.paper_id.isin(yielding)]

    modelled = [c for c in feats.columns
                if c not in ("paper_id", "reported_optimum_pct",
                             "empirical_optimum_pct")]
    fill = among[modelled].notna().mean().sort_values(ascending=False)
    dropped = [c for c in modelled if fill[c] < (1 - DROP_THRESHOLD)]
    if dropped:
        feats = feats.drop(columns=dropped)

    train = long.merge(feats, on="paper_id", how="left")
    train.to_csv(OUT / "train_features.csv", index=False, encoding="utf-8")
    feats.to_csv(OUT / "paper_level.csv", index=False, encoding="utf-8")

    lines = [f"Feature availability among the {len(yielding)} yielding papers", ""]
    for c, v in fill.items():
        lines.append(f"  {c:24s} {v*100:5.1f}%")
    lines += ["", f"dropped (<{(1-DROP_THRESHOLD)*100:.0f}% filled): "
                  f"{dropped or 'none'}", ""]
    lines.append(f"reported_optimum_pct present : "
                 f"{among['reported_optimum_pct'].notna().sum()} papers")
    lines.append(f"empirical_optimum_pct present: "
                 f"{among['empirical_optimum_pct'].notna().sum()} papers")
    (OUT / "feature_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"train_features.csv : {train.shape[0]} rows x {train.shape[1]} cols")
    print(f"paper_level.csv    : {feats.shape[0]} papers")
    print(f"dropped features   : {dropped or 'none'}")
    print("\nfill rates among yielding papers (worst 8):")
    for c, v in fill.tail(8).items():
        print(f"  {c:24s} {v*100:5.1f}%")
    print(f"\nmissingness span   : {(1-fill.max())*100:.0f}%-{(1-fill.min())*100:.0f}%")
    print(f"features >50% missing: {int((fill < 0.5).sum())} of {len(fill)}")
    for c in ("activation_method", "processing_method", "material_family",
              "mineralogy"):
        if c in feats.columns:
            print(f"\n{c}: {feats[c].value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
