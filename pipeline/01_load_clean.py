"""
Step 1 - Load & clean the RHA master sheet.

- Reads the raw CSV (latin-1) with its 4-row hierarchical header.
- Flattens the header into tidy "<Feature>::<subfield>" keys.
- Drops the ~917 blank spacer rows, keeping the ~79 real paper rows.
- Emits `papers_wide.csv`: one row per paper, with a stable `paper_id`,
  the 5 metadata columns, and every feature's Extracted Value + Unit.

Run: python pipeline/01_load_clean.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "datasets" / "raw" / "RHA Master sheet v2.csv"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# Sentinels that should be treated as missing.
NA_TOKENS = ["", "nan", "NaN", "NA", "N/A", "n/a", "None", "none", "-", "--", "—"]

# The four sub-fields each feature is split into.
SUBFIELDS = {"Extracted Value", "Unit", "Source Type", "Exact Evidence"}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(SRC, header=None, dtype=str, encoding="latin-1")


def build_columns(raw: pd.DataFrame):
    """Return a list of tidy column keys aligned to raw's columns.

    Header layout:
      row 0: top category (PAPER METADATA / INPUT / OUTPUT)
      row 1: section (e.g. RHA CHEMICAL PROPERTIES)
      row 2: feature name (e.g. SiO2)  -- forward-filled across its 4 subfields
      row 3: subfield (Extracted Value / Unit / Source Type / Exact Evidence)
    The first 5 columns are paper metadata (single-level).
    """
    feature = raw.iloc[2].ffill()
    subfield = raw.iloc[3]
    meta_names = raw.iloc[1, 0:5].tolist()  # Paper Title, Authors, ...

    cols = []
    for i in range(raw.shape[1]):
        if i < 5:
            cols.append(str(meta_names[i]).strip())
        else:
            f = str(feature.iloc[i]).strip()
            s = str(subfield.iloc[i]).strip()
            cols.append(f"{f}::{s}")
    return cols


def main():
    raw = load_raw()
    print(f"raw shape: {raw.shape}")

    cols = build_columns(raw)
    data = raw.iloc[4:].copy()
    data.columns = cols
    data = data.reset_index(drop=True)

    # Normalize sentinels to NaN everywhere.
    data = data.replace(NA_TOKENS, np.nan)

    # Drop blank spacer rows: keep rows with >=1 non-null in the body (non-metadata) cols.
    body_cols = cols[5:]
    keep_mask = data[body_cols].notna().any(axis=1)
    papers = data[keep_mask].reset_index(drop=True)
    papers.insert(0, "paper_id", range(len(papers)))

    print(f"kept paper rows: {len(papers)} (dropped {len(data) - len(papers)} blank rows)")
    assert len(papers) == papers[body_cols].notna().any(axis=1).sum(), "blank row leaked"

    # Keep metadata + Extracted Value + Unit columns (drop Source Type / Exact Evidence provenance).
    meta_cols = ["paper_id"] + cols[:5]
    value_cols = [c for c in papers.columns if c.endswith("::Extracted Value")]
    unit_cols = [c for c in papers.columns if c.endswith("::Unit")]
    wide = papers[meta_cols + value_cols + unit_cols].copy()

    out_path = OUT / "papers_wide.csv"
    wide.to_csv(out_path, index=False, encoding="utf-8")
    print(f"wrote {out_path}  shape={wide.shape}")
    print(f"  {len(value_cols)} Extracted Value cols, {len(unit_cols)} Unit cols")
    print(f"  unique paper titles: {wide['Paper Title'].nunique()}")


if __name__ == "__main__":
    main()
