"""
Step 1 - load and clean Master_sheet_RCF.xlsx.

Flattens Sheet1's four-row hierarchical header into "<Feature>::<subfield>",
keeps only the predictive subfields, normalises the sheet's "NULL" strings onto
real NaNs, and assigns a stable paper_id that the Experimental Matrix sheet is
joined against.

Run: python pipeline_rcf/01_load_clean.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, XLSX, null_norm  # noqa: E402

META_COLS = ["Paper Title", "Authors", "Journal Name", "Publication Year",
             "DOI (Digital Object Identifier)"]
N_META = 5
SUBFIELDS = ["Extracted Value", "Unit", "Source Type", "Exact Evidence"]
KEEP_SUBFIELDS = {"Extracted Value", "Unit"}


def load_sheet1():
    raw = pd.read_excel(XLSX, sheet_name="Sheet1", header=None, dtype=str)
    # rows 0-3 are the header band; the data block starts at row 4
    feat_row = raw.iloc[2].tolist()
    sub_row = raw.iloc[3].tolist()

    names = []
    current = None
    for i, (feat, sub) in enumerate(zip(feat_row, sub_row)):
        if i < N_META:
            names.append(META_COLS[i] if i < len(META_COLS) else f"meta_{i}")
            continue
        # the feature name sits only on the first column of each quartet
        if isinstance(feat, str) and feat.strip():
            current = feat.strip()
        sub = (sub or "").strip() if isinstance(sub, str) else ""
        names.append(f"{current}::{sub}")

    df = raw.iloc[4:].copy()
    df.columns = names

    # rows 110+ of the workbook are empty stubs, not spacer rows
    body = [c for c in df.columns if c not in META_COLS]
    df = df[df[body].notna().any(axis=1) | df["Paper Title"].notna()]
    df = df[df["Paper Title"].notna()].reset_index(drop=True)

    # drop provenance columns
    keep = [c for c in df.columns
            if c in META_COLS or c.split("::")[-1] in KEEP_SUBFIELDS]
    df = df[keep]

    for c in df.columns:
        df[c] = null_norm(df[c])
    return df


def assign_paper_ids(df):
    """One id per paper, deduplicating case-insensitively on title.

    The sheet holds the same paper twice under different capitalisation; DOI is
    the cleaner key (103 distinct, no duplicates) but two rows have none, so the
    lowercased title is the fallback.
    """
    title_key = df["Paper Title"].str.strip().str.lower()
    doi_key = df["DOI (Digital Object Identifier)"].str.strip().str.lower()
    key = doi_key.where(doi_key.notna(), title_key)
    # a duplicated title with two DOIs would defeat the dedup, so fold on title
    key = title_key

    df = df.copy()
    df["_key"] = key
    first = df.drop_duplicates("_key", keep="first").reset_index(drop=True)
    first.insert(0, "paper_id", range(len(first)))
    id_map = dict(zip(first["_key"], first["paper_id"]))
    n_dupes = len(df) - len(first)
    return first.drop(columns="_key"), id_map, n_dupes


def load_matrix(id_map):
    mx = pd.read_excel(XLSX, sheet_name="Experimental Matrix", dtype=str)
    mx.columns = [str(c).strip() for c in mx.columns]
    for c in mx.columns:
        mx[c] = null_norm(mx[c])
    mx = mx[mx["Paper Title"].notna()].reset_index(drop=True)
    key = mx["Paper Title"].str.strip().str.lower()
    mx.insert(0, "paper_id", key.map(id_map))
    unmatched = mx[mx.paper_id.isna()]
    mx = mx[mx.paper_id.notna()].copy()
    mx["paper_id"] = mx["paper_id"].astype(int)
    return mx, unmatched


def main():
    wide = load_sheet1()
    n_raw = len(wide)
    wide, id_map, n_dupes = assign_paper_ids(wide)
    mx, unmatched = load_matrix(id_map)

    wide.to_csv(OUT / "papers_wide.csv", index=False, encoding="utf-8")
    mx.to_csv(OUT / "mix_matrix.csv", index=False, encoding="utf-8")

    print(f"Sheet1            : {n_raw} data rows -> {len(wide)} unique papers "
          f"({n_dupes} case-insensitive duplicate title(s) folded)")
    print(f"                    {wide.shape[1]} columns after dropping provenance")
    print(f"Experimental Matrix: {len(mx)} mix rows over "
          f"{mx.paper_id.nunique()} papers")
    if len(unmatched):
        print(f"  {len(unmatched)} matrix rows had no Sheet1 match "
              f"(titles: {sorted(unmatched['Paper Title'].dropna().unique())[:3]})")

    in_sheet1_only = set(wide.paper_id) - set(mx.paper_id)
    if in_sheet1_only:
        titles = wide.loc[wide.paper_id.isin(in_sheet1_only), "Paper Title"]
        print(f"  {len(in_sheet1_only)} paper(s) in Sheet1 but not the matrix:")
        for t in titles:
            print(f"    - {t[:90]}")

    cs = wide["Compressive Strength::Extracted Value"]
    print(f"Compressive Strength cells non-null: {cs.notna().sum()} / {len(wide)}")
    print(f"Units seen: {wide['Compressive Strength::Unit'].dropna().unique().tolist()}")


if __name__ == "__main__":
    main()
