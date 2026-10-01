"""Assemble the delivery copies of the cleaned RHA and RCF datasets.

Reads the pipeline outputs (already produced by pipeline/ and pipeline_rcf/)
and writes a single self-contained `clean_datasets/` folder: per-measurement
tables, per-paper tables, paper metadata, and an Excel workbook holding all of
them. Adds two convenience columns the pipelines compute internally but never
persisted: the paper's own control (0%) strength at the same age, and the
relative strength w.r.t. that control.
"""
import os
import re

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "datasets", "clean_datasets")
os.makedirs(OUT, exist_ok=True)

META = ["paper_id", "Paper Title", "Authors", "Journal Name",
        "Publication Year", "DOI (Digital Object Identifier)"]
META_RENAME = {"Paper Title": "paper_title", "Authors": "authors",
               "Journal Name": "journal", "Publication Year": "year",
               "DOI (Digital Object Identifier)": "doi"}


def add_control(df):
    """Attach the paper's own 0%-replacement strength at the same curing age."""
    ctrl = (df[df["replacement_pct"] == 0]
            .groupby(["paper_id", "age_days"])["strength_mpa"].mean()
            .rename("ctrl_strength_mpa").reset_index())
    df = df.merge(ctrl, on=["paper_id", "age_days"], how="left")
    df["rel_strength"] = df["strength_mpa"] / df["ctrl_strength_mpa"]
    return df


def export(tag, pipe):
    meas = pd.read_csv(os.path.join(BASE, pipe, "out", "train_features.csv"))
    papers = pd.read_csv(os.path.join(BASE, pipe, "out", "paper_level.csv"))
    wide = pd.read_csv(os.path.join(BASE, pipe, "out", "papers_wide.csv"),
                       usecols=META).rename(columns=META_RENAME)

    meas = add_control(meas)
    counts = meas.groupby("paper_id").size().rename("n_measurements")
    papers = papers.merge(counts, on="paper_id", how="left")
    papers["n_measurements"] = papers["n_measurements"].fillna(0).astype(int)
    papers = wide.merge(papers, on="paper_id", how="right")

    # Papers whose Material Type is an aggregate/sand substitution AND that
    # declare no cement replacement level: the parser's scope filter keys on
    # the *levels* cell, so a NaN there lets them through even though their
    # "replacement %" means replaced aggregate, not replaced cement.
    AGG = re.compile(r"aggregate|\bsand\b|\bfra\b|\bfrca\b", re.I)
    mat = os.path.join(BASE, pipe, "out", "papers_wide.csv")
    wide_full = pd.read_csv(mat)
    mcol, lcol = "Material Type::Extracted Value", "Cement Replacement Level(s)::Extracted Value"
    if mcol in wide_full.columns and lcol in wide_full.columns:
        susp = wide_full[wide_full[mcol].astype(str).str.contains(AGG)
                         & wide_full[lcol].isna()]["paper_id"]
        papers["aggregate_replacement_suspect"] = papers["paper_id"].isin(set(susp))
        meas["aggregate_replacement_suspect"] = meas["paper_id"].isin(set(susp))

    excl_path = os.path.join(BASE, pipe, "out", "excluded_papers.csv")
    if os.path.exists(excl_path):
        excl = set(pd.read_csv(excl_path)["paper_id"])
        papers["in_scope"] = ~papers["paper_id"].isin(excl)

    files = {
        f"{tag}_measurements.csv": meas,
        f"{tag}_papers.csv": papers,
    }
    for name, df in files.items():
        df.to_csv(os.path.join(OUT, name), index=False)
        print(f"{name:28s} {df.shape[0]:5d} rows x {df.shape[1]:3d} cols")
    return files


sheets = {}
for tag, pipe in [("RHA", "pipeline"), ("RCF", "pipeline_rcf")]:
    print(f"\n--- {tag}")
    for name, df in export(tag, pipe).items():
        sheets[name.replace(".csv", "")] = df

xlsx = os.path.join(OUT, "cleaned_datasets.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    for name, df in sheets.items():
        df.to_excel(w, sheet_name=name, index=False)
print(f"\nwrote {xlsx}")
