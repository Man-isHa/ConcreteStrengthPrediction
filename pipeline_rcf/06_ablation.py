"""
Step 6 - the six-axis ablation study.

S1 framing   : how the target is posed
S2 model     : the candidate roster
S3 features  : which feature groups earn their place
S4 imputation: native NaN handling vs explicit imputers
S5 priors    : monotone constraints and log targets
S6 data      : which rows to train on (RCF-specific - two sources, and a large
               pool of rows whose curing age never resolved)

All numbers are repeated grouped hold-out, 30% of papers held out, 40 repeats.

Run: python pipeline_rcf/06_ablation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BASE, BASE_NOCTRL, CHEM, FINE, NOMINAL, N_REPEATS,  # noqa: E402
                    ORDINAL, OUT, PROC, RNG, add_control, evaluate,
                    evaluate_train_variant)

MIXCOL = ["wb_block", "graph_derived"]
TEXT_SOURCES = ["Material Type", "Source Material", "RCF/RCP Processing Method",
                "Activation Method", "Pozzolanic Reactivity", "XRD Phases",
                "Mineral Composition"]


def row(study, config, res, **extra):
    r = dict(study=study, config=config, **extra)
    r.update(res or {})
    return r


def text_features(paper_ids):
    """Paper-level text embeddings, keyed by paper_id.

    Returns (tfidf_frame, minilm_frame); either may be None.
    """
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    wide = pd.read_csv(OUT / "papers_wide.csv")
    cols = [f"{c}::Extracted Value" for c in TEXT_SOURCES
            if f"{c}::Extracted Value" in wide.columns]
    docs = (wide[cols].fillna("").agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    ids = wide["paper_id"].values

    tf = None
    try:
        mat = TfidfVectorizer(max_features=300, stop_words="english").fit_transform(docs)
        k = min(8, mat.shape[1] - 1, mat.shape[0] - 1)
        red = TruncatedSVD(k, random_state=RNG).fit_transform(mat)
        tf = pd.DataFrame(red, columns=[f"tfidf_{i}" for i in range(k)])
        tf.insert(0, "paper_id", ids)
    except Exception as e:
        print(f"  tfidf unavailable: {e}")

    ml = None
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2").encode(
            docs.tolist(), show_progress_bar=False)
        red = PCA(8, random_state=RNG).fit_transform(emb)
        ml = pd.DataFrame(red, columns=[f"minilm_{i}" for i in range(8)])
        ml.insert(0, "paper_id", ids)
    except Exception as e:
        print(f"  MiniLM unavailable ({type(e).__name__}); TF-IDF only")

    return tf, ml


def main():
    full = pd.read_csv(OUT / "train_features.csv")
    aged = full[full.age_known].reset_index(drop=True)

    structured_num = CHEM + FINE + PROC + ORDINAL
    featsA = BASE_NOCTRL + structured_num + NOMINAL
    featsC = BASE + structured_num + NOMINAL

    d = add_control(aged)
    d = d.copy()
    d["rel_ctrl"] = d.strength_mpa / d.ctrl_strength
    d["graph_derived"] = d["graph_derived"].astype(float)

    rows = []
    xgb = "xgb"

    # ---------------------------------------------------------- S1 framing
    print("S1 framing ...")
    rows.append(row("S1_framing", "A: absolute, all papers",
                    evaluate(aged, featsA, "strength_mpa", xgb)))
    rows.append(row("S1_framing", "B: relative-to-control",
                    evaluate(d, featsA, "rel_ctrl", xgb)))
    rows.append(row("S1_framing", "C: blend | control strength",
                    evaluate(d, featsC, "strength_mpa", xgb)))

    # ---------------------------------------------------------- S2 model
    print("S2 model ...")
    for kind, label in [("mean", "Mean baseline"), ("ridge", "Ridge"),
                        ("mlp", "MLP (64,32)"), ("histgb", "HistGB"),
                        ("gbr", "GradientBoosting"), ("extratrees", "ExtraTrees"),
                        ("rf", "Random Forest"), ("xgb", "XGBoost"),
                        ("stack", "Stack (RF+XGB+Ridge)")]:
        res = evaluate(d, featsC, "strength_mpa", kind)
        rows.append(row("S2_model", label, res))
        print(f"   {label:22s} R2={res['r2_mean']:+.3f}" if res else
              f"   {label:22s} failed")

    # ---------------------------------------------------------- S3 features
    print("S3 features ...")
    tf, ml = text_features(d.paper_id.unique())
    sets = [
        ("base [repl, age, control]", BASE),
        ("base + chemistry", BASE + CHEM),
        ("base + fineness", BASE + FINE),
        ("base + processing", BASE + PROC),
        ("base + categorical", BASE + NOMINAL + ORDINAL),
        ("base + mix descriptors", BASE + MIXCOL),
        ("base + all numeric", BASE + structured_num),
        ("all structured", featsC),
    ]
    for label, cols in sets:
        res = evaluate(d, cols, "strength_mpa", xgb)
        rows.append(row("S3_features", label, res))
        print(f"   {label:26s} R2={res['r2_mean']:+.3f}" if res else
              f"   {label:26s} failed")
    for label, frame in [("all + TF-IDF text", tf), ("all + MiniLM text", ml)]:
        if frame is None:
            continue
        dt = d.merge(frame, on="paper_id", how="left")
        cols = featsC + [c for c in frame.columns if c != "paper_id"]
        res = evaluate(dt, cols, "strength_mpa", xgb)
        rows.append(row("S3_features", label, res))
        print(f"   {label:26s} R2={res['r2_mean']:+.3f}" if res else
              f"   {label:26s} failed")

    # ---------------------------------------------------------- S4 imputation
    print("S4 imputation ...")
    for how, label in [("native", "XGB + native"), ("median", "XGB + median"),
                       ("knn", "XGB + KNN"), ("mice", "XGB + MICE")]:
        res = evaluate(d, featsC, "strength_mpa", xgb, impute=how)
        rows.append(row("S4_imputation", label, res))
        print(f"   {label:16s} R2={res['r2_mean']:+.3f} "
              f"repeats={res['repeats']}" if res else f"   {label} failed")
    rows.append(row("S4_imputation", "Ridge + median",
                    evaluate(d, featsC, "strength_mpa", "ridge")))

    # ---------------------------------------------------------- S5 priors
    print("S5 priors ...")
    mono = ["age_days", "ctrl_strength"]
    rows.append(row("S5_priors", "plain",
                    evaluate(d, BASE, "strength_mpa", xgb)))
    rows.append(row("S5_priors", "log-target",
                    evaluate(d, BASE, "strength_mpa", xgb, log_target=True)))
    rows.append(row("S5_priors", "monotone(age, control)",
                    evaluate(d, BASE, "strength_mpa", xgb, monotone_cols=mono)))
    rows.append(row("S5_priors", "log + monotone",
                    evaluate(d, BASE, "strength_mpa", xgb, log_target=True,
                             monotone_cols=mono)))

    # ---------------------------------------------------------- S6 data
    # Only the TRAINING rows change; the held-out rows are identical in every
    # variant, so these R2 values are comparable to one another.
    print("S6 data ...")
    unaged = full[~full.age_known].copy()
    unaged["age_days"] = 28.0                 # the assumption under test
    dun = add_control(pd.concat([aged, unaged], ignore_index=True))
    dun = dun[dun.index.isin([])] if dun.empty else dun
    extra = dun.merge(d[["paper_id", "replacement_pct", "age_days",
                         "strength_mpa"]].assign(_seen=1),
                      on=["paper_id", "replacement_pct", "age_days",
                          "strength_mpa"], how="left")
    extra = extra[extra._seen.isna()].drop(columns="_seen")

    variants = [
        ("both sources (reference)", dict()),
        ("Sheet1 rows only", dict(train_filter=(d.source == "sheet1"))),
        ("matrix rows only", dict(train_filter=(d.source == "matrix"))),
        ("+ age-unknown rows as 28 d", dict(train_extra=extra)),
        ("duplicate mixes averaged", dict(collapse=True)),
    ]
    for label, kw in variants:
        res = evaluate_train_variant(d, featsC, "strength_mpa", xgb, **kw)
        rows.append(row("S6_data", label, res))
        print(f"   {label:28s} R2={res['r2_mean']:+.3f}" if res else
              f"   {label:28s} failed")

    ab = pd.DataFrame(rows)
    ab.to_csv(OUT / "ablation.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 20)
    print()
    for study in ab.study.unique():
        sub = (ab[ab.study == study].dropna(subset=["r2_mean"])
               .sort_values("r2_mean", ascending=False))
        print(f"===== {study}")
        print(sub[["config", "r2_mean", "r2_std", "rmse_mean", "mae_mean",
                   "n_rows", "n_papers", "repeats"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()


if __name__ == "__main__":
    main()
