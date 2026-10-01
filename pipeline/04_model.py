"""
Step 4 - Modeling & evaluation.

Key finding from EDA (see report): absolute compressive strength across papers is
nearly unpredictable (R2 ~ 0) because the data mixes low-strength mortars, normal
concrete and UHPC (1-205 MPa), and each paper's *baseline* level is set by mix
factors (cement grade, aggregate, SP, UHPC vs normal) not captured by the sparse
RHA features. The RHA replacement effect is a *relative* modulation on that
baseline. We therefore evaluate three complementary framings:

  A. Absolute CS, no baseline info      (all papers)      -- documents the ceiling
  B. Relative-to-control CS             (papers w/ 0%)    -- isolates the RHA effect
  C. Absolute CS given control strength (papers w/ 0%)    -- best & most practical

Targets:
  1. Compressive strength  -- framings A/B/C above.
  2. Optimal replacement % -- (a) argmax of framing C over replacement, (b) direct.

Validation: GroupKFold by paper_id. Model: XGBoost if installed else HistGB
(both handle NaN). Embedding ablation runs on framing C (the one that works).

Run: python pipeline/04_model.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

N_REPEATS = 25  # repeated splits: with ~25-44 papers a single split is unreliable

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
RNG = 42

CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o", "loi",
        "amorphous_pct", "specific_gravity", "mean_particle_um",
        "specific_surface", "bet_surface", "blaine", "wb_ratio",
        "burn_temp_c", "burn_duration", "grind_duration", "pozzolanic_reactivity"]
NOMINAL = ["activation_method", "mineralogy", "processing_method"]
report_lines = []


def log(s=""):
    print(s)
    report_lines.append(s)


def make_regressor():
    try:
        from xgboost import XGBRegressor
        return ("xgboost", XGBRegressor(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            random_state=RNG, n_jobs=-1))
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return ("histgb", HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=RNG))


def encode(df, num_cols):
    X = df[[c for c in num_cols if c in df.columns]].copy()
    for c in NOMINAL:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype("object"), prefix=c, dummy_na=True)
            X = pd.concat([X, d.astype(float)], axis=1)
    return X


def group_cv(df, num_cols, target, label):
    """Repeated grouped hold-out (GroupShuffleSplit x N_REPEATS). Reports
    mean +/- std of R2/RMSE/MAE - a single split is unreliable at this n."""
    X = encode(df, num_cols)
    y = df[target].values
    groups = df["paper_id"].values
    name, _ = make_regressor()
    r2s, rmses, maes = [], [], []
    for seed in range(N_REPEATS):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                      .split(X, y, groups))
        try:
            _, m = make_regressor()
            m.fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
        except Exception:
            continue
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
    r2, rmse, mae = np.mean(r2s), np.mean(rmses), np.mean(maes)
    log(f"[{label}]  model={name} n={len(y)} papers={df.paper_id.nunique()} "
        f"repeats={len(r2s)}")
    log(f"    R2={r2:.3f}+/-{np.std(r2s):.3f}  RMSE={rmse:.2f}+/-{np.std(rmses):.2f}"
        f"  MAE={mae:.2f}")
    return dict(rmse=rmse, mae=mae, r2=r2, r2_std=float(np.std(r2s)))


def add_control(df):
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"])["strength_mpa"].mean()
            .rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    d["rel_ctrl"] = d["strength_mpa"] / d["ctrl_strength"]
    return d


def build_text_features():
    raw = pd.read_csv(OUT / "papers_wide.csv")
    tcols = [c for c in raw.columns if c.endswith("::Extracted Value")
             and any(k in c for k in ["Pozzolanic", "Activation", "XRD",
                                      "Processing", "Material Type", "Mineral"])]
    corpus = raw[tcols].fillna("").astype(str).agg(" ".join, axis=1)
    ids = raw["paper_id"].values
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2").encode(corpus.tolist(),
                                                             show_progress_bar=False)
        src = "MiniLM"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        M = TfidfVectorizer(max_features=300, stop_words="english").fit_transform(corpus)
        k = min(8, M.shape[1] - 1)
        emb = TruncatedSVD(n_components=k, random_state=RNG).fit_transform(M)
        src = f"TF-IDF+SVD({k})"
    from sklearn.decomposition import PCA
    k = min(8, np.asarray(emb).shape[1])
    emb = PCA(n_components=k, random_state=RNG).fit_transform(emb)
    cols = [f"txt_{i}" for i in range(k)]
    return src, pd.DataFrame(emb, columns=cols).assign(paper_id=ids), cols


def optimal_argmax(dctrl):
    """Optimum % = argmax over replacement of framing-C predictions at 28d,
    per held-out paper (GroupKFold). Uses the paper's known 28d control strength."""
    paper = pd.read_csv(OUT / "paper_level.csv").set_index("paper_id")
    numC = ["replacement_pct", "age_days", "ctrl_strength"] + CHEM
    X = encode(dctrl, numC)
    y = dctrl["strength_mpa"].values
    groups = dctrl["paper_id"].values
    grid = np.arange(0, 41, 2.5)
    gkf = GroupKFold(min(5, dctrl.paper_id.nunique()))
    preds = {}
    for tr, te in gkf.split(X, y, groups):
        _, m = make_regressor()
        m.fit(X.iloc[tr], y[tr])
        for pid in np.unique(groups[te]):
            rows = dctrl[dctrl.paper_id == pid]
            c28 = rows[rows.age_days == 28]["ctrl_strength"]
            if c28.empty:
                continue
            base = rows.median(numeric_only=True)
            cand = pd.DataFrame([base] * len(grid))
            cand["replacement_pct"] = grid
            cand["age_days"] = 28.0
            cand["ctrl_strength"] = float(c28.iloc[0])
            for c in NOMINAL:
                cand[c] = rows[c].mode().iloc[0] if not rows[c].mode().empty else np.nan
            Xc = encode(cand, numC).reindex(columns=X.columns, fill_value=0)
            preds[pid] = float(grid[int(np.argmax(m.predict(Xc)))])
    ps = pd.Series(preds)
    rep = paper["reported_optimum_pct"].reindex(ps.index)
    mask = rep.notna()
    mae = mean_absolute_error(rep[mask], ps[mask])
    bmae = mean_absolute_error(rep[mask], np.full(mask.sum(), rep[mask].mean()))
    log(f"[Optimal% via CS-argmax (framing C)]  papers={mask.sum()}")
    log(f"    MAE vs reported optimum = {mae:.2f} pts   (mean-baseline {bmae:.2f})")


def optimal_direct(df):
    paper = pd.read_csv(OUT / "paper_level.csv")
    paper = paper[paper.paper_id.isin(df.paper_id.unique())
                  & paper.reported_optimum_pct.notna()].reset_index(drop=True)
    X = encode(paper, CHEM)
    y = paper["reported_optimum_pct"].values
    kf = KFold(min(5, len(paper)), shuffle=True, random_state=RNG)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        _, m = make_regressor()
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
    mae = mean_absolute_error(y, oof)
    bmae = mean_absolute_error(y, np.full(len(y), y.mean()))
    log(f"[Optimal% via direct regressor]  papers={len(y)}")
    log(f"    MAE={mae:.2f} pts   (mean-baseline MAE={bmae:.2f} pts)")


def main():
    df = pd.read_csv(OUT / "train_features.csv")
    log("=" * 70)
    log(f"Modeling table: {df.shape}  papers={df.paper_id.nunique()}")
    log(f"Strength range: {df.strength_mpa.min()}-{df.strength_mpa.max()} MPa "
        f"(mixing mortar/normal/UHPC)")
    log("")

    # Framing A: absolute, no baseline info, ALL papers -> documents the ceiling
    group_cv(df, ["replacement_pct", "age_days"] + CHEM, "strength_mpa",
             "A: absolute CS, all papers, all features")

    d = add_control(df)
    # exclude the 0% control rows from evaluation: there strength==ctrl_strength
    # exactly (target equals a feature) which would trivially inflate scores.
    dpos = d[d.replacement_pct > 0].copy()
    log(f"\n(papers with a 0% control: {d.paper_id.nunique()}; RHA-blend rows "
        f"used for B/C: {len(dpos)})")

    # Framing B: relative-to-control (RHA-blend rows only)
    group_cv(dpos, ["replacement_pct", "age_days"] + CHEM, "rel_ctrl",
             "B: relative-to-control CS")

    # Framing C: absolute given control strength. Compare feature sets to show
    # sparse chem/text features do not add real value at this sample size.
    base = ["replacement_pct", "age_days", "ctrl_strength"]
    chem_small = ["sio2", "amorphous_pct", "mean_particle_um", "wb_ratio", "loi", "cao"]
    log("\n-- Framing C: RHA-blend CS | control strength (feature-set ablation) --")
    mC = group_cv(dpos, base, "strength_mpa", "C1: base [replacement, age, control]")
    group_cv(dpos, base + chem_small, "strength_mpa", "C2: base + chemistry")
    group_cv(dpos, base + CHEM, "strength_mpa", "C3: base + all features")

    # Embedding ablation vs the lean base model
    src, txt, tcols = build_text_features()
    dCpos = dpos.merge(txt, on="paper_id", how="left")
    mCt = group_cv(dCpos, base + tcols, "strength_mpa", f"C4: base + text embeddings [{src}]")
    verdict = "KEEP" if mCt["r2"] > mC["r2"] + mC["r2_std"] else "DROP (no real gain)"
    log(f"    >>> embeddings: R2 {mC['r2']:.3f} -> {mCt['r2']:.3f}  => {verdict}")

    log("")
    optimal_argmax(d)
    optimal_direct(df)

    (OUT / "cv_results.txt").write_text("\n".join(report_lines), encoding="utf-8")
    log("\nSaved metrics -> out/cv_results.txt")


if __name__ == "__main__":
    main()
