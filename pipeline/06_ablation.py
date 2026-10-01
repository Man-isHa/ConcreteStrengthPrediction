"""
Step 6 - Comprehensive ablation study.

Five studies, each via repeated grouped hold-out (GroupShuffleSplit x N_REPEATS,
30% papers held out; no paper in both train & test). Results -> out/ablation.csv.

  S1 Target framing   : A absolute-all | B relative-to-control | C blend|control
  S2 Model            : mean, ridge, random-forest, HistGB, XGBoost
  S3 Feature set      : base -> +chem/+fineness/+process/+categorical -> all -> +text
  S4 Imputation       : native-NaN vs median / KNN / MICE (tree vs linear)
  S5 Physics priors   : plain | log-target | monotone(age,control) | log+monotone

Run: python pipeline/06_ablation.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
N_REPEATS = 40
RNG = 42

CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o", "loi", "amorphous_pct"]
FINE = ["mean_particle_um", "specific_surface", "bet_surface", "blaine", "specific_gravity"]
PROC = ["wb_ratio", "burn_temp_c", "burn_duration", "grind_duration"]
NOMINAL = ["activation_method", "mineralogy", "processing_method"]
ORDINAL = ["pozzolanic_reactivity"]


def xgb_available():
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- data prep
def load():
    df = pd.read_csv(OUT / "train_features.csv")
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"])["strength_mpa"].mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    d["rel_ctrl"] = d["strength_mpa"] / d["ctrl_strength"]
    dpos = d[d.replacement_pct > 0].copy()
    return df, dpos


def text_features():
    raw = pd.read_csv(OUT / "papers_wide.csv")
    tcols = [c for c in raw.columns if c.endswith("::Extracted Value")
             and any(k in c for k in ["Pozzolanic", "Activation", "XRD",
                                      "Processing", "Material Type", "Mineral"])]
    corpus = raw[tcols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    ids = raw["paper_id"].values
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD, PCA
    M = TfidfVectorizer(max_features=300, stop_words="english").fit_transform(corpus)
    tfidf = PCA(8, random_state=RNG).fit_transform(TruncatedSVD(8, random_state=RNG).fit_transform(M))
    tfidf = pd.DataFrame(tfidf, columns=[f"tfidf_{i}" for i in range(8)]).assign(paper_id=ids)
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("all-MiniLM-L6-v2").encode(corpus, show_progress_bar=False)
        emb = PCA(8, random_state=RNG).fit_transform(emb)
        minilm = pd.DataFrame(emb, columns=[f"minilm_{i}" for i in range(8)]).assign(paper_id=ids)
    except Exception:
        minilm = None
    return tfidf, minilm


def onehot(df):
    parts = []
    for c in NOMINAL:
        if c in df.columns:
            parts.append(pd.get_dummies(df[c].astype("object"), prefix=c, dummy_na=True).astype(float))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


# ---------------------------------------------------------------- estimators
def make_model(kind, monotone_cols=None, feature_order=None):
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    if kind == "ridge":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])
    if kind == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("m", RandomForestRegressor(n_estimators=400, max_depth=6,
                                                     min_samples_leaf=3, random_state=RNG, n_jobs=-1))])
    if kind == "histgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400,
                                             l2_regularization=1.0, random_state=RNG)
    if kind.startswith("xgb"):
        from xgboost import XGBRegressor
        kw = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                  random_state=RNG, n_jobs=-1)
        if monotone_cols and feature_order:
            mc = tuple(1 if c in monotone_cols else 0 for c in feature_order)
            kw["monotone_constraints"] = mc
        return XGBRegressor(**kw)
    raise ValueError(kind)


def impute_matrix(X, how):
    """Return an imputer transformer for tree models when explicitly requested."""
    if how == "native":
        return None
    if how == "median":
        return SimpleImputer(strategy="median")
    if how == "knn":
        return KNNImputer(n_neighbors=5)
    if how == "mice":
        return IterativeImputer(random_state=RNG, max_iter=10, sample_posterior=False)
    raise ValueError(how)


# ---------------------------------------------------------------- evaluation
def evaluate(data, feature_cols, target, model_kind, *, log_target=False,
             monotone_cols=None, impute="native"):
    numeric_cols = [c for c in feature_cols if c not in NOMINAL and c in data.columns]
    X = data[numeric_cols].copy()
    if set(NOMINAL) & set(feature_cols):  # add one-hot only when categoricals requested
        X = pd.concat([X, onehot(data)], axis=1)
    y = data[target].values
    groups = data["paper_id"].values
    order = list(X.columns)
    r2s, rmses, maes = [], [], []
    for seed in range(N_REPEATS):
        tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, groups))
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
        imp = impute_matrix(X, impute) if model_kind.startswith(("xgb", "histgb")) else None
        try:
            if imp is not None:
                imp.fit(Xtr)
                Xtr = pd.DataFrame(imp.transform(Xtr), columns=order, index=Xtr.index)
                Xte = pd.DataFrame(imp.transform(Xte), columns=order, index=Xte.index)
            m = make_model(model_kind, monotone_cols, order)
            yt = np.log1p(ytr) if log_target else ytr
            m.fit(Xtr, yt)
            p = m.predict(Xte)
            if log_target:
                p = np.expm1(p)
        except Exception:
            continue
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        maes.append(mean_absolute_error(y[te], p))
    if not r2s:
        return None
    return dict(r2_mean=np.mean(r2s), r2_std=np.std(r2s),
                rmse_mean=np.mean(rmses), mae_mean=np.mean(maes),
                n_rows=len(data), n_papers=data.paper_id.nunique(), repeats=len(r2s))


def row(study, name, res, **extra):
    r = dict(study=study, config=name, **extra)
    r.update(res or {})
    return r


def main():
    df, dpos = load()
    tfidf, minilm = text_features()
    dpos = dpos.merge(tfidf, on="paper_id", how="left")
    if minilm is not None:
        dpos = dpos.merge(minilm, on="paper_id", how="left")
    df = df.merge(tfidf, on="paper_id", how="left")

    base_noctrl = ["replacement_pct", "age_days"]
    base = base_noctrl + ["ctrl_strength"]
    structured_num = base + CHEM + FINE + PROC + ORDINAL
    structured = structured_num + NOMINAL  # full structured incl. one-hot categoricals
    xgb = "xgb" if xgb_available() else "histgb"
    results = []

    # S1 Framing (model=xgb, all structured features)
    results.append(row("S1_framing", "A: absolute, all papers",
        evaluate(df, base_noctrl + CHEM + FINE + PROC + ORDINAL + NOMINAL, "strength_mpa", xgb)))
    results.append(row("S1_framing", "B: relative-to-control",
        evaluate(dpos, base_noctrl + CHEM + FINE + PROC + ORDINAL + NOMINAL, "rel_ctrl", xgb)))
    results.append(row("S1_framing", "C: blend | control strength",
        evaluate(dpos, structured, "strength_mpa", xgb)))

    # S2 Model (framing C, all structured features)
    for k in ["mean", "ridge", "rf", "histgb"] + (["xgb"] if xgb_available() else []):
        results.append(row("S2_model", k, evaluate(dpos, structured, "strength_mpa", k)))

    # S3 Feature set (framing C, model=xgb)
    sets = {
        "base [repl,age,control]": base,
        "base+chem": base + CHEM,
        "base+fineness": base + FINE,
        "base+process": base + PROC,
        "base+categorical": base + ORDINAL + NOMINAL,
        "base+all numeric": base + CHEM + FINE + PROC,
        "all structured": structured + NOMINAL,
        "all + TF-IDF text": structured + NOMINAL + [f"tfidf_{i}" for i in range(8)],
    }
    if minilm is not None:
        sets["all + MiniLM text"] = structured + NOMINAL + [f"minilm_{i}" for i in range(8)]
    for name, cols in sets.items():
        results.append(row("S3_features", name, evaluate(dpos, cols, "strength_mpa", xgb)))

    # S4 Imputation (framing C, all numeric structured)
    numeric_struct = base + CHEM + FINE + PROC + ORDINAL
    for how in ["native", "median", "knn", "mice"]:
        results.append(row("S4_imputation", f"{xgb}+{how}",
            evaluate(dpos, numeric_struct, "strength_mpa", xgb, impute=how)))
    for how in ["median", "knn", "mice"]:
        results.append(row("S4_imputation", f"ridge+{how}",
            evaluate(dpos, numeric_struct, "strength_mpa", "ridge")))  # ridge imputes median internally
        break  # ridge shown once (its pipeline uses median); knn/mice covered by tree study

    # S5 Physics priors (framing C, model=xgb, base features)
    results.append(row("S5_priors", "plain", evaluate(dpos, base, "strength_mpa", xgb)))
    results.append(row("S5_priors", "log-target",
        evaluate(dpos, base, "strength_mpa", xgb, log_target=True)))
    if xgb_available():
        results.append(row("S5_priors", "monotone(age,control)",
            evaluate(dpos, base, "strength_mpa", xgb, monotone_cols={"age_days", "ctrl_strength"})))
        results.append(row("S5_priors", "log+monotone",
            evaluate(dpos, base, "strength_mpa", xgb, log_target=True,
                     monotone_cols={"age_days", "ctrl_strength"})))

    res = pd.DataFrame(results)
    res.to_csv(OUT / "ablation.csv", index=False, encoding="utf-8")
    pd.set_option("display.width", 140, "display.max_columns", 20)
    for st in res.study.unique():
        sub = res[res.study == st].sort_values("r2_mean", ascending=False)
        print(f"\n=== {st} ===")
        print(sub[["config", "r2_mean", "r2_std", "rmse_mean", "mae_mean",
                   "n_rows", "n_papers"]].to_string(index=False,
                   float_format=lambda x: f"{x:.3f}"))
    print("\nsaved -> out/ablation.csv")


if __name__ == "__main__":
    main()
