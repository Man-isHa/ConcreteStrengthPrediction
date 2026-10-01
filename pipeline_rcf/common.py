"""
Shared helpers for the RCF pipeline.

The RHA pipeline (../pipeline/) duplicates its path block, encoders, control-merge
and CV loop across four scripts. Nothing there needs to change, so this pipeline
keeps them in one place instead.

Not runnable on its own; imported by 01..08.
"""
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XLSX = ROOT / "datasets" / "raw" / "Master_sheet_RCF.xlsx"
OUT = HERE / "out"
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

RNG = 42
N_REPEATS = 40

# ------------------------------------------------------------------ null handling
# RCF stores missing values as the literal string "NULL", frequently with a
# trailing parenthetical justification, e.g.
#   NULL (Note: The paper reports water/cement (w/c) ratio ...)
# An equality test against "NULL" would keep those as data.
NA_EXACT = {"", "-", "--", "—", "N/A", "NA", "NAN", "NONE", "NIL", "?"}


def is_null(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    if s.upper() in NA_EXACT:
        return True
    return s.upper().startswith("NULL")


def null_norm(s: pd.Series) -> pd.Series:
    """Map every RCF null spelling onto a real NaN."""
    return s.map(lambda x: np.nan if is_null(x) else str(x).strip())


# ------------------------------------------------------------------ numbers
NUM = r"\d+(?:\.\d+)?"          # unsigned on purpose: a leading '-' is a separator
NUM_RE = re.compile(NUM)


def parse_numeric_mean(cell):
    """Mean of every number in a cell.

    Handles multi-sample entries like "94.6 (RCP-A), 93.7 (RCP-B)" and ranges.
    Percent signs, units and sample labels are ignored.
    """
    if is_null(cell):
        return np.nan
    nums = [float(v) for v in NUM_RE.findall(str(cell))]
    return float(np.mean(nums)) if nums else np.nan


# ------------------------------------------------------------------ feature groups
CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o",
        "loi", "amorphous_pct"]
FINE = ["d10", "d50", "d90", "mean_particle_um", "specific_surface",
        "bet_surface", "blaine", "specific_gravity", "bulk_density",
        "water_absorption"]
PROC = ["wb_ratio", "thermal_temp_c", "thermal_duration", "grind_duration",
        "grind_speed"]
NOMINAL = ["activation_method", "processing_method", "mineralogy",
           "material_family"]
ORDINAL = ["pozzolanic_reactivity"]

BASE_NOCTRL = ["replacement_pct", "age_days"]
BASE = BASE_NOCTRL + ["ctrl_strength"]


# ------------------------------------------------------------------ modelling helpers
def onehot(df, cols=None):
    """One-hot the nominal columns present in df, NaN kept as its own level."""
    cols = NOMINAL if cols is None else cols
    parts = []
    for c in cols:
        if c in df.columns:
            parts.append(pd.get_dummies(df[c], prefix=c, dummy_na=True).astype(float))
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def build_matrix(data, feature_cols):
    """Numeric columns as-is + one-hot for any nominal columns requested."""
    numeric_cols = [c for c in feature_cols
                    if c not in NOMINAL and c in data.columns]
    X = data[numeric_cols].copy()
    wanted = [c for c in NOMINAL if c in feature_cols]
    if wanted:
        X = pd.concat([X, onehot(data, wanted)], axis=1)
    return X


def add_control(df):
    """Attach each paper's own 0% control strength at the same curing age.

    Returns only the blend rows (replacement > 0): keeping the control rows would
    make the target trivially equal to the feature.
    """
    ctrl = (df[df.replacement_pct == 0]
            .groupby(["paper_id", "age_days"], as_index=False)["strength_mpa"]
            .mean()
            .rename(columns={"strength_mpa": "ctrl_strength"}))
    merged = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return merged[merged.replacement_pct > 0].reset_index(drop=True)


def make_model(kind, monotone_cols=None, feature_order=None):
    """The candidate roster. `kind` is the ablation config name."""
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                  HistGradientBoostingRegressor,
                                  RandomForestRegressor, StackingRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def imputed(est):
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", est)])

    def scaled(est):
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()), ("m", est)])

    if kind == "mean":
        return DummyRegressor(strategy="mean")
    if kind == "ridge":
        return scaled(Ridge(alpha=1.0))
    if kind == "rf":
        return imputed(RandomForestRegressor(
            n_estimators=400, max_depth=6, min_samples_leaf=3,
            random_state=RNG, n_jobs=-1))
    if kind == "extratrees":
        return imputed(ExtraTreesRegressor(
            n_estimators=400, max_depth=8, min_samples_leaf=3,
            random_state=RNG, n_jobs=-1))
    if kind == "gbr":
        return imputed(GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=RNG))
    if kind == "mlp":
        return scaled(MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-2, learning_rate_init=3e-3,
            max_iter=2000, early_stopping=True, n_iter_no_change=25,
            random_state=RNG))
    if kind == "histgb":
        kw = dict(max_depth=3, learning_rate=0.05, max_iter=400,
                  l2_regularization=1.0, random_state=RNG)
        if monotone_cols and feature_order:
            # sklearn's HistGB supports monotone constraints too, so the
            # physics prior is not an XGBoost-only capability here
            kw["monotonic_cst"] = [1 if c in monotone_cols else 0
                                   for c in feature_order]
        return HistGradientBoostingRegressor(**kw)
    if kind == "stack":
        return imputed(StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(n_estimators=300, max_depth=6,
                                             min_samples_leaf=3,
                                             random_state=RNG, n_jobs=-1)),
                ("xgb", make_model("xgb")),
                ("ridge", Pipeline([("sc", StandardScaler()),
                                    ("m", Ridge(alpha=1.0))])),
            ],
            final_estimator=RidgeCV(),
            cv=3, n_jobs=1))
    if kind.startswith("xgb"):
        from xgboost import XGBRegressor
        kw = dict(n_estimators=400, max_depth=3, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                  reg_lambda=1.0, random_state=RNG, n_jobs=-1)
        if monotone_cols and feature_order:
            kw["monotone_constraints"] = tuple(
                1 if c in monotone_cols else 0 for c in feature_order)
        return XGBRegressor(**kw)
    raise ValueError(kind)


def impute_matrix(how):
    """Imputer for Study 4. Always fit inside a split, never on the full data."""
    from sklearn.impute import KNNImputer, SimpleImputer
    if how == "native":
        return None
    if how == "median":
        return SimpleImputer(strategy="median")
    if how == "knn":
        return KNNImputer(n_neighbors=5)
    if how == "mice":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        return IterativeImputer(random_state=RNG, max_iter=10,
                                sample_posterior=False)
    raise ValueError(how)


def evaluate(data, feature_cols, target, model_kind, *, log_target=False,
             monotone_cols=None, impute="native", n_repeats=N_REPEATS):
    """Repeated grouped hold-out: 30% of *papers* held out, `n_repeats` times.

    A single split is unreliable at this sample size, so R2 is reported as
    mean +/- std over the repeats. No paper ever appears on both sides.
    """
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import GroupShuffleSplit

    X = build_matrix(data, feature_cols)
    y = data[target].values
    groups = data["paper_id"].values
    order = list(X.columns)

    r2s, rmses, maes = [], [], []
    for seed in range(n_repeats):
        try:
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                          .split(X, y, groups))
        except ValueError:
            continue
        if len(te) == 0 or len(tr) == 0:
            continue
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]

        use_imp = model_kind.startswith("xgb") or model_kind.startswith("histgb")
        imp = impute_matrix(impute) if use_imp else None
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
        rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
        maes.append(float(mean_absolute_error(y[te], p)))

    if not r2s:
        return None
    return dict(r2_mean=float(np.mean(r2s)), r2_std=float(np.std(r2s)),
                rmse_mean=float(np.mean(rmses)), mae_mean=float(np.mean(maes)),
                n_rows=len(data), n_papers=int(data.paper_id.nunique()),
                repeats=len(r2s))


def evaluate_train_variant(data, feature_cols, target, model_kind, *,
                           train_filter=None, train_extra=None, collapse=False,
                           monotone_cols=None, n_repeats=N_REPEATS):
    """Like `evaluate`, but varies only what the model is TRAINED on.

    The held-out rows are always the same, so the resulting R2 values are
    directly comparable across variants. Comparing variants that also change the
    evaluation set would compare difficulty, not generalisation.

    train_filter : boolean mask over `data`, restricting the training rows
    train_extra  : extra rows to append to training when their paper is in the
                   training half (never to the test half)
    collapse     : average duplicate (paper, level, age) training rows first
    """
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    from sklearn.model_selection import GroupShuffleSplit

    y = data[target].values
    groups = data["paper_id"].values
    X_all = build_matrix(data, feature_cols)
    order = list(X_all.columns)

    r2s, rmses, maes = [], [], []
    for seed in range(n_repeats):
        try:
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                          .split(X_all, y, groups))
        except ValueError:
            continue
        tr_rows = data.iloc[tr]
        if train_filter is not None:
            tr_rows = tr_rows[train_filter.iloc[tr].values]
        if train_extra is not None and len(train_extra):
            train_papers = set(data.iloc[tr]["paper_id"])
            add = train_extra[train_extra.paper_id.isin(train_papers)]
            tr_rows = pd.concat([tr_rows, add], ignore_index=True)
        if collapse:
            keys = ["paper_id", "replacement_pct", "age_days"]
            num = tr_rows.select_dtypes("number").columns.tolist()
            tr_rows = tr_rows.groupby(keys, as_index=False)[num].mean()
        if len(tr_rows) < 20 or tr_rows[target].nunique() < 3:
            continue

        # absent one-hot levels mean 'not this category', i.e. 0, not unknown
        Xtr = build_matrix(tr_rows, feature_cols).reindex(columns=order,
                                                          fill_value=0.0)
        Xte = X_all.iloc[te]
        try:
            m = make_model(model_kind, monotone_cols, order)
            m.fit(Xtr, tr_rows[target].values)
            p = m.predict(Xte)
        except Exception:
            continue

        r2s.append(r2_score(y[te], p))
        rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
        maes.append(float(mean_absolute_error(y[te], p)))

    if not r2s:
        return None
    return dict(r2_mean=float(np.mean(r2s)), r2_std=float(np.std(r2s)),
                rmse_mean=float(np.mean(rmses)), mae_mean=float(np.mean(maes)),
                n_rows=len(data), n_papers=int(data.paper_id.nunique()),
                repeats=len(r2s))


# ------------------------------------------------------------------ plotting
BLUE = "#2c6fbb"
RED = "#c1584b"
GREY = "#b0b0b0"
LIGHT = "#7aa8d6"


def use_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    return plt
