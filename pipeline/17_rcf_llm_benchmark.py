"""
Step 17 - RCF LLM benchmark: gpt-oss on a fixed 6-paper held-out set, using the
SAME feature set as RCF's own tuned final model (pipeline_rcf/07_best_model.py):
base+cat+chem = [replacement_pct, age_days, ctrl_strength] + [activation_method,
processing_method, mineralogy, material_family] + [pozzolanic_reactivity] +
[sio2, cao, al2o3, fe2o3, mgo, so3, na2o, k2o, loi, amorphous_pct]. ctrl_strength
is the paper's own 0%-replacement strength at the SAME curing age (RCF's
definition, not a fixed 28-day value -- RCF ages are heterogeneous).

Held-out papers span the control-strength range: 51, 81 (low), 37, 19 (mid),
27, 44 (high). All other framing-C papers are training data. This mirrors
RHA's Part III LLM section, using RCF's own already-established best recipe
(no re-run of the S0-S4 settings ablation, per user's choice: apply the
proven greedy setting -- S1 in 14_llm_repro.py -- directly).

Commands:
  python pipeline/17_rcf_llm_benchmark.py prep            # build train/test CSVs
  python pipeline/17_rcf_llm_benchmark.py run <n_runs>     # call gpt-oss n_runs times
  python pipeline/17_rcf_llm_benchmark.py xgb              # tuned XGBoost on same split
  python pipeline/17_rcf_llm_benchmark.py score            # aggregate gpt-oss runs
"""
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
RCF_OUT = HERE.parent / "pipeline_rcf" / "out"
OUT = HERE / "out_rcf_llm"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:120b-cloud"

CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o", "loi", "amorphous_pct"]
NOMINAL = ["activation_method", "processing_method", "mineralogy", "material_family"]
ORDINAL = ["pozzolanic_reactivity"]
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
FEATS = BASE + NOMINAL + ORDINAL + CHEM  # "base+cat+chem", the tuned winner
HELD_OUT = [51, 81, 37, 19, 27, 44]
GREEDY = {"temperature": 0, "seed": 0, "top_k": 1, "top_p": 1.0,
          "repeat_penalty": 1.0, "num_predict": 16384}
BEST_XGB = dict(max_depth=2, learning_rate=0.1, n_estimators=500,
                min_child_weight=5, reg_lambda=3.0, subsample=0.8,
                colsample_bytree=0.9, random_state=42, n_jobs=-1)
MONO = ["age_days", "ctrl_strength"]


def framing_c():
    tf = pd.read_csv(RCF_OUT / "train_features.csv")
    df = tf[tf.age_known].reset_index(drop=True) if "age_known" in tf.columns else tf
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"], as_index=False)
            ["strength_mpa"].mean().rename(columns={"strength_mpa": "ctrl_strength"}))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return d[d.replacement_pct > 0].reset_index(drop=True)


def to_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().strip()


def prep():
    d = framing_c()
    print(f"framing-C table: {len(d)} rows, {d.paper_id.nunique()} papers")
    test = d[d.paper_id.isin(HELD_OUT)].copy()
    train = d[~d.paper_id.isin(HELD_OUT)].copy()
    # RCF has genuine duplicate (paper_id, replacement_pct, age_days) rows --
    # repeat specimens, e.g. paper 44 has 3 samples at 10% replacement/7 days.
    # Fix a row order ONCE here and carry it via row_id everywhere downstream
    # (prompt query order, ground truth, XGBoost predictions) instead of
    # re-joining on (paper, replacement, age), which is not a unique key.
    test = test.sort_values(["paper_id", "replacement_pct", "age_days"]).reset_index(drop=True)
    test.insert(0, "row_id", range(len(test)))
    n_dup_keys = len(test) - test.groupby(["paper_id", "replacement_pct", "age_days"]).ngroups
    print(f"held out: {len(test)} rows across {test.paper_id.nunique()} papers "
          f"({n_dup_keys} are duplicate-key repeat specimens); "
          f"train: {len(train)} rows across {train.paper_id.nunique()} papers")
    for pid in HELD_OUT:
        sub = test[test.paper_id == pid]
        print(f"  paper {pid}: n={len(sub)} ctrl={sub.ctrl_strength.iloc[0]:.1f} "
              f"strength {sub.strength_mpa.min():.1f}-{sub.strength_mpa.max():.1f}")

    train_cols = ["paper_id"] + FEATS + ["strength_mpa"]
    train[train_cols].to_csv(OUT / "train.csv", index=False)
    test_cols = ["row_id", "paper_id"] + FEATS
    test[test_cols].to_csv(OUT / "test.csv", index=False)
    test[["row_id", "paper_id"] + BASE + ["strength_mpa"]].to_csv(OUT / "ground_truth.csv", index=False)
    print(f"saved -> {OUT}")


def build_prompt(train_df, test_df):
    train_csv = to_csv(train_df.sort_values(["paper_id", "replacement_pct", "age_days"])
                       .rename(columns={"strength_mpa": "compressive_strength_mpa"}))
    tq = test_df.sort_values("row_id").reset_index(drop=True)
    lines = []
    for i, r in tq.iterrows():
        row = {}
        for c in ["paper_id"] + FEATS:
            v = r[c]
            if pd.isna(v):
                continue
            row[c] = round(v, 3) if isinstance(v, float) else v
        lines.append(f"Query {i+1}: {json.dumps(row)}")
    n_train, n_q = len(train_df), len(tq)
    prompt = f"""You are a concrete materials scientist predicting the compressive strength of
recycled-concrete-fines (RCF) blended concrete and mortar from published literature data.

TRAINING DATA
Below are {n_train} measured data points from other papers, as CSV with a header row.
Each row is one (paper, RCF replacement %, curing age) combination. ctrl_strength is
that same paper's own 0% RCF (plain cement) strength AT THE SAME CURING AGE as the row.
Other columns describe the RCF powder: its chemistry (sio2, cao, al2o3, fe2o3, mgo, so3,
na2o, k2o, loi, amorphous_pct, as percentages), how it was processed and activated
(activation_method, processing_method), its mineralogy and material family, and its
pozzolanic reactivity. compressive_strength_mpa is the measured strength of the blend.
Learn how these inputs relate to compressive_strength_mpa.
--- BEGIN TRAINING CSV ---
{train_csv}
--- END TRAINING CSV ---

TASK
Predict compressive_strength_mpa for the {n_q} query rows below. They come from papers
that are NOT in the training data. Each row states its paper_id; rows with the same
paper_id share one mix design. ctrl_strength is that paper's own 0% RCF strength at that
SAME curing age -- use it as that paper's baseline strength level at that age.

{chr(10).join(lines)}

OUTPUT FORMAT
Return only a JSON array of exactly {n_q} numbers: one predicted compressive strength in
MPa per query, in query order (Query 1 first). No units, no text, no code fence.
Example for three queries: [12.3, 45.6, 30.1]"""
    return prompt, tq


def extract(content, n):
    text = re.sub(r"```(?:json)?", "", content)
    for m in reversed(re.findall(r"\[[^\[\]]*\]", text, flags=re.S)):
        try:
            arr = json.loads(m)
        except Exception:
            continue
        if isinstance(arr, list) and len(arr) == n and all(isinstance(x, (int, float)) for x in arr):
            return [float(x) for x in arr]
    return None


def call(prompt, n):
    payload = {"model": MODEL, "stream": False,
               "messages": [{"role": "user", "content": prompt}],
               "options": dict(GREEDY, num_ctx=65536)}
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d["message"].get("content", ""), time.time() - t0


def run(n_runs):
    train = pd.read_csv(OUT / "train.csv")
    test = pd.read_csv(OUT / "test.csv")
    prompt, tq = build_prompt(train, test)
    print(f"prompt chars: {len(prompt)}  n_test: {len(tq)}")
    for i in range(n_runs):
        f = OUT / f"run{i}.json"
        if f.exists():
            print(f"run {i}: cached")
            continue
        try:
            content, dt = call(prompt, len(tq))
        except Exception as e:
            print(f"run {i}: ERROR {e}")
            continue
        preds = extract(content, len(tq))
        row_ids = tq["row_id"].tolist()
        rec = dict(run=i, elapsed_s=round(dt, 1), row_ids=row_ids, preds=preds, raw=content)
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"run {i}: {'OK' if preds else 'INVALID'} ({dt:.0f}s)")


def _score_vec(row_ids, preds, gt):
    gtm = gt.set_index("row_id")["strength_mpa"]
    true = gtm.loc[row_ids].values
    return np.asarray(true), np.asarray(preds)


def score():
    gt = pd.read_csv(OUT / "ground_truth.csv")
    runs = []
    for f in sorted(OUT.glob("run*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec["preds"] and len(rec["preds"]) == len(rec["row_ids"]):
            runs.append(rec)
    print(f"valid runs: {len(runs)} / {len(list(OUT.glob('run*.json')))}")
    stats = []
    for rec in runs:
        y, p = _score_vec(rec["row_ids"], rec["preds"], gt)
        stats.append(dict(run=rec["run"], r2=r2_score(y, p), rmse=mean_squared_error(y, p) ** 0.5,
                          mae=mean_absolute_error(y, p), bias=float((p - y).mean())))
    s = pd.DataFrame(stats)
    s.to_csv(OUT / "single_run_stats.csv", index=False)
    print("\nSingle-run stats:")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nR2   = {s.r2.mean():.3f} +/- {s.r2.std():.3f}  (range {s.r2.min():.3f} to {s.r2.max():.3f})")
    print(f"RMSE = {s.rmse.mean():.2f} +/- {s.rmse.std():.2f}")
    print(f"MAE  = {s.mae.mean():.2f} +/- {s.mae.std():.2f}")

    blocks = [runs[i:i + 5] for i in range(0, len(runs) - len(runs) % 5, 5)]
    for b, blk in enumerate(blocks):
        row_ids = blk[0]["row_ids"]
        P = np.array([r["preds"] for r in blk])
        med = np.median(P, axis=0)
        y, _ = _score_vec(row_ids, med, gt)
        print(f"\nMedian of {len(blk)} runs (block {b}, runs {[r['run'] for r in blk]}):")
        print(f"  R2={r2_score(y, med):.3f}  RMSE={mean_squared_error(y, med)**0.5:.2f}  "
              f"MAE={mean_absolute_error(y, med):.2f}  bias={(med-y).mean():+.2f}")
    if len(blocks) > 1:
        meds = []
        for blk in blocks:
            P = np.array([r["preds"] for r in blk])
            meds.append(np.median(P, axis=0))
        M = np.vstack(meds)
        print(f"\nAgreement between {len(blocks)} block medians: "
              f"median SD {np.median(M.std(axis=0)):.2f} MPa, max {M.std(axis=0).max():.2f} MPa")

    # by paper
    print("\nBy paper (median-of-block-0 predictions):")
    if blocks:
        P = np.array([r["preds"] for r in blocks[0]])
        med = np.median(P, axis=0)
        row_ids = blocks[0][0]["row_ids"]
        d = pd.DataFrame({"row_id": row_ids, "pred": med})
        d = d.merge(gt, on="row_id")
        for pid, g in d.groupby("paper_id"):
            print(f"  paper {pid}: n={len(g)} MAE={mean_absolute_error(g.strength_mpa, g.pred):.2f} "
                  f"bias={(g.pred-g.strength_mpa).mean():+.2f}")


def xgb_direct():
    import sys as _s
    _s.path.insert(0, str(HERE.parent / "pipeline_rcf"))
    from common import build_matrix
    from xgboost import XGBRegressor
    train = pd.read_csv(OUT / "train.csv")
    test = pd.read_csv(OUT / "test.csv")
    gt = pd.read_csv(OUT / "ground_truth.csv")
    Xtr = build_matrix(train, FEATS)
    Xte = build_matrix(test, FEATS).reindex(columns=Xtr.columns, fill_value=0.0)
    order = list(Xtr.columns)
    kw = dict(BEST_XGB)
    kw["monotone_constraints"] = tuple(1 if c in MONO else 0 for c in order)
    m = XGBRegressor(**kw)
    m.fit(Xtr, train.strength_mpa.values)
    pred = m.predict(Xte)
    out = pd.DataFrame({"row_id": test["row_id"].values, "pred": pred})
    out = out.merge(gt, on="row_id")
    y, p = out.strength_mpa.values, out.pred.values
    print(f"XGBoost (tuned final model) on the 6 held-out papers:")
    print(f"  n={len(out)}  R2={r2_score(y, p):.3f}  RMSE={mean_squared_error(y, p)**0.5:.2f}  "
          f"MAE={mean_absolute_error(y, p):.2f}  bias={(p-y).mean():+.2f}")
    out.to_csv(OUT / "xgb_predictions.csv", index=False)
    for pid, g in out.groupby("paper_id"):
        print(f"  paper {pid}: n={len(g)} MAE={mean_absolute_error(g.strength_mpa, g.pred):.2f} "
              f"bias={(g.pred-g.strength_mpa).mean():+.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if cmd == "prep":
        prep()
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    elif cmd == "score":
        score()
    elif cmd == "xgb":
        xgb_direct()
