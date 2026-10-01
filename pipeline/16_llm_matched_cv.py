"""
Step 16 - gpt-oss on the SAME 40 repeated grouped hold-out splits used for the
XGBoost final model in Part III (13_ml_le60.py / 07_best_model.py), using the
already-established greedy (S1) settings from 14_llm_repro.py.

For each of the 40 splits (identical GroupShuffleSplit(1, test_size=0.3,
random_state=seed) on the same framing-C, <=60 MPa table, 308 rows / 21
papers), gpt-oss sees only the three base inputs XGBoost uses --
replacement_pct, age_days, ctrl_strength (the paper's control strength at the
SAME age as the row, not a fixed 28-day value) -- and predicts strength_mpa
for the held-out rows. One call per split (not 5), to match "the same 40
grouped hold-out" one-fit-per-split protocol XGBoost uses.

Run: python pipeline/16_llm_matched_cv.py [n_splits]
"""
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_llm_cv"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:120b-cloud"
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
GREEDY = {"temperature": 0, "seed": 0, "top_k": 1, "top_p": 1.0,
          "repeat_penalty": 1.0, "num_predict": 16384}


def add_ctrl(df):
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"])["strength_mpa"]
            .mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return d[d.replacement_pct > 0].copy()


def to_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().strip()


def build_prompt(train_df, test_df):
    train_csv = to_csv(train_df[["paper_id"] + BASE + ["strength_mpa"]]
                       .sort_values(["paper_id", "replacement_pct", "age_days"]))
    tq = test_df.sort_values(["paper_id", "replacement_pct", "age_days"]).reset_index(drop=True)
    q_lines = "\n".join(
        f"Query {i+1}: {json.dumps({'paper_id': r.paper_id, 'replacement_pct': r.replacement_pct, 'age_days': r.age_days, 'ctrl_strength_same_age_mpa': round(r.ctrl_strength, 3)})}"
        for i, r in tq.iterrows())
    n_train, n_q = len(train_df), len(tq)
    prompt = f"""You are a concrete materials scientist predicting the compressive strength of
rice-husk-ash (RHA) blended concrete and mortar from published literature data.

TRAINING DATA
Below are {n_train} measured data points from other papers, as CSV with a header row.
Each row is one (paper, RHA replacement %, curing age) combination. ctrl_strength is
that same paper's own 0% RHA (plain cement) strength AT THE SAME CURING AGE as the row.
compressive_strength_mpa is the measured strength of the RHA blend. Learn how
replacement_pct, age_days and ctrl_strength relate to compressive_strength_mpa.
--- BEGIN TRAINING CSV ---
{train_csv.replace("strength_mpa", "compressive_strength_mpa")}
--- END TRAINING CSV ---

TASK
Predict compressive_strength_mpa for the {n_q} query rows below. They come from papers
that are NOT in the training data. Each row states its paper_id; rows with the same
paper_id share one mix design. Each query gives ctrl_strength_same_age_mpa: the SAME
paper's own 0% RHA strength at that SAME curing age. Use it as that paper's baseline
strength level at that age.

{q_lines}

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
               "options": dict(GREEDY, num_ctx=49152)}
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    d = r.json()
    return d["message"].get("content", ""), time.time() - t0


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tf = pd.read_csv(HERE / "out_le60" / "train_features.csv")
    d = add_ctrl(tf).reset_index(drop=True)
    print(f"framing-C table: {len(d)} rows, {d.paper_id.nunique()} papers")
    X, y, g = d[BASE], d.strength_mpa.values, d.paper_id.values

    rows = []
    for seed in range(n_splits):
        f = OUT / f"split{seed}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
        else:
            tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(X, y, g))
            train_df, test_df = d.iloc[tr], d.iloc[te]
            prompt, tq = build_prompt(train_df, test_df)
            try:
                content, dt = call(prompt, len(tq))
            except Exception as e:
                print(f"split {seed}: ERROR {e}")
                continue
            preds = extract(content, len(tq))
            rec = dict(seed=seed, n_train=len(train_df), n_test=len(tq), elapsed_s=round(dt, 1),
                       true=tq.strength_mpa.tolist(), preds=preds, raw=content)
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"split {seed}: {'OK' if preds else 'INVALID'} ({dt:.0f}s, n_test={len(tq)})", flush=True)
        if rec["preds"] is None:
            continue
        y_true, y_pred = rec["true"], rec["preds"]
        rows.append(dict(seed=rec["seed"], n_test=rec["n_test"],
                         r2=r2_score(y_true, y_pred), rmse=mean_squared_error(y_true, y_pred) ** 0.5,
                         mae=mean_absolute_error(y_true, y_pred),
                         bias=sum(p - t for p, t in zip(y_pred, y_true)) / len(y_true)))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    print(f"\nValid splits: {len(res)} / {n_splits}")
    print(f"R2   = {res.r2.mean():.3f} +/- {res.r2.std():.3f}  (range {res.r2.min():.3f} to {res.r2.max():.3f})")
    print(f"RMSE = {res.rmse.mean():.2f} +/- {res.rmse.std():.2f}")
    print(f"MAE  = {res.mae.mean():.2f} +/- {res.mae.std():.2f}")
    print(f"Bias = {res.bias.mean():+.2f}")


if __name__ == "__main__":
    main()
