"""
Step 10 - RCF <= 60 MPa: gpt-oss vs the final ML model on IDENTICAL data.

Parity rules enforced here (and recorded in out_llm_le60/parity_manifest.json):
  * identical held-out papers and identical test rows, in one fixed order;
  * identical training rows;
  * identical input columns -- the prompt's training CSV and query lines contain
    exactly the model's feature columns, nothing else. In particular paper_id,
    row_id, mix_id and source are NOT shown to the LLM, because the ML model does
    not get them either. Predictions are aligned by query order alone.
  * the target is never shown for test rows to either method.

Commands:
  python pipeline_rcf/10_llm_le60.py prep
  python pipeline_rcf/10_llm_le60.py run <n_runs>
  python pipeline_rcf/10_llm_le60.py ml
  python pipeline_rcf/10_llm_le60.py score
"""
import hashlib
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
SRC = HERE / "out_le60"
OUT = HERE / "out_llm_le60"
TEMPLATE = HERE.parent / "pipeline" / "prompts" / "rcf_strength_le60_v2.txt"
URL = "http://localhost:11434"
MODEL = "gpt-oss:120b-cloud"
CUT = 60.0

CHEM = ["sio2", "cao", "al2o3", "fe2o3", "mgo", "so3", "na2o", "k2o", "loi", "amorphous_pct"]
NOMINAL = ["activation_method", "processing_method", "mineralogy", "material_family"]
ORDINAL = ["pozzolanic_reactivity"]
BASE = ["replacement_pct", "age_days", "ctrl_strength"]
FINE = ["d10", "d50", "d90", "mean_particle_um", "specific_surface", "bet_surface",
        "blaine", "specific_gravity", "bulk_density", "water_absorption"]
# <=60 MPa winner from pipeline_rcf/out_le60/best_model.txt: histgb, base+cat+fine,
# monotone in age and control. Both methods get exactly these columns.
FEATS = BASE + NOMINAL + ORDINAL + FINE
# Held-out papers span the control-strength range (12 - 60 MPa). Four are the papers
# used in the full-data benchmark (51, 81, 37, 19); papers 27 and 44 were replaced by
# 80 and 91 because the <=60 MPa filter leaves 27 with 2 rows and removes 44 entirely.
HELD_OUT = [51, 81, 37, 19, 80, 91]
GREEDY = {"temperature": 0, "seed": 0, "top_k": 1, "top_p": 1.0,
          "repeat_penalty": 1.0, "num_predict": 16384}
ML = dict(kind="histgb", params=dict(max_depth=2, learning_rate=0.03, max_iter=300,
                                     l2_regularization=3.0, min_samples_leaf=20,
                                     random_state=42),
          monotone=["age_days", "ctrl_strength"])
MAX_ATTEMPTS = 3


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def to_csv(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().strip()


def framing_c():
    """<=60 MPa rows only (controls included), then blends with a same-age control."""
    tf = pd.read_csv(SRC / "train_features.csv")
    df = tf[tf.age_known & (tf.strength_mpa <= CUT)].reset_index(drop=True)
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"], as_index=False)
            ["strength_mpa"].mean().rename(columns={"strength_mpa": "ctrl_strength"}))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    return d[d.replacement_pct > 0].reset_index(drop=True)


def prep():
    OUT.mkdir(exist_ok=True)
    d = framing_c()
    test = d[d.paper_id.isin(HELD_OUT)].sort_values(
        ["paper_id", "replacement_pct", "age_days", "strength_mpa"]).reset_index(drop=True)
    train = d[~d.paper_id.isin(HELD_OUT)].sort_values(
        ["paper_id", "replacement_pct", "age_days", "strength_mpa"]).reset_index(drop=True)
    test.insert(0, "row_id", range(len(test)))

    # the ONLY tables either method may read
    train[FEATS + ["strength_mpa"]].to_csv(OUT / "train_inputs.csv", index=False)
    test[FEATS].to_csv(OUT / "test_inputs.csv", index=False)
    test[["row_id", "paper_id"] + BASE + ["strength_mpa"]].to_csv(OUT / "ground_truth.csv", index=False)

    dup = len(test) - test.groupby(["paper_id", "replacement_pct", "age_days"]).ngroups
    manifest = dict(cut_mpa=CUT, held_out_papers=HELD_OUT, feature_columns=FEATS,
                    target="strength_mpa", n_train=len(train), n_test=len(test),
                    train_papers=int(train.paper_id.nunique()), duplicate_key_test_rows=dup,
                    columns_hidden_from_both=["paper_id", "row_id", "mix_id", "source",
                                              "wb_block", "graph_derived", "mix_activation"],
                    alignment="query order == row_id order",
                    train_inputs_sha256=sha((OUT / "train_inputs.csv").read_text(encoding="utf-8")),
                    test_inputs_sha256=sha((OUT / "test_inputs.csv").read_text(encoding="utf-8")))
    (OUT / "parity_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"train {len(train)} rows / {train.paper_id.nunique()} papers; "
          f"test {len(test)} rows / {test.paper_id.nunique()} papers ({dup} duplicate-key rows)")
    for pid in HELD_OUT:
        s = test[test.paper_id == pid]
        if s.empty:
            raise SystemExit(f"paper {pid} has no rows after the <= {CUT:.0f} MPa filter")
        print(f"  paper {pid}: n={len(s)} ctrl={s.ctrl_strength.mean():.1f} "
              f"strength {s.strength_mpa.min():.1f}-{s.strength_mpa.max():.1f}")
    print(f"features given to BOTH methods ({len(FEATS)}): {FEATS}")


def build_prompt(show_paper_id=False):
    """Parity prompt. show_paper_id=True is a DIAGNOSTIC ONLY (breaks parity: the ML
    model never sees paper_id), used to measure what that identifier is worth."""
    train = pd.read_csv(OUT / "train_inputs.csv")
    test = pd.read_csv(OUT / "test_inputs.csv")
    if show_paper_id:
        gt = pd.read_csv(OUT / "ground_truth.csv")
        test = test.copy()
        test.insert(0, "paper_id", gt.paper_id.values)
        d = framing_c()
        tr_ids = d[~d.paper_id.isin(HELD_OUT)].sort_values(
            ["paper_id", "replacement_pct", "age_days", "strength_mpa"]).paper_id.values
        train = train.copy()
        train.insert(0, "paper_id", tr_ids)
    lines = []
    for i, r in test.iterrows():
        row = {c: (round(v, 3) if isinstance(v, float) else v)
               for c, v in r.items() if not pd.isna(v)}
        lines.append(f"Query {i + 1}: {json.dumps(row)}")
    tmpl = TEMPLATE.read_text(encoding="utf-8")
    prompt = tmpl.format(n_train=len(train),
                         train_csv=to_csv(train.rename(columns={"strength_mpa": "compressive_strength_mpa"})),
                         n_queries=len(test), queries="\n".join(lines))
    return prompt, len(test), sha(tmpl)


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


def digest():
    for m in requests.get(f"{URL}/api/tags", timeout=30).json()["models"]:
        if m["name"] == MODEL:
            return m["digest"]
    return None


def run(n_runs, show_paper_id=False):
    prompt, n, tmpl_sha = build_prompt(show_paper_id)
    dig = digest()
    tag = "diag_run" if show_paper_id else "run"
    print(f"prompt chars {len(prompt)}  n_test {n}  template {tmpl_sha[:12]}  model {dig[:12]}"
          f"{'  [DIAGNOSTIC: paper_id visible]' if show_paper_id else ''}")
    for i in range(n_runs):
        f = OUT / f"{tag}{i}.json"
        if f.exists():
            continue
        preds, attempts, times = None, 0, []
        while preds is None and attempts < MAX_ATTEMPTS:
            attempts += 1
            try:
                t0 = time.time()
                r = requests.post(f"{URL}/api/chat", json={
                    "model": MODEL, "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": dict(GREEDY, num_ctx=65536)}, timeout=900)
                r.raise_for_status()
                content = r.json()["message"].get("content", "")
                times.append(round(time.time() - t0, 1))
            except Exception as e:
                print(f"run {i} attempt {attempts}: ERROR {e}", flush=True)
                continue
            preds = extract(content, n)
        f.write_text(json.dumps(dict(run=i, attempts=attempts, elapsed_s=times,
                                     paper_id_visible=show_paper_id,
                                     model=MODEL, digest=dig, template_sha256=tmpl_sha,
                                     prompt_sha256=sha(prompt), preds=preds, raw=content)),
                     encoding="utf-8")
        print(f"run {i}: {'OK' if preds else 'INVALID'} attempts={attempts} "
              f"t={sum(times):.0f}s", flush=True)


def ml():
    """The final ML model, trained on exactly the rows and columns the LLM sees."""
    train = pd.read_csv(OUT / "train_inputs.csv")
    test = pd.read_csv(OUT / "test_inputs.csv")
    X = pd.get_dummies(pd.concat([train[FEATS], test[FEATS]]), columns=NOMINAL, dummy_na=True)
    Xtr, Xte = X.iloc[:len(train)], X.iloc[len(train):]
    order = list(X.columns)
    if ML["kind"] == "xgb":
        from xgboost import XGBRegressor
        kw = dict(ML["params"])
        kw["monotone_constraints"] = tuple(1 if c in ML["monotone"] else 0 for c in order)
        m = XGBRegressor(**kw)
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        kw = dict(ML["params"])
        kw["monotonic_cst"] = [1 if c in ML["monotone"] else 0 for c in order]
        m = HistGradientBoostingRegressor(**kw)
    m.fit(Xtr, train.strength_mpa.values)
    pred = m.predict(Xte)
    gt = pd.read_csv(OUT / "ground_truth.csv")
    gt["ml_pred"] = pred
    gt.to_csv(OUT / "ml_predictions.csv", index=False)
    print(f"{ML['kind']} on {len(train)} rows x {len(order)} encoded columns -> "
          f"R2 {r2_score(gt.strength_mpa, pred):.3f} RMSE "
          f"{np.sqrt(mean_squared_error(gt.strength_mpa, pred)):.2f} MAE "
          f"{mean_absolute_error(gt.strength_mpa, pred):.2f}")


def metrics(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=mean_absolute_error(y, p), bias=float(np.mean(np.asarray(p) - np.asarray(y))))


def score():
    gt = pd.read_csv(OUT / "ground_truth.csv")
    y = gt.strength_mpa.values
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("run*.json"))]
    ok = [r for r in runs if r["preds"]]
    rows = [dict(run=r["run"], attempts=r["attempts"], **metrics(y, r["preds"])) for r in ok]
    single = pd.DataFrame(rows)
    single.to_csv(OUT / "single_run_stats.csv", index=False)
    print(f"runs {len(runs)}, valid {len(ok)} (after retries), "
          f"first-attempt valid {sum(1 for r in ok if r['attempts'] == 1)}")
    print(f"single run: R2 {single.r2.mean():.3f} +/- {single.r2.std(ddof=0):.3f} "
          f"(range {single.r2.min():.3f}..{single.r2.max():.3f}); RMSE {single.rmse.mean():.2f}; "
          f"MAE {single.mae.mean():.2f}")
    P = np.vstack([r["preds"] for r in ok])
    blocks = [np.median(P[i:i + 5], axis=0) for i in range(0, len(ok) - len(ok) % 5, 5)]
    for b, med in enumerate(blocks):
        m = metrics(y, med)
        print(f"median of 5, block {b + 1}: R2 {m['r2']:.3f} RMSE {m['rmse']:.2f} "
              f"MAE {m['mae']:.2f} bias {m['bias']:+.2f}")
    if len(blocks) > 1:
        A = np.vstack(blocks)
        print(f"block-to-block per-point spread: median {np.median(A.std(axis=0)):.2f} MPa, "
              f"max {A.std(axis=0).max():.2f} MPa")
    agg = np.median(P, axis=0)
    gt["llm_median_all"] = agg
    if (OUT / "ml_predictions.csv").exists():
        gt["ml_pred"] = pd.read_csv(OUT / "ml_predictions.csv")["ml_pred"]
        mm = metrics(y, gt.ml_pred.values)
        print(f"ML on identical rows/inputs: R2 {mm['r2']:.3f} RMSE {mm['rmse']:.2f} "
              f"MAE {mm['mae']:.2f} bias {mm['bias']:+.2f}")
        per = gt.groupby("paper_id").apply(
            lambda g: pd.Series(dict(n=len(g),
                                     ml_mae=mean_absolute_error(g.strength_mpa, g.ml_pred),
                                     llm_mae=mean_absolute_error(g.strength_mpa, g.llm_median_all),
                                     ctrl=g.ctrl_strength.iloc[0])), include_groups=False)
        print(per.round(2).to_string())
    gt.to_csv(OUT / "final_predictions.csv", index=False)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(int(sys.argv[2]))
    elif cmd == "rundiag":
        run(int(sys.argv[2]), show_paper_id=True)
    else:
        {"prep": prep, "ml": ml, "score": score}[cmd]()
