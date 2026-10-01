"""
Step 16 - gpt-oss:120b-cloud on the SAME 40 repeated grouped hold-outs used for
XGBoost in Part III (<=60 MPa, framing C: blend rows with a same-age control).

For each split s = 0..39 (GroupShuffleSplit(1, test_size=0.3, random_state=s) over
paper_id, identical to 06_ablation / 07_best_model), the training papers' rows go into
the prompt as CSV and the held-out papers' rows are the queries. The model sees every
available feature. Prompt template (prompts/rha_strength_v1.txt), greedy settings (S1)
and answer parsing are reused from 14_llm_repro.py. Each split is run 5 times; an
invalid answer (wrong length / not numeric) is retried up to 2 more times.
Outputs are saved per call to out_llm_cv/ and skipped on rerun (resumable).

Usage:
  python pipeline/16_llm_grouped_cv.py run [first_split] [last_split]
  python pipeline/16_llm_grouped_cv.py score
"""
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_llm_cv"
N_SPLITS, N_RUNS, MAX_ATTEMPTS = 40, 5, 3

spec = importlib.util.spec_from_file_location("llm14", HERE / "14_llm_repro.py")
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
CFG = llm.SETTINGS["S1_greedy"]

RENAME = {"strength_mpa": "compressive_strength_mpa", "ctrl_strength": "control_strength_same_age_mpa",
          "sio2": "sio2_pct", "cao": "cao_pct", "al2o3": "al2o3_pct", "fe2o3": "fe2o3_pct",
          "mgo": "mgo_pct", "so3": "so3_pct", "na2o": "na2o_pct", "k2o": "k2o_pct", "loi": "loi_pct"}
NOTE = ("Each row includes control_strength_same_age_mpa: the SAME paper's own 0% RHA (plain "
        "cement) compressive strength at the same curing age. Use it as that paper's baseline "
        "strength level.")
POZZ = {0.0: "low", 1.0: "medium", 2.0: "high"}


def load_blend():
    df = pd.read_csv(HERE / "out_le60" / "train_features.csv")
    ctrl = (df[df.replacement_pct == 0].groupby(["paper_id", "age_days"])["strength_mpa"]
            .mean().rename("ctrl_strength"))
    d = df.merge(ctrl, on=["paper_id", "age_days"], how="inner")
    d = d[d.replacement_pct > 0].drop(columns=["is_range"]).reset_index(drop=True)
    d["pozzolanic_reactivity"] = d["pozzolanic_reactivity"].map(POZZ)
    d["paper_id"] = d["paper_id"].map(lambda p: f"P{int(p):03d}")
    return d.rename(columns=RENAME)


def splits(d):
    g = d["paper_id"].values
    return [next(GroupShuffleSplit(1, test_size=0.3, random_state=s).split(d, groups=g))
            for s in range(N_SPLITS)]


def to_csv(rows):
    buf = io.StringIO()
    rows.to_csv(buf, index=False, lineterminator="\n", float_format="%g")
    return buf.getvalue().strip()


def build(template, train, test):
    key = ["paper_id", "replacement_pct", "age_days"]
    train = train.sort_values(key)
    test = test.sort_values(key)
    q = test.drop(columns=["compressive_strength_mpa"])
    lines = []
    for i, (_, r) in enumerate(q.iterrows(), 1):
        feats = {k: (v if not isinstance(v, float) else float(f"{v:g}"))
                 for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}
        lines.append(f"Query {i}: {json.dumps(feats)}")
    prompt = template.format(n_train=len(train), train_csv=to_csv(train), n_queries=len(test),
                             control_note=NOTE, queries="\n".join(lines))
    return prompt, test


def run(first, last):
    OUT.mkdir(exist_ok=True)
    template = llm.TEMPLATE_PATH.read_text(encoding="utf-8")
    d = load_blend()
    digest = llm.model_digest()
    for s, (tr, te) in enumerate(splits(d)):
        if not first <= s <= last:
            continue
        prompt, test = build(template, d.iloc[tr], d.iloc[te])
        n = len(test)
        for r in range(N_RUNS):
            f = OUT / f"split{s:02d}_run{r}.json"
            if f.exists():
                continue
            preds, attempts, times = None, 0, []
            while preds is None and attempts < MAX_ATTEMPTS:
                attempts += 1
                try:
                    content, think, dt = llm.call(prompt, CFG, n)
                except Exception as e:
                    print(f"split{s:02d} run{r} attempt{attempts}: ERROR {e}", flush=True)
                    continue
                times.append(round(dt, 1))
                preds = llm.parse(content, n)
            rec = dict(split=s, run=r, attempts=attempts, elapsed_s=times, digest=digest,
                       template_sha256=llm.sha(template), prompt_sha256=llm.sha(prompt),
                       test_papers=sorted(test.paper_id.unique().tolist()), n_test=n,
                       keys=test[["paper_id", "replacement_pct", "age_days"]].values.tolist(),
                       true=test["compressive_strength_mpa"].tolist(), preds=preds)
            f.write_text(json.dumps(rec), encoding="utf-8")
            print(f"split{s:02d} run{r}: {'OK' if preds else 'INVALID'} attempts={attempts} "
                  f"n={n} t={sum(times):.0f}s", flush=True)


def metrics(y, p):
    return dict(r2=r2_score(y, p), rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=mean_absolute_error(y, p))


def score():
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(OUT.glob("split*_run*.json"))]
    rows, med_rows = [], []
    for s in sorted({r["split"] for r in recs}):
        rs = [r for r in recs if r["split"] == s and r["preds"]]
        for r in rs:
            rows.append(dict(split=s, run=r["run"], attempts=r["attempts"],
                             time_s=sum(r["elapsed_s"]), **metrics(r["true"], r["preds"])))
        if len(rs) == N_RUNS:
            P = np.vstack([r["preds"] for r in rs])
            med_rows.append(dict(split=s, query_sd_median=float(np.median(P.std(axis=0))),
                                 **metrics(rs[0]["true"], np.median(P, axis=0))))
    single, med = pd.DataFrame(rows), pd.DataFrame(med_rows)
    single.to_csv(OUT / "per_run.csv", index=False)
    med.to_csv(OUT / "per_split_median5.csv", index=False)
    all_attempts = [r["attempts"] for r in recs]
    print(f"calls saved {len(recs)}; valid {sum(1 for r in recs if r['preds'])}; "
          f"first-attempt valid {sum(1 for a, r in zip(all_attempts, recs) if a == 1 and r['preds'])}; "
          f"extra attempts {sum(a - 1 for a in all_attempts)}")
    per_pass = single.groupby("run")[["r2", "rmse", "mae"]].agg(["mean", "std"])
    print("single run, per pass of 40 splits:\n", per_pass.round(3).to_string())
    print(f"single run (all valid split-runs): R2 {single.r2.mean():.3f} +/- {single.r2.std(ddof=0):.3f}; "
          f"RMSE {single.rmse.mean():.2f}; MAE {single.mae.mean():.2f}; time/call {single.time_s.mean():.1f}s")
    if len(med):
        print(f"median of 5 ({len(med)} splits): R2 {med.r2.mean():.3f} +/- {med.r2.std(ddof=0):.3f}; "
              f"RMSE {med.rmse.mean():.2f} +/- {med.rmse.std(ddof=0):.2f}; MAE {med.mae.mean():.2f}; "
              f"run-to-run query SD median {med.query_sd_median.median():.2f} MPa")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        b = int(sys.argv[3]) if len(sys.argv) > 3 else N_SPLITS - 1
        run(a, b)
    else:
        score()
