"""
Step 14 - Part III LLM half: gpt-oss:120b-cloud on the <=60 MPa benchmark, with a
cumulative ablation of reproducibility measures and a final WITH/NO control run.

One prompt template (pipeline/prompts/rha_strength_v1.txt) is used for every call;
its SHA-256, the full prompt's SHA-256 and the model digest are logged per call.
Each call's output is saved to out_llm/<setting>/<cond>/run<r>[_<paper>].json and
skipped if it already exists, so an interrupted run resumes where it stopped.

Usage:
  python pipeline/14_llm_repro.py run <setting> <with|no> <n_runs>
  python pipeline/14_llm_repro.py score
"""
import csv
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "llm_benchmark_rha" / "filtered60"
OUT = HERE / "out_llm"
TEMPLATE_PATH = HERE / "prompts" / "rha_strength_v1.txt"
URL = "http://localhost:11434"
MODEL = "gpt-oss:120b-cloud"

CONDITIONS = {
    "with": dict(train="train_with.csv", test="test_with.csv", gt="gt_with.csv",
                 note="Each query includes control_strength_28d_mpa: the SAME paper's own "
                      "0% RHA (plain cement) 28-day compressive strength. Use it as that "
                      "paper's baseline strength level."),
    "no": dict(train="train_no.csv", test="test_no.csv", gt="gt_no.csv",
               note="No control (0% RHA) strength is given. Infer each paper's strength "
                    "level from its material and mix features alone."),
}

GREEDY = {"top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0, "num_predict": 16384}
SETTINGS = {
    "S0_temp0_seed": dict(options={"temperature": 0, "seed": 0}, per_paper=False),
    "S1_greedy": dict(options={"temperature": 0, "seed": 0, **GREEDY}, per_paper=False),
    "S2_schema": dict(options={"temperature": 0, "seed": 0, **GREEDY}, schema=True, per_paper=False),
    "S3_think_low": dict(options={"temperature": 0, "seed": 0, **GREEDY}, schema=True,
                         think="low", per_paper=False),
    "S4_per_paper": dict(options={"temperature": 0, "seed": 0, **GREEDY}, schema=True,
                         think="low", per_paper=True),
}


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sorted_train_csv(path):
    rows = read_rows(path)
    rows.sort(key=lambda r: (r["paper_id"], float(r["replacement_pct"]), float(r["age_days"])))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().strip(), len(rows)


def sorted_queries(path):
    rows = read_rows(path)
    rows.sort(key=lambda r: (r["test_case_id"], float(r["replacement_pct"]), float(r["age_days"])))
    return rows


def build_prompt(template, train_csv, n_train, queries, note):
    q = "\n".join(f"Query {i}: {json.dumps({k: v for k, v in r.items() if v != '' and k != 'test_case_id'})}"
                  for i, r in enumerate(queries, 1))
    return template.format(n_train=n_train, train_csv=train_csv, n_queries=len(queries),
                           control_note=note, queries=q)


def model_digest():
    for m in requests.get(f"{URL}/api/tags", timeout=30).json()["models"]:
        if m["name"] == MODEL:
            return m["digest"]
    return None


def parse(content, n):
    text = re.sub(r"```(?:json)?", "", content)
    for m in reversed(re.findall(r"\[[^\[\]]*\]", text, flags=re.S)):
        try:
            arr = json.loads(m)
        except Exception:
            continue
        if isinstance(arr, list) and len(arr) == n and all(isinstance(x, (int, float)) for x in arr):
            return [float(x) for x in arr]
    return None


def call(prompt, cfg, n):
    payload = {"model": MODEL, "stream": False,
               "messages": [{"role": "user", "content": prompt}],
               "options": dict(cfg["options"], num_ctx=131072)}
    if cfg.get("schema"):
        payload["format"] = {"type": "array", "items": {"type": "number"},
                             "minItems": n, "maxItems": n}
    if cfg.get("think"):
        payload["think"] = cfg["think"]
    t0 = time.time()
    r = requests.post(f"{URL}/api/chat", json=payload, timeout=1200)
    r.raise_for_status()
    d = r.json()
    return d["message"].get("content", ""), len(d["message"].get("thinking") or ""), time.time() - t0


def run(setting, cond, n_runs):
    cfg, c = SETTINGS[setting], CONDITIONS[cond]
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    train_csv, n_train = sorted_train_csv(DATA / c["train"])
    queries = sorted_queries(DATA / c["test"])
    digest = model_digest()
    groups = ({q["test_case_id"]: [x for x in queries if x["test_case_id"] == q["test_case_id"]]
               for q in queries} if cfg["per_paper"] else {"all": queries})
    d = OUT / setting / cond
    d.mkdir(parents=True, exist_ok=True)
    for r in range(n_runs):
        for gid, qs in groups.items():
            f = d / (f"run{r}.json" if gid == "all" else f"run{r}_{gid}.json")
            if f.exists():
                continue
            prompt = build_prompt(template, train_csv, n_train, qs, c["note"])
            try:
                content, think_len, dt = call(prompt, cfg, len(qs))
            except Exception as e:
                print(f"{setting} {cond} run{r} {gid}: ERROR {e}", flush=True)
                continue
            preds = parse(content, len(qs))
            rec = dict(setting=setting, cond=cond, run=r, group=gid, model=MODEL, digest=digest,
                       template_sha256=sha(template), prompt_sha256=sha(prompt),
                       options=cfg["options"], schema=bool(cfg.get("schema")),
                       think=cfg.get("think"), elapsed_s=round(dt, 1), thinking_chars=think_len,
                       keys=[[q["test_case_id"], float(q["replacement_pct"]), float(q["age_days"])] for q in qs],
                       preds=preds, raw=content)
            f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"{setting} {cond} run{r} {gid}: {'OK' if preds else 'INVALID'} {dt:.0f}s "
                  f"think={think_len}", flush=True)


def score():
    """Collect every saved run into one CSV of per-point predictions."""
    rows = []
    for f in sorted(OUT.glob("*/*/run*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        gt = {(r["test_case_id"], float(r["replacement_pct"]), float(r["age_days"])):
              (float(r["true_strength_mpa"]), r["strength_band"])
              for r in read_rows(DATA / CONDITIONS[rec["cond"]]["gt"])}
        for i, k in enumerate(rec["keys"]):
            t, band = gt[tuple(k)]
            rows.append(dict(setting=rec["setting"], cond=rec["cond"], run=rec["run"],
                             group=rec["group"], test_case_id=k[0], replacement_pct=k[1],
                             age_days=k[2], true_mpa=t, band=band,
                             pred_mpa=rec["preds"][i] if rec["preds"] else "",
                             valid=int(rec["preds"] is not None), elapsed_s=rec["elapsed_s"],
                             thinking_chars=rec["thinking_chars"],
                             prompt_sha256=rec["prompt_sha256"], digest=rec["digest"]))
    with open(OUT / "all_predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {OUT / 'all_predictions.csv'}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        score()
