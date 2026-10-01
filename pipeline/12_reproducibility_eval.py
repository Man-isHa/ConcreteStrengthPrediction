"""
Step 12 - Part III: reproducibility test.

Two changes from Part II (11_deepseek_llm_eval.py):
  1. Scope: drop every data point (train AND test/ground-truth) with
     compressive strength > 60 MPa -- "not interested in" the high-strength
     regime for this benchmark. Filtered files live in
     datasets/llm_benchmark_rha/filtered60/{train,test,gt}_{with,no}.csv (built separately).
     57 -> 49 test points; 215->193 / 566->501 training rows.
  2. Protocol: ONE combined prompt per condition (all 6 remaining test papers
     together, not one call per paper) so the same fixed prompt can be sent
     repeatedly at temperature=0 and any variation in the output is due to
     the model/serving stack, not to us changing anything between calls.
     Each condition's prompt is run REPEATS times per model.

Run: python pipeline/12_reproducibility_eval.py [gptoss|deepseek] [with|no] [n_repeats]
"""
import csv
import json
import re
import sys
import time
from pathlib import Path
import requests

BASE = (Path(__file__).resolve().parent.parent / "datasets" / "llm_benchmark_rha"
        / "filtered60")
OUT = Path(__file__).resolve().parent / "out_repro"
OUT.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODELS = {"gptoss": "gpt-oss:120b-cloud", "deepseek": "deepseek-r1:8b"}

CONDITIONS = {
    "with": dict(train=BASE / "train_with.csv", test=BASE / "test_with.csv",
                 gt=BASE / "gt_with.csv", num_ctx=49152, has_control=True),
    "no": dict(train=BASE / "train_no.csv", test=BASE / "test_no.csv",
               gt=BASE / "gt_no.csv", num_ctx=131072, has_control=False),
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_text(path):
    return path.read_text(encoding="utf-8-sig")


def rkey(pct, age):
    return (round(float(pct), 3), round(float(age), 3))


def build_mega_prompt(train_text, test_rows, has_control):
    n = len(test_rows)
    lines = []
    for i, r in enumerate(test_rows, 1):
        feats = {k: v for k, v in r.items() if v != ""}
        lines.append(f"Query {i}: {json.dumps(feats)}")
    queries = "\n".join(lines)
    control_note = (
        "Each row includes control_strength_28d_mpa, the SAME paper's own "
        "0%-replacement (plain OPC) 28-day compressive strength. Use it as the "
        "baseline strength level for that paper."
        if has_control else
        "No control/baseline strength is given. Infer the paper's strength "
        "scale from its material/mix features alone."
    )
    return f"""You are a concrete materials scientist predicting compressive strength
of rice-husk-ash (RHA) blended concrete/mortar from literature data. All strengths in
this dataset are 60 MPa or below (the high-strength/UHPC regime has been excluded).

Below is a TRAINING dataset ({train_text.count(chr(10))} rows, CSV, header included).
Each row is one (paper, RHA replacement %, curing age) point with its known
compressive_strength_mpa. Learn the relationship between the input columns and
compressive_strength_mpa from this data.

--- TRAINING DATA (CSV) ---
{train_text}
--- END TRAINING DATA ---

Now predict compressive strength for {n} held-out query rows, spanning SEVERAL papers
NOT in the training data above (each row states its own paper_id -- rows from the same
paper share a mix design, rows from different paper_ids do not). {control_note}

{queries}

Respond with ONLY a JSON array of exactly {n} numbers, no other text, no units, no
explanation, no markdown code fence -- just the array, one predicted MPa value per
query IN THE SAME ORDER as listed (Query 1 first, Query {n} last), e.g.
[12.3, 45.6, ...]."""


def extract_predictions(content, n):
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S)
    matches = re.findall(r"\[[^\[\]]*\]", content, flags=re.S)
    for m in reversed(matches):
        try:
            arr = json.loads(m)
            if isinstance(arr, list) and len(arr) == n and all(isinstance(x, (int, float)) for x in arr):
                return [float(x) for x in arr]
        except Exception:
            continue
    nums = re.findall(r"-?\d+\.?\d*", content)
    nums = [float(x) for x in nums if x not in ("", "-", ".")]
    if len(nums) >= n:
        return nums[-n:]
    return None


def call_ollama(model, prompt, num_ctx, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0, "seed": 0},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "gptoss"
    cond_key = sys.argv[2] if len(sys.argv) > 2 else "with"
    n_repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    call_timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 900

    model = MODELS[model_key]
    cfg = CONDITIONS[cond_key]
    train_text = read_text(cfg["train"])
    test_rows = read_rows(cfg["test"])
    gt_rows = read_rows(cfg["gt"])
    gt_by_key = {(r["test_case_id"],) + rkey(r["replacement_pct"], r["age_days"]): float(r["true_strength_mpa"]) for r in gt_rows}

    prompt = build_mega_prompt(train_text, test_rows, cfg["has_control"])
    print(f"model={model} cond={cond_key} n_test={len(test_rows)} prompt_chars={len(prompt)} "
          f"repeats={n_repeats} num_ctx={cfg['num_ctx']}")

    results = []
    for rep in range(n_repeats):
        t0 = time.time()
        try:
            content = call_ollama(model, prompt, cfg["num_ctx"], call_timeout)
        except Exception as e:
            print(f"  rep {rep}: ERROR {e}")
            continue
        dt = time.time() - t0
        preds = extract_predictions(content, len(test_rows))
        tag = f"{model_key}_{cond_key}_rep{rep}"
        (OUT / f"{tag}_raw.txt").write_text(content, encoding="utf-8")
        if preds is None:
            print(f"  rep {rep} ({dt:.0f}s): PARSE FAILED, raw saved as {tag}_raw.txt")
            continue
        row_preds = []
        for r, p in zip(test_rows, preds):
            key = (r["test_case_id"],) + rkey(r["replacement_pct"], r["age_days"])
            row_preds.append((key, p, gt_by_key.get(key)))
        results.append(dict(rep=rep, elapsed_s=dt, preds=row_preds, content_hash=hash(content)))
        print(f"  rep {rep} ({dt:.0f}s): got {len(preds)} preds, first 5={preds[:5]}")

    out_path = OUT / f"{model_key}_{cond_key}_repro.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)}/{n_repeats} successful repeats -> {out_path}")


if __name__ == "__main__":
    main()
