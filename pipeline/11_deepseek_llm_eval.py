"""
Step 11 - Evaluate gpt-oss:120b-cloud (via Ollama's cloud model proxy) on the
same fixed paper-level held-out benchmark used for the Fable ablation
(RHA_LLM_Results_So_Far.md): 6 papers (P036, P022, P007, P030, P054, P064),
57 strength points, WITH_CONTROL vs NO_CONTROL.

Model note: DeepSeek-R1 8B was tried first, locally, but a single call with
the full training CSV in context (~39k tokens) took 10+ minutes on the
available 8GB laptop GPU -- impractical for a 12-call run. kimi-k2.6:cloud
was tried next but returned 402 Payment Required (needs a paid ollama.com
plan/credits). gpt-oss:120b-cloud is free-tier accessible, cloud-hosted (not
constrained by local VRAM), and has a native 131072-token context window, so
it replaces both.

Protocol: one Ollama call per test paper per condition (12 calls total, not
57x2). Each call gets the paper's full held-out feature rows plus the entire
condition's training CSV as in-context reference data, and must return a
strict JSON array of predictions in the same order as the queries. Ground
truth (never shown to the model) comes from 09_/10_GROUND_TRUTH_*.csv.

Run: python pipeline/11_deepseek_llm_eval.py
Requires: `ollama signin` completed and a free-tier-eligible cloud model.
"""
import csv
import json
import re
import time
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent.parent / "datasets" / "llm_benchmark_rha"
OUT = Path(__file__).resolve().parent / "out_gptoss"
OUT.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gpt-oss:120b-cloud"

CONDITIONS = {
    "WITH_CONTROL": dict(
        train=BASE / "01_TRAIN_strength_WITH_control (1).csv",
        test=BASE / "02_TEST_strength_WITH_control_CLEAN (1).csv",
        gt=BASE / "09_GROUND_TRUTH_strength_WITH_control (1).csv",
        num_ctx=131072,
    ),
    "NO_CONTROL": dict(
        train=BASE / "05_TRAIN_strength_NO_control (1).csv",
        test=BASE / "06_TEST_strength_NO_control (1).csv",
        gt=BASE / "10_GROUND_TRUTH_strength_NO_control.csv",
        num_ctx=131072,
    ),
}


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_text(path):
    return path.read_text(encoding="utf-8-sig")


def rkey(pct, age):
    return (round(float(pct), 3), round(float(age), 3))


def build_prompt(train_text, paper_rows, condition):
    n = len(paper_rows)
    query_lines = []
    for i, r in enumerate(paper_rows, 1):
        feats = {k: v for k, v in r.items() if k not in ("test_case_id",) and v != ""}
        query_lines.append(f"Query {i}: {json.dumps(feats)}")
    queries = "\n".join(query_lines)

    control_note = (
        "Each row includes control_strength_28d_mpa, the SAME paper's own "
        "0%-replacement (plain OPC) 28-day compressive strength. Use it as the "
        "baseline strength level for that paper."
        if condition == "WITH_CONTROL" else
        "No control/baseline strength is given for these papers. You must infer "
        "the paper's strength scale from its material/mix features alone."
    )

    return f"""You are a concrete materials scientist predicting compressive strength
of rice-husk-ash (RHA) blended concrete/mortar from literature data.

Below is a TRAINING dataset of {train_text.count(chr(10))} rows from OTHER papers (CSV,
header row included). Each row is one (paper, RHA replacement %, curing age) data
point with its known compressive_strength_mpa. Learn the relationship between the
input columns and compressive_strength_mpa from this data.

--- TRAINING DATA (CSV) ---
{train_text}
--- END TRAINING DATA ---

Now predict compressive strength for {n} held-out query rows, ALL from one paper
NOT in the training data above. {control_note}

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


def call_ollama(prompt, num_ctx, timeout=1800):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0.1},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"]


def main():
    all_results = []
    for cond_name, cfg in CONDITIONS.items():
        print(f"\n{'='*70}\n{cond_name}\n{'='*70}")
        train_text = read_text(cfg["train"])
        test_rows = read_rows(cfg["test"])
        gt_rows = read_rows(cfg["gt"])

        gt_by_case = {}
        for r in gt_rows:
            gt_by_case.setdefault(r["test_case_id"], {})[rkey(r["replacement_pct"], r["age_days"])] = {
                "true": float(r["true_strength_mpa"]),
                "band": r["strength_band"],
            }

        by_case = {}
        for r in test_rows:
            by_case.setdefault(r["test_case_id"], []).append(r)

        for case_id, rows in sorted(by_case.items()):
            paper_id = rows[0]["paper_id"]
            print(f"  [{case_id} / {paper_id}] n={len(rows)} -- calling {MODEL} "
                  f"(num_ctx={cfg['num_ctx']})...", flush=True)
            prompt = build_prompt(train_text, rows, cond_name)
            t0 = time.time()
            try:
                content = call_ollama(prompt, cfg["num_ctx"])
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            dt = time.time() - t0
            preds = extract_predictions(content, len(rows))
            (OUT / f"{cond_name}_{case_id}_raw.txt").write_text(content, encoding="utf-8")
            if preds is None:
                print(f"    FAILED to parse {len(rows)} predictions ({dt:.0f}s). Raw saved.")
                continue
            print(f"    OK ({dt:.0f}s): {preds}")

            for r, p in zip(rows, preds):
                key = rkey(r["replacement_pct"], r["age_days"])
                gt = gt_by_case[case_id].get(key)
                if gt is None:
                    continue
                all_results.append(dict(
                    condition=cond_name, test_case_id=case_id, paper_id=paper_id,
                    replacement_pct=float(r["replacement_pct"]), age_days=float(r["age_days"]),
                    true_mpa=gt["true"], pred_mpa=p, band=gt["band"],
                ))

    out_path = OUT / "deepseek_predictions.csv"
    if all_results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved {len(all_results)} scored predictions -> {out_path}")
    else:
        print("\nNo predictions were successfully scored.")


if __name__ == "__main__":
    main()
