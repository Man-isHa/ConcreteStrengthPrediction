"""
Step 15 - Score the Part III LLM runs (reads pipeline/out_llm/all_predictions.csv,
produced by `14_llm_repro.py score`).

Per setting/condition: valid runs, R2/RMSE/MAE/bias mean +/- SD over runs, number of
distinct prediction vectors, per-query SD across runs, median-of-5 aggregates.
Writes out_llm/summary_settings.csv, summary_bands.csv, summary_aggregates.csv.

Run: python pipeline/15_llm_analyze.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT = Path(__file__).resolve().parent / "out_llm"
KEY = ["test_case_id", "replacement_pct", "age_days"]
N_POINTS = 49


def metrics(y, p):
    return dict(r2=r2_score(y, p), rmse=np.sqrt(mean_squared_error(y, p)),
                mae=mean_absolute_error(y, p), bias=float(np.mean(p - y)))


def run_table(df):
    """One prediction vector per (setting, cond, run); per-paper runs are stitched."""
    df = df.copy()
    df["pred_mpa"] = pd.to_numeric(df.pred_mpa, errors="coerce")
    runs = []
    for (s, c, r), g in df.groupby(["setting", "cond", "run"]):
        complete = g[KEY].drop_duplicates().shape[0] == N_POINTS
        valid = complete and g.valid.all() and g.pred_mpa.notna().all()
        runs.append(dict(setting=s, cond=c, run=r, valid=bool(valid),
                         elapsed_s=g.groupby("group").elapsed_s.first().sum(),
                         thinking_chars=g.groupby("group").thinking_chars.first().sum(),
                         data=g.sort_values(KEY)))
    return runs


def main():
    df = pd.read_csv(OUT / "all_predictions.csv")
    runs = run_table(df)
    settings, bands, aggs = [], [], []
    for (s, c) in sorted({(r["setting"], r["cond"]) for r in runs}):
        rs = [r for r in runs if r["setting"] == s and r["cond"] == c]
        ok = [r for r in rs if r["valid"]]
        if not ok:
            settings.append(dict(setting=s, cond=c, runs=len(rs), valid=0))
            continue
        m = pd.DataFrame([metrics(r["data"].true_mpa.values, r["data"].pred_mpa.values) for r in ok])
        P = np.vstack([r["data"].pred_mpa.values for r in ok])
        sd = P.std(axis=0)
        settings.append(dict(
            setting=s, cond=c, runs=len(rs), valid=len(ok),
            r2_mean=m.r2.mean(), r2_std=m.r2.std(ddof=0), r2_min=m.r2.min(), r2_max=m.r2.max(),
            rmse_mean=m.rmse.mean(), rmse_std=m.rmse.std(ddof=0),
            mae_mean=m.mae.mean(), mae_std=m.mae.std(ddof=0), bias_mean=m.bias.mean(),
            distinct_outputs=len({tuple(np.round(v, 6)) for v in P}),
            query_sd_median=float(np.median(sd)), query_sd_max=float(sd.max()),
            queries_identical=int((sd == 0).sum()), n_queries=P.shape[1],
            elapsed_mean=np.mean([r["elapsed_s"] for r in ok]),
            thinking_mean=np.mean([r["thinking_chars"] for r in ok])))

        base = ok[0]["data"]
        for band in sorted(base.band.unique()):
            mask = (base.band == band).values
            bm = pd.DataFrame([metrics(r["data"].true_mpa.values[mask], r["data"].pred_mpa.values[mask])
                               for r in ok])
            bands.append(dict(setting=s, cond=c, band=band, n=int(mask.sum()),
                              mae_mean=bm.mae.mean(), mae_std=bm.mae.std(ddof=0),
                              bias_mean=bm.bias.mean()))

        # median-of-5 aggregates from disjoint blocks of valid runs
        blocks = [ok[i:i + 5] for i in range(0, len(ok) - len(ok) % 5, 5)]
        agg_vecs = []
        for b, blk in enumerate(blocks):
            med = np.median(np.vstack([r["data"].pred_mpa.values for r in blk]), axis=0)
            agg_vecs.append(med)
            aggs.append(dict(setting=s, cond=c, block=b, **metrics(base.true_mpa.values, med)))
        if len(agg_vecs) > 1:
            A = np.vstack(agg_vecs)
            aggs.append(dict(setting=s, cond=c, block="spread",
                             query_sd_median=float(np.median(A.std(axis=0))),
                             query_sd_max=float(A.std(axis=0).max())))

    pd.DataFrame(settings).to_csv(OUT / "summary_settings.csv", index=False)
    pd.DataFrame(bands).to_csv(OUT / "summary_bands.csv", index=False)
    pd.DataFrame(aggs).to_csv(OUT / "summary_aggregates.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    fmt = lambda x: f"{x:.3f}"
    print(pd.DataFrame(settings).to_string(index=False, float_format=fmt))
    print(pd.DataFrame(bands).to_string(index=False, float_format=fmt))
    if aggs:
        print(pd.DataFrame(aggs).to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
