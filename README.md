# Predicting Compressive Strength of Blended Concrete (RHA and RCF)

Machine-learning study on two literature corpora — **RHA** (rice-husk ash) and **RCF**
(recycled concrete fines) — predicting the compressive strength of concrete in which part
of the cement is replaced by the material. Both final models cover the range up to
**60 MPa** and require the paper's own control strength as an input.

---

## Quick start: predicting strength

### Web chat

Start the local chat app from the repository root:

```bash
.venv/bin/python app.py
```

Open `http://127.0.0.1:8000`. Select RHA or RCF and provide all three inputs:
replacement level (0-30% by mass of cement), curing age (7 or 28 days), and control
mixture strength (1-60 MPa) at the same age. Missing or out-of-range values are rejected
before prediction. On macOS, RHA inference may require `brew install libomp`.

The expected prediction target is compressive strength in MPa. Use control strength from
the same mix at 0% replacement and the same curing age. Both models are research estimates,
not substitutes for laboratory testing or design approval.

To deploy on Render, push this project to a GitHub repository, connect that repository
to Render, and create a new Blueprint from `render.yaml`. The Docker image installs
only the web runtime dependencies and copies the predictor scripts and saved model
artifacts; no research datasets are included. The free web service can sleep while idle.

Two ready-to-use models, one per material. Both need the same three core inputs:

| Input | Meaning |
|---|---|
| `replacement_pct` | 0-30% of cement mass replaced by RHA / RCF |
| `age_days` | 7 or 28 days |
| `ctrl_strength` | control mixture strength (1-60 MPa) at the **same curing age** |

The control strength is not cheating: the plain mix can always be cast and tested first,
and it is the single most important input for both materials.

```bash
# RCF — one mix
python inference/predict_rcf.py --replacement_pct 20 --age_days 28 --ctrl_strength 42

# RCF — batch from CSV
python inference/predict_rcf.py --csv inference/example_rcf_inputs.csv --out predictions.csv

# RHA — one mix
python inference/predict_rha.py --replacement_pct 10 --age_days 28 --ctrl_strength 40

# RHA — strength across replacement levels for one mix design
python inference/predict_rha.py --age_days 28 --ctrl_strength 40 --sweep 0:30:2.5
```

Run `--help` on either script for the full list of optional inputs. RCF accepts eleven
further powder properties (fineness, processing, mineralogy, …); anything omitted is
treated as "not reported", which the model handles natively.

> **On this machine only:** the RHA model uses XGBoost, whose DLL is blocked by Windows
> Smart App Control here. Run it under WSL instead:
> `~/rha_venv/bin/python inference/predict_rha.py ...`
> On any normal machine with `xgboost` installed, plain `python` works.

---

## The two final models

| | RHA | RCF |
|---|---|---|
| Model | XGBoost, depth 2 | HistGradientBoosting, depth 2 |
| Inputs | 3 | 18 |
| Constraint | strength rises with age and control strength | same |
| Accuracy on an unseen paper | R² 0.35 ± 0.25, MAE 8.2 MPa | R² 0.31 ± 0.19, MAE 8.5 MPa |
| Trained on | 308 rows / 21 papers | 688 rows / 41 papers |
| Model file | `models/rha_strength_model.json` (+ `.joblib`) | `models/rcf_strength_model.joblib` |

Accuracy is measured by holding out whole papers, which is the honest test for a new
study; see `models/README.md` for the model cards and `reports/` for how these were
chosen.

---

## Repository layout

```
models/           the two final trained models + metadata (feature list, encoding, scores)
inference/        prediction scripts, example inputs, and the script that saved the models
reports/          all reports, numbered in the order they were produced
  sources/        LaTeX / HTML sources and the PDF build script
  archive/        superseded builds, kept for reference
datasets/
  raw/            the two source spreadsheets
  clean_datasets/ shareable cleaned datasets (per-measurement and per-paper)
  llm_benchmark_rha/  fixed train/test splits used for the RHA LLM experiments
  llm_benchmark_rcf_early/  earlier RCF train/test splits, kept for reference
pipeline/         RHA research pipeline (parsing → features → ablation → final model)
pipeline_rcf/     RCF research pipeline (same stages, plus the LLM parity benchmark)
docs/             working notes
```

Each pipeline folder keeps its own `out*/` directories holding the results the reports
quote — `out/` for the full-data study and `out_le60/` for the ≤60 MPa study that produced
the final models.

---

## Reports, in order

| # | File | What it covers |
|---|---|---|
| 1 | `reports/01_rha_strength_study.pdf` | RHA: data, parsing, ablation, final model; then the ≤60 MPa re-analysis and the LLM reproducibility study |
| 2 | `reports/02_rcf_strength_study.pdf` | RCF: the same programme on the recycled-fines corpus, plus its first LLM benchmark |
| 3 | `reports/03_rcf_final_model_study_le60.pdf` | RCF restricted to ≤60 MPa — the study that selected the final RCF model, with a strict ML-vs-LLM comparison |
| 4 | `reports/04_rha_vs_rcf_comparison.pdf` | Why the two corpora need different models |

Rebuild the HTML-based reports (3 and 4) with:

```bash
python reports/sources/build_report.py 03_rcf_final_model_study_le60.html
```

Reports 1 and 2 are LaTeX; no LaTeX toolchain is installed on this machine at present, so
their PDFs are the last built versions and their `.tex` sources live in `reports/sources/`.

---

## Reproducing the research

Not needed for prediction, and not cheap — the ablations take tens of minutes.

```bash
python pipeline/01_load_clean.py          # RHA: parse the master sheet
python pipeline/13_ml_le60.py             # RHA: ≤60 MPa ablation + final model
python pipeline_rcf/09_ml_le60.py         # RCF: ≤60 MPa ablation + final model
python inference/build_final_models.py    # save the chosen models to models/
```

The LLM experiments (`pipeline/14`, `pipeline/16`, `pipeline_rcf/10`) additionally need
Ollama running with a cloud model available.

## Requirements

`pandas`, `numpy`, `scikit-learn`, `joblib` for RCF; `xgboost` additionally for RHA.
See `requirements.txt`.
