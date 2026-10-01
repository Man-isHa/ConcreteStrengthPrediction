# Final models

Two trained models, one per material. Each is the configuration its study selected; the
accompanying `*_metadata.json` records the exact inputs, encoding, column order,
hyper-parameters and measured performance, and the inference scripts read the column
order from it so preprocessing can never drift from the fitted model.

> These files were produced by `inference/build_final_models.py`, which fits the frozen
> configurations on the same ≤60 MPa training tables the reports used. The research
> pipelines only ever saved metrics and predictions, never a fitted model, so this step
> exists solely to persist an artifact — no model selection, tuning or experimentation is
> repeated there.

---

## RHA — `rha_strength_model.json` / `.joblib`

Predicts the compressive strength of rice-husk-ash blended concrete, up to 60 MPa.

| | |
|---|---|
| Algorithm | XGBoost regressor, depth 2, learning rate 0.03, 300 trees, min. child weight 5, L2 = 3.0 |
| Inputs (3) | `replacement_pct`, `age_days`, `ctrl_strength` |
| Preprocessing | none — three numeric columns, missing values handled natively |
| Constraint | predicted strength never falls as `age_days` or `ctrl_strength` rise |
| Trained on | 308 blend rows from 21 papers (`pipeline/out_le60/train_features.csv`) |
| Performance | R² 0.346 ± 0.249, RMSE 10.86 MPa, MAE 8.18 MPa |
| Selected in | `reports/01_rha_strength_study.pdf`, Part III; config in `pipeline/out_le60/best_model.txt` |

The study tested adding ash chemistry, fineness, processing and text features: none
improved predictions on unseen papers at this sample size, so the final model keeps three
inputs. Two files are saved — the `.json` is XGBoost's portable format (preferred, survives
library upgrades); the `.joblib` is the pickled estimator.

---

## RCF — `rcf_strength_model.joblib`

Predicts the compressive strength of concrete blended with recycled concrete fines, up to
60 MPa.

| | |
|---|---|
| Algorithm | HistGradientBoostingRegressor, depth 2, learning rate 0.03, 300 iterations, L2 = 3.0, min. 20 samples per leaf |
| Inputs (18) | `replacement_pct`, `age_days`, `ctrl_strength`; `activation_method`, `processing_method`, `mineralogy`, `material_family`; `pozzolanic_reactivity`; `d10`, `d50`, `d90`, `mean_particle_um`, `specific_surface`, `bet_surface`, `blaine`, `specific_gravity`, `bulk_density`, `water_absorption` |
| Preprocessing | the four categorical columns are one-hot encoded with "not reported" as its own level, giving 37 encoded columns in a fixed order (stored in the metadata); numeric gaps are left as NaN and handled natively |
| Constraint | predicted strength never falls as `age_days` or `ctrl_strength` rise |
| Trained on | 688 blend rows from 41 papers (`pipeline_rcf/out_le60/train_features.csv`) |
| Performance | R² 0.308 ± 0.187, RMSE 11.02 MPa, MAE 8.47 MPa |
| Selected in | `reports/03_rcf_final_model_study_le60.pdf`; config in `pipeline_rcf/out_le60/best_model.txt` |

A tuned XGBoost is statistically indistinguishable (R² 0.309); what matters is the large
minimum leaf size, not the library. Unlike RHA, the extra powder descriptors do help here,
because the RCF corpus has roughly twice as many papers.

---

## How performance was measured, and what to expect

Both figures come from repeated grouped hold-out: 30% of **papers** held out, 40 repeats,
so no paper contributes to both training and testing. This is deliberately harsher than a
random row split, which would leak a paper's own control level into its test rows.

Expect a typical error near **8–8.5 MPa** on a mix from a paper the model has never seen,
and treat predictions outside roughly 1–60 MPa as extrapolation; the scripts flag these.
Neither model predicts the optimal replacement level reliably — both studies found that
target ill-posed on these corpora.
