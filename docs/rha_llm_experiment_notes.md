# RHA LLM Prediction Experiment — Results So Far

## 1. Objective

We are testing whether a large language model (Opus/Fable) can predict compressive strength and, ultimately, optimal RHA replacement directly from the cleaned RHA literature dataset.

The experiment has two strength-prediction conditions:

1. **WITH_CONTROL** — the paper's 28-day 0% RHA control strength is available as an input.
2. **NO_CONTROL** — control strength is deliberately withheld because it may not be available during real inference.

The two conditions form an **ablation study**: the test papers are held fixed, and the only intended difference is whether control strength is supplied.

The source report states that the original RHA Master sheet contains 79 papers and that strength-cell parsing recovered 756 `(paper, replacement %, age)` strength points across 44 papers. The report emphasizes that parsing the packed free-text strength cells is a major part of the problem. It also reports that the data are highly heterogeneous, spanning very different concrete/mortar strength regimes. 

---

# 2. Dataset and Cleaning Background

The RHA Master sheet has a four-row hierarchical header and 225 columns. Each feature is represented by columns for:

- Extracted Value
- Unit
- Source Type
- Exact Evidence

The report states that the first two are predictive information, while the latter two are provenance fields that can be discarded.

The strength parser converts packed free-text strength cells into a tidy long representation:

`paper × replacement % × curing age → compressive strength`

The report describes safeguards including:

- unsigned-number matching to avoid interpreting identifiers such as `RHA8-10%` as `-10`;
- per-paper replacement-level whitelists;
- rejection of mislabeled sample IDs;
- physical sanity constraints:
  - `0 ≤ replacement % ≤ 100`
  - `0 < age ≤ 400 days`
  - `0.5 ≤ strength ≤ 250 MPa`

The report recovered 756 points across 44 strength-yielding papers, including 226 points at 28 days.

---

# 3. Why Control Strength Matters

The central modeling difficulty is that absolute strength varies substantially across papers because the corpus contains different material/mix regimes.

The report explicitly compares three target framings:

### A. Absolute strength without control

Predict MPa directly from RHA/material inputs.

### B. Relative-to-control strength

Predict strength normalized by the paper's own 0% control strength.

### C. Absolute strength with control

Predict blended concrete strength using the paper's 0% control strength as an input.

The reported results were:

| Framing | R² | RMSE (MPa) | MAE (MPa) |
|---|---:|---:|---:|
| **C: Blend \| control strength** | **0.395** | **16.45** | **10.64** |
| B: Relative-to-control | -0.159 | 0.29 | 0.22 |
| A: Absolute, all papers | -0.776 | 30.99 | 20.95 |

The report therefore concluded that absolute strength is difficult to predict across papers without a baseline-strength signal, while supplying the control strength makes the RHA effect substantially more learnable.

---

# 4. Existing ML Baseline From the Report

The report compared several models under the control-strength formulation.

## Model comparison

| Model | R² | RMSE (MPa) |
|---|---:|---:|
| Random Forest | **0.436** | **15.82** |
| XGBoost | 0.395 | 16.45 |
| Mean baseline | -0.310 | 23.97 |
| HistGradientBoosting | -0.524 | 22.28 |
| Ridge | -14.7 | 44.74 |

Random Forest had the highest mean R² in that particular study, but the report notes that the model-to-model gap was small compared with run-to-run variation. XGBoost was carried forward because it supported the monotonic constraints used in the later physics-prior study.

---

# 5. Feature-Set Ablation From the Report

The report found that a **very small feature set was preferable** on this dataset.

The best feature set was:

`{replacement %, age, control strength}`

Feature-set results:

| Feature set | R² | RMSE (MPa) |
|---|---:|---:|
| **base [replacement, age, control]** | **0.442** | **15.54** |
| base + fineness | 0.435 | 15.73 |
| base + categorical | 0.427 | 15.93 |
| all + MiniLM text | 0.426 | 16.13 |
| all + TF-IDF text | 0.403 | 16.42 |
| all structured | 0.395 | 16.45 |
| base + all numeric | 0.349 | 16.85 |
| base + chemistry | 0.317 | 17.18 |
| base + processing | 0.236 | 17.69 |

The interpretation in the report is that, at this sample size, adding more features tends to encourage overfitting rather than improving generalization.

---

# 6. Physics-Prior Ablation From the Report

The report also compared target transformations and monotonic constraints.

| Prior / transformation | R² | RMSE (MPa) |
|---|---:|---:|
| **monotone(age, control)** | **0.482** | **15.02** |
| log + monotone | 0.458 | 15.71 |
| plain | 0.442 | 15.54 |
| log-target | 0.423 | 16.09 |

The final ML model therefore combined:

- control-strength formulation;
- the three base features;
- monotonicity in age and control strength;
- shallow XGBoost;
- max depth = 2;
- learning rate = 0.03;
- 300 trees.

Its repeated grouped hold-out performance was:

| Metric | Final ML model |
|---|---:|
| **R²** | **0.50 ± 0.20** |
| **RMSE** | **14.9 MPa** |
| **MAE** | **10.1 MPa** |

The report stresses that this estimate comes from repeated grouped hold-outs by paper. A separate pooled single 5-fold GroupKFold visualization produced R² = 0.63 and MAE = 9.4 MPa, but that pooled number is more optimistic and is not the headline estimate.

---

# 7. LLM Experiment: Fixed Test Set

For the LLM experiment, the test set was deliberately constructed at the **paper level**, so no paper is represented in both training and testing.

The strength test set contains:

- **6 held-out papers**
- **57 strength observations**

The same six papers are used for both the WITH_CONTROL and NO_CONTROL strength ablations.

The six test papers are:

- P036
- P022
- P007
- P030
- P054
- P064

This gives a fixed common benchmark for the control-strength ablation.

---

# 8. LLM Strength Results — NO_CONTROL

For the NO_CONTROL condition, Opus was given the strength training data without access to the paper's control strength during inference.

The submitted predictions contained 57 test points.

## Overall metrics

| Metric | Opus — NO_CONTROL |
|---|---:|
| **N test points** | **57** |
| **MAE** | **12.83 MPa** |
| **RMSE** | **19.50 MPa** |
| **R²** | **0.392** |
| **Mean signed error / bias** | **-6.25 MPa** |
| **Median signed error** | **+1.32 MPa** |

The main interpretation is that the model has moderate overall explanatory power but tends to underestimate the absolute strength level, particularly for high-strength papers.

---

## NO_CONTROL performance by strength regime

The test points were grouped into:

- Low: `< 30 MPa`
- Middle: `30–60 MPa`
- High: `≥ 60 MPa`

| Strength regime | N | MAE (MPa) | RMSE (MPa) | Bias (MPa) |
|---|---:|---:|---:|---:|
| **Low (<30 MPa)** | 24 | **7.31** | **8.06** | **+7.31** |
| **Middle (30–60 MPa)** | 18 | **3.19** | **3.73** | **-1.87** |
| **High (≥60 MPa)** | 15 | **33.22** | **36.40** | **-33.22** |

The strongest performance is in the middle regime.

The major weakness is the high-strength regime, where the model underpredicts substantially.

---

## NO_CONTROL performance by paper

| Paper | N | MAE (MPa) | RMSE (MPa) | Bias (MPa) |
|---|---:|---:|---:|---:|
| **P007** | 12 | **2.37** | **2.67** | **-0.38** |
| **P022** | 12 | **4.77** | **4.97** | **+4.77** |
| **P030** | 6 | **4.85** | **5.25** | **-4.85** |
| **P036** | 12 | **9.86** | **10.26** | **+9.86** |
| **P054** | 9 | **23.40** | **23.60** | **-23.40** |
| **P064** | 6 | **47.95** | **49.77** | **-47.95** |

The dominant failure is P064.

Example P064 predictions:

| Replacement | Age | True strength | NO_CONTROL prediction |
|---:|---:|---:|---:|
| 25% | 28 d | 111.3 MPa | 40.0 MPa |
| 15% | 28 d | 94.67 MPa | 44.0 MPa |
| 40% | 28 d | 90.42 MPa | 34.0 MPa |

This is a clear example of the difficulty of inferring the absolute paper-specific strength scale without a baseline-strength input.

---

# 9. LLM Strength Results — WITH_CONTROL

A corrected WITH_CONTROL test file was subsequently generated because the earlier version appeared corrupted.

The cleaned file was verified as:

- **57 test rows**
- **31 columns**
- consistent CSV formatting
- UTF-8 encoding
- control strength retained as an input column

Opus was then run on the WITH_CONTROL strength test set.

## Overall metrics

| Metric | Opus — WITH_CONTROL |
|---|---:|
| **N test points** | **57** |
| **MAE** | **5.84 MPa** |
| **RMSE** | **8.79 MPa** |
| **R²** | **0.877** |
| **Mean signed error / bias** | **-2.21 MPa** |
| **Median signed error** | **+0.30 MPa** |

---

# 10. Direct LLM Ablation: WITH_CONTROL vs NO_CONTROL

The most important result so far is the fixed-test-set ablation.

| Metric | WITH_CONTROL | NO_CONTROL | Change |
|---|---:|---:|---:|
| **R²** | **0.877** | 0.392 | **+0.485** |
| **MAE** | **5.84 MPa** | 12.83 MPa | **54.5% lower with control** |
| **RMSE** | **8.79 MPa** | 19.50 MPa | **54.9% lower with control** |
| Mean bias | -2.21 MPa | -6.25 MPa | bias reduced |

Thus, on this fixed six-paper/57-point benchmark, supplying the paper's 28-day control strength produces a very large improvement.

The central empirical finding is:

> **The LLM can infer the absolute strength scale much better when the paper's 0% RHA control strength is supplied.**

This is consistent with the existing ML analysis, which identified control strength as one of the dominant predictive signals.

---

# 11. WITH_CONTROL Performance by Strength Regime

| Strength regime | N | MAE (MPa) | RMSE (MPa) | Bias (MPa) |
|---|---:|---:|---:|---:|
| **Low (<30 MPa)** | 24 | **3.84** | **4.62** | **+3.70** |
| **Middle (30–60 MPa)** | 18 | **3.85** | **4.50** | **-2.89** |
| **High (≥60 MPa)** | 15 | **11.43** | **15.33** | **-10.85** |

The most important improvement is in the high-strength regime:

- NO_CONTROL high-strength MAE = **33.22 MPa**
- WITH_CONTROL high-strength MAE = **11.43 MPa**

So the high-strength MAE falls by about two-thirds once control strength is supplied.

---

# 12. WITH_CONTROL Performance by Paper

| Paper | N | MAE (MPa) | RMSE (MPa) | Bias (MPa) |
|---|---:|---:|---:|---:|
| **P007** | 12 | **2.75** | **3.34** | **-1.32** |
| **P022** | 12 | **2.26** | **2.44** | **+2.23** |
| **P030** | 6 | **6.05** | **6.20** | **-6.05** |
| **P036** | 12 | **5.42** | **6.06** | **+5.17** |
| **P054** | 9 | **4.81** | **5.94** | **-3.83** |
| **P064** | 6 | **21.37** | **23.13** | **-21.37** |

P064 remains the hardest paper, but the error is substantially reduced relative to the NO_CONTROL condition.

---

# 13. Important Comparison With the Existing ML Model

The existing ML report gives:

> **Final ML: R² = 0.50 ± 0.20, RMSE = 14.9 MPa, MAE = 10.1 MPa**

The fixed LLM test-set results are:

### Opus WITH_CONTROL

> **R² = 0.877, RMSE = 8.79 MPa, MAE = 5.84 MPa**

### Opus NO_CONTROL

> **R² = 0.392, RMSE = 19.50 MPa, MAE = 12.83 MPa**

However, these are **not yet a strictly apples-to-apples comparison** with the ML report's numbers.

The reason is that:

- the LLM numbers above are obtained on the fixed six-paper/57-point test set;
- the headline ML number is a repeated grouped hold-out estimate over many different paper splits.

Therefore we should not claim that Opus definitively outperforms the ML model yet.

The important clean comparison we already have is:

> **Opus WITH_CONTROL vs Opus NO_CONTROL on exactly the same held-out test papers.**

That ablation is valid and shows a very large benefit from control strength.

---

# 14. Optimal Replacement: Current Understanding

There are two possible ways to predict the optimum replacement.

## Approach A — Train a direct optimum predictor

Input paper characteristics and predict:

`optimal_replacement_pct`

This is a conventional supervised-learning formulation.

The existing report tried this and found that the direct optimum problem is extremely difficult on the current dataset.

Reported results:

- direct per-paper regressor:
  - MAE = **6.5 percentage points**
  - baseline MAE = **5.3 percentage points**
- only 20 held-out papers were available for the argmax evaluation described in that section;
- the argmax approach produced **7.5% for every one of the 21 held-out papers**;
- the corpus-mean optimum was **11.5%**;
- the mean-prior baseline achieved MAE = **3.45 percentage points**.

The report therefore concluded that the current corpus does not provide enough information to reliably learn paper-to-paper optimum shifts.

---

# 15. Better Approach for the LLM: Strength Model → Replacement Sweep → Argmax

A more natural approach is:

```text
Paper information
       +
Control strength (if available)
       ↓
Strength prediction model
       ↓
Evaluate many candidate RHA replacement levels
       ↓
Predict 28-day strength at each replacement
       ↓
Choose replacement giving maximum predicted strength
       ↓
Predicted optimum replacement %
```

For example:

| Replacement | Predicted 28-day strength |
|---:|---:|
| 0% | 50.0 |
| 2.5% | 52.1 |
| 5% | 55.3 |
| 7.5% | 57.0 |
| 10% | **58.2** |
| 12.5% | 57.5 |
| 15% | 55.8 |
| 20% | 52.0 |

The derived optimum would be **10%**.

This is conceptually attractive because the model that predicts optimum is then the same model that predicts compressive strength. We are not asking a separate small model to learn a difficult optimum label from very few papers.

---

# 16. The Main Limitation of the Strength-Sweep Approach

The optimization operation itself is straightforward.

The difficult part is whether the learned **strength-versus-replacement curve** has the correct shape.

A strength model can have reasonable pointwise MAE while still placing the maximum at the wrong replacement level.

For example, the real relationship might be:

```text
5%   → 62 MPa
10%  → 70 MPa   ← true optimum
15%  → 68 MPa
20%  → 60 MPa
```

while the model predicts:

```text
5%   → 63 MPa
10%  → 66 MPa
15%  → 69 MPa   ← predicted optimum
20%  → 65 MPa
```

The individual errors are not necessarily catastrophic, but the predicted optimum shifts from **10% to 15%**.

Therefore:

> Good strength prediction does not automatically guarantee good optimum prediction.

The model needs to get the **relative ordering of replacement levels** correct, not merely the overall strength scale.

---

# 17. Why the Existing ML Optimum Approach Failed

The report found that the final monotone, shallow XGBoost model produced the same optimum:

> **7.5%**

for every held-out paper.

The report explains that the model architecture had very limited interaction between replacement and the other predictors.

Because the model was monotone in age and control strength and had max depth = 2, changes in control strength primarily shifted predicted strength curves vertically rather than moving their replacement-level maximum.

Therefore different papers inherited the same argmax.

This is an important warning:

> A strong strength predictor is not automatically a useful optimizer if its functional form does not allow the replacement response curve to change shape across papers.

---

# 18. Physical Pattern in the Dataset Relevant to Optimum Prediction

The report's within-paper analysis found that the replacement-strength curve is not uniformly unimodal.

Among 33 papers with at least three distinct replacement levels at 28 days:

- **14 papers (42%)** peaked at an interior replacement level;
- **12 papers (36%)** were highest at the lowest level tested, often 0%;
- **7 papers (21%)** were still increasing at the highest replacement level tested.

Therefore the optimum is not necessarily:

> "somewhere in the middle."

It can be:

- 0%;
- an interior replacement;
- the highest tested replacement level.

This makes flexible curve prediction more important than imposing a fixed bell-shaped response.

---

# 19. Recommended Optimum Experiment

For the LLM experiment, the cleanest next experiment is:

### WITH_CONTROL

For each test paper:

1. Provide the paper features and control strength.
2. Have Opus predict 28-day compressive strength for a predefined grid of RHA replacement levels.
3. Use the maximum predicted strength to derive the optimum.

### NO_CONTROL

Repeat exactly the same procedure, but remove control strength.

Then compare:

| Optimum method | WITH_CONTROL | NO_CONTROL |
|---|---:|---:|
| Direct LLM optimum prediction | later | later |
| Strength-model sweep → argmax | **recommended** | **recommended** |

The ground truth remains the reported experimental optimum.

This produces a much cleaner causal/ablation question:

> **Does giving the LLM the paper's control strength improve its ability to reconstruct the strength-vs-replacement curve well enough to identify the optimum?**

---

# 20. Current Results Summary

## Strength prediction

| Experiment | R² | MAE (MPa) | RMSE (MPa) |
|---|---:|---:|---:|
| **Opus — WITH_CONTROL** | **0.877** | **5.84** | **8.79** |
| **Opus — NO_CONTROL** | **0.392** | **12.83** | **19.50** |
| Existing final ML model | 0.50 ± 0.20 | 10.1 | 14.9 |

Again, the first two are fixed-test-set LLM results and the ML result is a repeated grouped hold-out estimate, so the LLM-vs-ML rows should not yet be treated as a perfectly matched benchmark.

## Control-strength ablation

Providing control strength caused:

- **R² improvement:** `0.392 → 0.877`
- **absolute R² gain:** `+0.485`
- **MAE reduction:** `12.83 → 5.84 MPa`
- **MAE reduction:** **54.5%**
- **RMSE reduction:** `19.50 → 8.79 MPa`
- **RMSE reduction:** **54.9%**

The largest practical effect is on high-strength papers:

- NO_CONTROL high-strength MAE = **33.22 MPa**
- WITH_CONTROL high-strength MAE = **11.43 MPa**

## Optimum prediction

No final LLM optimum result has been evaluated yet.

The current recommendation is to derive optimum by:

> **strength prediction → sweep replacement levels → choose the maximum predicted 28-day strength**

rather than relying primarily on a separate direct optimum model.

---

# 21. Bottom-Line Interpretation So Far

The results support a strong and coherent conclusion:

**1. Control strength is extremely important.**

The fixed-test LLM ablation shows a large improvement when the 28-day 0% RHA strength is supplied.

**2. Without control strength, the LLM can still learn broad age/replacement behavior, but it has difficulty inferring the absolute strength scale of a new paper.**

This is most obvious in the high-strength regime.

**3. The LLM's WITH_CONTROL strength performance is encouraging.**

On the fixed test set, it achieves R² = 0.877 and MAE = 5.84 MPa.

**4. Optimum prediction should probably be formulated as an optimization of the predicted strength curve rather than as an independent regression target.**

**5. The central challenge for optimum prediction is curve-shape accuracy.**

The model needs to correctly determine which replacement level is better than neighboring levels, not merely predict the average strength accurately.

**6. The existing ML results show that the current dataset is too small and heterogeneous for a simple direct optimum model to learn reliable paper-specific optima.**

---

# 22. Current Experimental Status

### Completed

- RHA Master sheet cleaning/parsing
- train/test construction
- WITH_CONTROL strength split
- NO_CONTROL strength split
- fixed six-paper strength test benchmark
- Opus NO_CONTROL strength predictions
- Opus WITH_CONTROL strength predictions
- evaluation of both strength ablations
- comparison of control vs no-control performance

### Still to evaluate

- WITH_CONTROL optimum via strength sweep
- NO_CONTROL optimum via strength sweep
- potentially direct LLM optimum predictions for comparison
- final LLM-vs-ML comparison on a matched evaluation protocol
- detailed optimum-error analysis

---

## Source reference

The dataset construction, grouped validation strategy, ablation studies, final ML results, and optimum-prediction findings summarized above are based on:

**Predicting Compressive Strength and Optimal Replacement Level for Rice-Husk-Ash Blended Concrete from a Sparse Literature Dataset, August 18, 2026.**
