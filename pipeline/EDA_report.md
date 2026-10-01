# RHA Concrete — EDA & Baseline Modeling Report

Pipeline over `RHA Master sheet v2.csv` (dataset extracted from ~100 research
papers on rice-husk-ash blended cement/concrete). Goal: predict **compressive
strength** and the **optimal cement-replacement level** (the % that maximizes
strength).

Reproduce: run `pipeline/01_load_clean.py` → `02_parse_explode.py` →
`03_features.py` → `04_model.py` → `05_plots.py`. Artifacts land in `pipeline/out/`.

---

## 1. What the dataset actually is

- Raw file is **1000 × 225** with a **4-row hierarchical header**: category →
  section → feature (55 features) → sub-field. Each feature is split into 4
  columns: `Extracted Value`, `Unit`, `Source Type`, `Exact Evidence`.
  Only `Extracted Value` (+ `Unit`) is predictive; `Source Type`/`Exact Evidence`
  are provenance and are dropped. Encoding is **latin-1**, not utf-8.
- **Only 79 rows are real** (79 unique papers); the other **917 are blank spacer
  rows**. So "mostly null" is two problems: blank rows (trivially dropped) and
  genuine feature sparsity across the 79 papers.
- Targets among the 79 papers: Compressive Strength in 70, Optimum Replacement in
  71, Cement Replacement Level(s) in 76.

## 2. The key data-representation insight

Each Compressive Strength cell packs a **grid of (replacement %, curing age) →
MPa** in ~12 different textual layouts, e.g.
`"3 days: 20 (7%), 20 (10%); 28 days: 44.5 (7%), 46 (10%)…"`.

`02_parse_explode.py` parses these into a **tidy long table** with one row per
`(paper_id, replacement_pct, age_days) → strength_mpa`. This is the single most
valuable step: it turns ~70 packed rows into **756 clean data points across 44
papers**. Precision safeguards:
- unsigned number matching (fixes `RHA8-10%` → 10, not −10);
- a **per-paper whitelist**: a resolved % must match the paper's declared
  replacement levels (∪ {0} for control), which rejects mislabeled sample IDs
  (`RHA1`, `M2`) and recovers trailing-code mixes (`Aa-10`, `A35-10`);
- sanity clamps (0 ≤ % ≤ 100, 0 < age ≤ 400, 0.5 ≤ MPa ≤ 250).

**Parse yield:** 44 / 70 CS papers yielded (756 points; 192 at 28-day). The 26
non-yielding papers are graph-derived narratives (4), pure ranges (no per-level
values), bare number lists with no age/% mapping (6), and unmappable mix IDs.
These are genuinely unrecoverable without the source PDFs — see
`out/parse_failures.csv`.

## 3. Feature engineering (`03_features.py`)

- Numeric features (oxides, fineness, w/b, burn temp, …) parsed as the mean of
  numbers in each cell (handles `"94.6 (RHA-A), 93.7 (RHA-B)"`).
- Categorical text **canonicalized** into small clean variables:
  `activation_method` {none/thermal/mechanical/both}, `mineralogy`
  {amorphous/mixed/crystalline}, `processing_method`
  {controlled/uncontrolled/…}, `pozzolanic_reactivity` ordinal.
- Fill-rates among yielding papers: SiO₂ 89%, oxides 60–77%, w/b 80%, fineness
  (BET/Blaine/SSA) only 18–25%. No feature exceeded the 95%-missing drop rule.
- **Imputation:** none applied — XGBoost / HistGradientBoosting handle NaN
  natively (learned default split direction), which is best for sparse trees.

## 4. Modeling results (`04_model.py`)

Model: **XGBoost 3.4.1** (HistGradientBoosting fallback). Evaluation:
**repeated grouped hold-out** (GroupShuffleSplit × 25, 30% papers held out) —
reported as mean ± std, because a single split is unreliable at this sample size.
No paper appears in both train and test.

### Compressive strength

| Framing | R² | MAE (MPa) | Note |
|---|---|---|---|
| A: absolute CS, all 44 papers, all features | **−0.46** | 20.4 | ill-posed |
| B: relative-to-control (RHA-blend rows) | −0.16 | 0.22 | poor |
| C1: RHA-blend CS given control, base [%, age, control] | **0.42 ± 0.21** | 10.8 | usable |
| C2: base + chemistry | 0.26 ± 0.62 | 11.1 | chem overfits |
| C3: base + all features | **0.46 ± 0.19** | 9.9 | best structured |
| C4: base + MiniLM text embeddings | 0.53 ± 0.15 | 9.5 | gain within noise |

**Interpretation.** Absolute strength across papers is essentially
unpredictable (R² < 0) because the data mixes low-strength mortars, normal
concrete and UHPC (1–205 MPa), and each paper's baseline level is set by mix
factors (cement grade, aggregate, superplasticizer, concrete-vs-mortar) that the
sparse RHA features don't capture. Once the paper's **control (0%) strength** is
provided as an input — always known in practice — the RHA effect becomes
learnable: **R² ≈ 0.42–0.46, MAE ≈ 10 MPa.** The strongest signal is just
`(replacement %, curing age, control strength)`; adding chemistry does not help
reliably (and can hurt) at n = 23 papers.

### Optimal replacement level

| Method | MAE (pts) | Mean-baseline MAE |
|---|---|---|
| Argmax of CS model at 28 d | 8.2 | 3.5 |
| Direct per-paper regressor | 6.5 | 5.3 |

Neither method beats simply predicting the mean optimum (~10–12%). The optimum
has low variance across papers and only ~21–43 labels exist, so there is not
enough signal to beat the naive baseline yet.

### The BERT / embeddings question — answered empirically

Text embeddings (MiniLM, and a TF-IDF+SVD fallback) gave the numerically highest
R² (0.53) but the gain over the simple base model (0.42) is **within the
run-to-run noise band (±0.15–0.21)**. With only 23–44 papers, embeddings add heavy
dependencies for no reliable benefit. **Recommendation: do not use embeddings for
this dataset size.** The short domain phrases are far better captured by the
canonicalized categoricals.

## 5. Bottom line & recommendations

1. **The bottleneck is data quantity/quality, not model choice.** XGBoost vs
   HistGB vs embeddings all land in R² ≈ 0.4–0.5 with ±0.2 noise. The parsing
   step (70 → 756 points) mattered far more than any modeling choice.
2. **Best usable model today:** XGBoost predicting RHA-blend strength from
   `(replacement %, curing age, control strength)` → MAE ≈ 10 MPa. Report the
   optimal replacement as the argmax of this curve, but flag that it does not yet
   beat a naive ~10–12% prior.
3. **Highest-leverage next steps:**
   - Recover more data: re-extract the 26 unparsed papers (especially the
     graph-derived and bare-list ones) with a structured schema; digitize
     graph-only strength curves.
   - Capture the missing **base-mix descriptors** (cement grade/type, aggregate,
     superplasticizer, concrete-vs-mortar, target grade, curing regime) — these
     dominate absolute strength and are currently absent.
   - Only after more data: revisit model complexity; embeddings and deep models
     are premature at n ≈ 40 papers.

## Artifacts (`pipeline/out/`)

`papers_wide.csv` (79×116) · `train_long.csv` (756 CS points) ·
`parse_failures.csv` · `train_features.csv` (modeling table) · `paper_level.csv`
(per-paper features + optima) · `feature_report.txt` · `cv_results.txt` ·
`figs/` (cs_vs_age, cs_vs_replacement, data_availability).
