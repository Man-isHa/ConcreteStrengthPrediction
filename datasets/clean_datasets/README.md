# Cleaned datasets — RHA and RCF

Modelling-ready tables produced by `pipeline/` (rice husk ash) and
`pipeline_rcf/` (recycled concrete fines/powder) from the two master sheets.
Regenerate with `python export_clean_datasets.py` after re-running either
pipeline. All files are UTF-8 CSV; `cleaned_datasets.xlsx` holds the same four
tables as sheets.

| file | rows × cols | one row = |
|---|---|---|
| `RHA_measurements.csv` | 756 × 31 | one compressive-strength measurement |
| `RHA_papers.csv` | 79 × 33 | one source paper |
| `RCF_measurements.csv` | 1825 × 45 | one compressive-strength measurement |
| `RCF_papers.csv` | 104 × 41 | one source paper |

## Coverage

|  | RHA | RCF |
|---|---|---|
| papers in sheet | 79 | 104 (1 dup title, 2 off-scope) |
| papers yielding strength data | 44 | 85 |
| strength points | 756 | 1825 |
| points with a stated curing age | 756 | 1284 |
| blend rows with a matching control (framing C) | 335 over 23 papers | 823 over 44 papers |
| strength range (MPa) | 1.0 – 205.5 | 0.6 – 184.3 |

Missing values are genuine `NaN` (no imputation applied — the tree models
handle NaN natively, and any imputer must be fit inside a CV fold).

## `*_measurements.csv`

Target and identifiers:

- `paper_id` — join key to `*_papers.csv`; **also the CV grouping key** (never
  let one paper span a train/test split).
- `replacement_pct` — cement replaced by RHA/RCF, % by mass. `0` = control.
- `age_days` — curing age. RCF: may be NaN, see `age_known`.
- `strength_mpa` — compressive strength, the prediction target.
- `ctrl_strength_mpa` — the *same paper's* 0% strength at the *same age*
  (NaN when that paper reports no control at that age).
- `rel_strength` — `strength_mpa / ctrl_strength_mpa`.
- `is_range` (RHA) — value came from a reported range; midpoint stored.

The paper-level columns (chemistry, fineness, processing, categoricals) are
joined onto every measurement row, so these files are usable standalone.

- Oxides, % by mass: `sio2 cao al2o3 fe2o3 mgo so3 na2o k2o loi amorphous_pct`
- Fineness/physical: `mean_particle_um specific_surface bet_surface blaine
  specific_gravity` (+ RCF: `d10 d50 d90 water_absorption bulk_density`)
- Mix: `wb_ratio` (water/binder)
- Processing: RHA `burn_temp_c burn_duration grind_duration`;
  RCF `thermal_temp_c thermal_duration grind_duration grind_speed`
- Canonicalised categoricals:
  - `pozzolanic_reactivity` ∈ {low, medium, high}
  - `activation_method` — RHA {none, thermal, mechanical, both};
    RCF {none, thermal, carbonation, mechanical, chemical, multiple, other}
  - `mineralogy` ∈ {amorphous, crystalline, mixed}
  - `processing_method` — RHA {uncontrolled_burn, controlled_burn, burn_other,
    other}; RCF {crushing, milling, thermal, combined, other}
  - `material_family` (RCF) ∈ {powder, fines, paste, cdw, aac, other}

RCF-only provenance columns: `mix_id`, `mix_activation` (per-mix, free text),
`wb_block` (which w/b sub-grid the value came from), `graph_derived`,
`age_known` (False ⇒ `age_days` is unknown, **not** 28 d), `source`
(`sheet1` grid vs `matrix` = Experimental Matrix sheet).

## `*_papers.csv`

`paper_id`, `paper_title`, `authors`, `journal`, `year`, `doi`, the same
paper-level feature columns as above, plus:

- `reported_optimum_pct` — the optimum the paper itself states.
- `empirical_optimum_pct` — argmax of that paper's own 28-day strength curve
  (NaN if it reports too few levels).
- `n_measurements` — rows contributed to the measurements table (0 ⇒ the
  strength cell could not be parsed; see `parse_failures.csv` in the pipeline).
- `in_scope` (RCF only) — False for the 2 papers the pipeline excluded as
  replacing *aggregate* rather than cement.
- `aggregate_replacement_suspect` — True for papers whose `Material Type` is an
  aggregate/sand substitution **and** which declare no cement replacement
  level. See the note below; also present on the measurement rows so it can be
  filtered in one step.

## Caveats worth carrying into any analysis

1. Both corpora mix pastes, mortars, normal concrete and UHPC. Absolute
   strength is not comparable across papers — model it *given* the paper's own
   control (`ctrl_strength_mpa`), or within-paper.
2. RCF: 180 of 260 (paper, level, age) cells contain several distinct mixes
   whose activation treatment is unrecorded — an irreducible noise floor.
3. The optimum-shape prior differs by material: RHA peaks at an interior level
   in 42% of papers, RCF in only 23% (71% peak at the lowest level tested).

## Known scope gap in the RCF corpus (unresolved)

The pipeline's off-scope filter keys on the *declared cement replacement
levels* cell. Five RCF papers describe an **aggregate or sand** substitution
but leave that cell blank, so they pass the filter and contribute **85 rows
whose `replacement_pct` means replaced aggregate, not replaced cement**:

| paper_id | rows | material |
|---|---|---|
| 0 | 21 | Fine recycled concrete aggregates (FRA) |
| 6 | 24 | FRCA / recycled sand |
| 13 | 7 | Fine Recycled Concrete Aggregates (fine RCA) |
| 28 | 27 | fRCA, recycled concrete powder, AD |
| 66 | 6 | Carbonated Recycled Fine Aggregate (C-RFA) / RFA |

They are flagged, not removed, because papers 28 and 66 also involve powder or
carbonated fines and may be partly in scope — deciding needs the source papers.
Dropping them would change the RCF headline counts (1825/85) and therefore the
tables in `rcf_report.tex`, so nothing downstream has been touched.

To exclude them: `df = df[~df.aggregate_replacement_suspect]`.
The RHA corpus has no such cases.
