# Step 14 refactor: implementation and validation

Review/run date: 2026-10-09. Base commit:
`af568ed22a33323870dfb25d77b1712db1623c18` (`main`). Changes are isolated to
Step 14 code, its tests and documentation on `codex/step14-refactor-validation`.

## Implementation

- Full → Reduced A → Reduced B → Minimal is a fixed, pre-fit selection rule.
  Each candidate has its own complete cases, design rank, support, events per
  parameter and missing/constant-field diagnostics. Missing duration no longer
  removes supported age/sex. Estimator failure does not reselect adjustments.
- Interval age/sex/duration remain static official Step 11 baseline values.
  No baseline, episode ordering, thresholds, first-event risk sets, evaluability,
  lab as-of timing or clinical definitions were changed. Steps 10/11/13 and all
  study modules are unchanged.
- Domain lagged sensitivity now uses FROM total minus the **config-weighted
  ordinal** contribution. Articular score 2 subtracts 8; biological score 2
  subtracts 2. Canonical config aliases gland_swell→glandular and
  neuro_periph→pns are explicit. Missing/impossible composition and complete-panel
  reconstruction conflicts exclude sensitivity values with aggregate QC.
- Historical family BH remains `q_value`, with `q_value_family` alias.
  Independent outcome-wide BH uses valid canonical primary hypotheses within
  analysis × outcome × effect measure. Sensitivities, alternate/duplicate
  representations and failed models have missing outcome-wide q-values with
  exclusion reasons. Neither correction is selected by its significance.
- Glandular ESSDAI ordinal is canonical; salivary gland swelling remains an
  alternate representation in CSVs, excluded from primary outcome-wide BH/forest
  plots. This policy is prespecified, not recalculated from correlation.
  Redundancy QC requires at least 20 complete pairs and variation; n=3 with
  correlation 1 is explicitly insufficient evidence.
- Claim tiers, formulas, weights/sources, actual estimators and selected levels
  are retained. Rejected MixedLM attempts remain in QC; Gaussian GEE is the
  prespecified fallback. Existing outputs are backed up before replacement.
  Run metadata includes rule versions, context, hashes, Git/worktree state,
  dependencies and actual trajectory estimator/fallback counts.

## Duration source audit

Canonical Step 11 Parquet, variable dictionary and availability exports are
absent from this checkout. Source inspection confirms that
`add_demographic_history_derivations` creates `disease_duration` only when
`dx_date` exists, potentially from the canonical Sjögren diagnosis-date field.
Its official baseline anchor equals clinical_baseline_date under the existing
input contract. No later diagnosis or follow-up is used to recover missing data.

Pharma's `time_since_diagnosis_years` and `disease_duration_years` are name hints,
not proof of presence/definition in Step 11. No producer/schema evidence was
found for `demo__disease_duration`. These unverified aliases are not resolved;
the previously accepted duration aliases are retired under the documented
source policy. Config's legacy `disease_duration_yrs_model` is documented as
clipped at zero and is not adopted. No upstream derivation was changed without
canonical schema evidence. Actual negative, longer-than-age or future-diagnosis
duration values are excluded with availability/QC counts rather than clipped.
When a diagnosis-date field exists but is missing, duration remains unknown.

## Schema and behavior comparison

All **64 historical result columns** were checked against the base commit's
`prepare_model`/`result_template` for trajectory, lagged, Cox and Poisson tasks:
none disappeared. Existing model/availability/registry CSVs retain their columns.

| Interface | Before | After |
| --- | --- | --- |
| Adjustment status | full / prespecified_reduced | full / reduced_A / reduced_B / minimal; selected level none if unsupported |
| Legacy adjustment label | adjustment_status | additional adjustment_status_legacy alias |
| Candidate audit | full and selected counts | all four levels in JSON and flattened adjustment QC |
| Lagged domain sensitivity | without_baseline_essdai, unchanged state formula | domain_from_other_domains_adjusted, weighted other-domains state |
| Baseline omission label | without_baseline_essdai | without_baseline_total; omission still applies only to baseline total |
| Multiplicity | family q_value | unchanged family q_value plus q_value_family and q_value_outcome_wide |
| Glandular representation | independently presented results | explicit canonical/alternate construct registry; CSV models preserved |
| Redundancy support | ≥3 pairs | ≥20 pairs; small support remains not evaluable |
| Interpretation | interpretation_status | retained plus claim_status / inference_tier |
| Additional QC | none | adjustment_selection, domain_composition, outcome_wide_fdr |

Deterministic synthetic tasks with missing duration and supported age/sex were
run through the base commit and the refactor. All four retained **120 model
rows before and after**; counts are synthetic, not cohort results:

| Model | Previous adjustment | New adjustment / selected level |
| --- | --- | --- |
| Trajectory | baseline_essdai | baseline_essdai + age + C(sex) / Reduced A |
| Cox | baseline_essdai | baseline_essdai + age + C(sex) / Reduced A |
| Lagged | age, plus FROM ESSDAI/interval terms | age + C(sex), plus FROM ESSDAI/interval terms / Reduced A |
| Poisson | baseline_essdai, plus bands/offset | baseline_essdai + age + C(sex), plus bands/offset / Reduced A |

This comparison establishes behavior/schema preservation, not scientific effect
stability. Productive coefficients, confidence intervals and p/q-values were
not calculated or compared.

## Executed validation

Python 3.12 environment with the repository's installed analysis dependencies:

```bash
python -m pytest -q tests/test_risk_factors_progression.py
python -m pytest -q tests/test_integrated_longitudinal_dataset.py \
  tests/test_integrated_baseline_characterization.py \
  tests/test_disease_activity_progression.py
```

- Step 14: **55 passed**. Steps 10/11/13: **79 passed**. Total: **134 passed**.
  Existing NumPy timedelta deprecation warnings were reported; no failed tests.
- Tests cover all four adjustment levels, duration present/absent, categorical
  missingness, fixed-cohort outcome perturbations, events-per-parameter reduction,
  0/1/2 events, baseline propagation, leakage, first-event/censoring contracts,
  weighted composition, invalid/missing panels, real supported lagged GEE,
  supported Cox/Poisson including offset and clustering, independent BH vectors,
  duplicate exclusions and estimator-group separation, sparse-pair QC, rejected
  MixedLM/fallback diagnostics, full output smoke tests, backups and metadata.
- A separate CLI `--dry-run` completed on explicitly synthetic Step 10/11/13
  fixtures: **60 baseline patients, 240 episodes, 1360 planned associations,
  72 supported**. All domain totals were reconstructed using config weights and
  duration was deliberately absent. This dry-run fitted no models. The synthetic
  QC had 6 structural and 12 temporal checks PASS, with zero invalid subtraction
  or complete-panel reconstruction mismatches. Supported adjustment levels were
  Reduced A 46, Reduced B 6 and Minimal 20; reductions reflect rank/support gates.
  Each domain had 180 available FROM compositions.
  The synthetic
  fixtures and run outputs are temporary validation assets, not committed
  scientific exports.
- A full synthetic small-support run in the tests verified every required result
  and QC CSV, all three Parquets and the three empty-inference PDFs. Supported
  inference and estimator fallbacks are separately exercised by the fitting tests.
- `git diff --check` passed.

## Productive validation status

The default productive `--dry-run` was attempted with an isolated output root.
It failed with **FileNotFoundError** for
`data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet`.
No productive run or regeneration is claimed. Canonical Step 11/13 exports,
the context companion and the supplied review snapshot CSVs are also absent;
the snapshot numbers below come exclusively from the implementation request.

| Dimension | Review snapshot | Productive refactor run |
| --- | --- | --- |
| Baseline patients / episodes | 159 / 497 | Pending canonical inputs; must remain identical |
| Structural QC | 6 PASS | Pending productive validation |
| Duration available | 0 / 159 | Pending schema audit; legitimate missingness accepted |
| Valid primary models | 115 | Pending, by outcome and canonical role |
| Fully adjusted models | 0 | Pending; no target count forced |
| Supported slope models using GEE | 17 / 17 | Pending; actual method/attempts now instrumented |
| Family q < .05 findings | 4 | Pending effect/CI/p/q comparison |
| Outcome-wide q < .05 findings | Prior snapshot sensitivity only | Pending; separate correction, canonical hypotheses |
| Domain lagged sensitivity | Redundant baseline-omission label | Correct formula verified synthetically; productive inference pending |
| Glandular primary multiplicity | Redundant representations | One prespecified canonical representation |
| Estimable ESSDAI ≥5 models | 11 baseline Cox / 0 TV | Pending support-based comparison |
| Supported cross-domain models | 3 | Pending risk/support comparison |

No claim is made about which scientific associations persist. Once canonical
products are available, rerun dry-run then full Step 14, review all structural,
temporal, adjustment/composition/FDR/attempt QC, and compare samples, events,
adjustments, estimators, effects, confidence intervals, p-values and both q-values
separately. Patient/episode/event drift must be investigated before accepting
results. Backups protect prior outputs; no patient data belong in this note or
public validation assets.

The optional TO other-domains outcome sensitivity is not implemented. Treatment
adjustment remains outside Step 14. Results describe observational associations,
not causality or a validated clinical prediction model. Changes in significance
are not an implementation success criterion.
