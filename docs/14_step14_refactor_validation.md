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

## Second-review correction and validation

The preceding sections are the historical first-refactor record. This section
records the second review, also on 2026-10-09, on
`codex/step14-second-review`, based on `83a26fe` from `origin/main`.
Only Step 14 code, its tests and the two Step 14 documents were changed.

### Source inspection and root-cause decisions

The second-review CSV/JSON/PDF attachments named in the request are **not
present** in this workspace. Neither are the productive Step 10/11/13
Parquets, the Step 10 variable registry/context, or the Step 11 dictionary.
The numbers 159 patients, 497 episodes, 63 discordant panels, 48 negative
articular subtractions and six failed articular Cox hypotheses are therefore
request-supplied reference observations, not independently reproduced findings.
The reference commit `433ff9827938bd80dcd4e6aa11e6f9f27a628d27`, together with
`git_worktree_dirty=true` and no supplied patch hash, does not identify the
exact code that produced that run.

| P0 | Evidence and confirmed cause | Action and remaining limitation |
| --- | --- | --- |
| Composition | Old aggregate flags overlap and repeat complete-panel counts across domains; partial-panel subtraction lacked documentary inclusion evidence. The productive 63/48 causes and their overlap are **not determined**. | `derive_other_domain_adjustments` adds an exclusive priority partition, full discordant × negative intersection, unique-FROM delta distribution, exact reconstruction and stricter partial-panel provenance. No totals, weights, ordinal scores or TO outcomes are corrected. Productive episode-level diagnosis is `source_unavailable`. |
| Duration | Canonical producer can derive duration from `dx_date`, but productive schema/coverage are unavailable. Other similarly named variables do not establish a source. | `duration_source_report` records the explicit decision, definition, baseline anchor, coverage, reason and version. Synthetic inputs have `not_available`, 0/60. Verified canonical dates without duration yield a separate upstream proposal; Step 14 derives nothing. Productive source audit is `source_unavailable`. |
| Static age/sex reporting | `build_predictor_availability` inspected raw integrated columns while interval construction uses official baseline demographics by patient ID. | Preserve raw availability and add effective interval and exact selected complete-case scopes. This is a confirmed reporting inconsistency, not a new demographic derivation. Tests verify constant propagation, missing baseline age and rejection of TO age. |
| Six articular Cox failures | The generic warning does not distinguish separation, invalid covariance, design problems or numerical exceptions. Productive designs are unavailable, so their causes are **not determined**. | `fit_baseline_cox_model` records per-attempt aggregate design/group diagnostics, typed warnings/exceptions, raw coefficient/SE/covariance diagnostics and concrete failure reasons, including successful controls. Invalid inference remains NA; no convergence-driven reduction or penalized rescue. |
| Execution identity | Commit alone is inadequate for dirty code. | `runtime_code_fingerprint` records branch, dirty state, binary-diff SHA-256, every code/config file hash and a code-tree hash without exporting patch content. Metadata is written before input loading; unidentifiable code cannot be labeled fully reproducible. |

Zero-domain provenance is explicitly unknown when upstream metadata cannot
distinguish observed from filled zeros. Producer names and an arithmetically
matching total are not substituted for documentary proof on partial panels.
No upstream producer, study module, clinical definition, support gate,
episode ordering, risk-set rule or pre-fit adjustment hierarchy was changed.

### QC before/after and scientific comparison

The prior first-refactor **synthetic** inputs were reused byte-for-byte. All
1,360 historical hypotheses retain their pre-fit patient, interval, event and
complete-case counts, adjustment level/covariates and support status. No
historical feasibility columns disappeared. The new 12 separately labeled
`complete_concordant_panel_only` hypotheses bring the total to 1,372; three
are supported in these synthetic inputs, giving 75 supported instead of 72.
This is an added sensitivity, not increased support for historical hypotheses.

| Check | Historical synthetic validation | Second-review synthetic validation |
| --- | --- | --- |
| Patients / episodes | 60 / 240 | 60 / 240, identical input SHA-256 |
| Structural / temporal QC | 6 / 12 PASS | 6 / 12 PASS |
| Available other-domain FROM values | 180 per domain | 180 per domain |
| Complete-panel conflicts / negative articular subtraction | 0 / 0 | 0 / 0 |
| Missing duration | 0/60 available | 0/60 available; explicit `not_available` |
| Planned / supported hypotheses | 1,360 / 72 | 1,372 / 75; all historical counts unchanged |
| Full-run planned hypotheses | Not run on this fixture previously | 75 valid; 1,169 not estimable; 128 descriptive-only |
| Supported primary trajectory estimator | Not fitted in prior dry-run | 5/5 Gaussian GEE after retained MixedLM rejection |
| Canonical primary Cox failures | Not fitted in prior dry-run | 0; does not explain the six productive failures |

A separate 325-row **synthetic** audit fixture deliberately contains 63
discordant panels, 48 negative articular subtractions and 21 missing totals.
The articular 48 are constructed inside the 63 only in that fixture; this is
**not evidence of the productive overlap**. It validates 241 available values,
exclusive-category sums of 325 for each domain, and a distribution counting
304 unique complete/known-total FROM episodes with 63 discrepancies once.
Tests also reject noninteger tiny deltas rather than assuming rounding.

The effective-availability fixture reports baseline age on 117/120 eligible
intervals for 39 patients despite absent raw longitudinal age; each selected
model scope matches its own complete cases. A 40-patient matched-comparison
fixture reduces 120 primary intervals to a shared 80/80 and checks identical
sample hashes and frozen primary demographic adjustments for all 12 domains.
All 36 comparison rows are retained, including nonestimable fits; additional
attempts are labeled separately and never enter primary BH.

Productive N/events, beta/HR/IRR, CI95%, p, historical family q and outcome-wide
q **cannot be compared** without the reference outputs and canonical inputs.
No claim is made that a productive association persists, disappears or changes
significance. The synthetic full run has 40 valid canonical primary hypotheses,
six with family q < .05 and five with outcome-wide q < .05; these are test-data
outputs with no clinical interpretation. The all-hypothesis summary and
matched table retain nonsignificant, unsupported and failed rows as well.
The run-comparison outer join compares every planned hypothesis when a backed-up
reference exists; dry-run comparisons explicitly suppress fitted-inference
claims. Identical nonmissing canonical input hashes enforce patient/episode
invariance. Changed inputs do not falsely enforce the old cohort size.

### Executed checks and commands

Use the installed Python 3.12 environment, with caches outside the checkout:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/workspace/work/step14-second-review/matplotlib
export XDG_CACHE_HOME=/workspace/work/step14-second-review/cache
PYTHON=/workspace/.venvs/obj1_sjd/bin/python
"$PYTHON" -m pytest -q tests/test_risk_factors_progression.py
"$PYTHON" -m pytest -q tests/test_integrated_longitudinal_dataset.py \
  tests/test_integrated_baseline_characterization.py \
  tests/test_disease_activity_progression.py
"$PYTHON" src/block_A/14_risk_factors_progression.py --dry-run \
  --output-root /workspace/work/step14-second-review/production-check
"$PYTHON" src/block_A/14_risk_factors_progression.py --dry-run \
  --integrated /workspace/work/step14-refactor-validation/synthetic_integrated.parquet \
  --baseline /workspace/work/step14-refactor-validation/synthetic_baseline.parquet \
  --progression /workspace/work/step14-refactor-validation/synthetic_progression.parquet \
  --output-root /workspace/work/step14-second-review/synthetic-dry-run
"$PYTHON" src/block_A/14_risk_factors_progression.py \
  --integrated /workspace/work/step14-refactor-validation/synthetic_integrated.parquet \
  --baseline /workspace/work/step14-refactor-validation/synthetic_baseline.parquet \
  --progression /workspace/work/step14-refactor-validation/synthetic_progression.parquet \
  --output-root /workspace/work/step14-second-review/synthetic-full
git diff --check
```

Final tests: **74 Step 14 passed + 79 upstream passed = 153 passed**.
The initial iterations exposed incorrect synthetic test-column assumptions;
they were corrected and the final complete Step 14 suite passed. NumPy
timedelta deprecations and warnings from deliberate nonfinite-design fixtures
remain visible. Tests cover complete/partial/incompatible panels, duplicate
FROM rejection, effective static covariates, absence of duration, real complete
separation, collinearity, extreme scale, nonfinite design, invalid times,
invalid covariance/SE, successful Cox controls, matched fits, independent BH
with six named failed hypotheses, source-schema reporting, fingerprints and
input-hash invariance. Failed models and alternate/sensitivity representations
are excluded from outcome-wide BH; `q_value == q_value_family` is checked.

The productive dry-run failed with the expected missing Step 10 Parquet and
preserved early provenance metadata. Synthetic dry-run and full runs completed.
The full CLI is repeated after committing, against the same synthetic files
and output root, to verify clean committed-source metadata and automatic backup
hashes before push. The definitive commit, clean/dirty state, backup path,
dependencies, parameters and code-tree hash are in that run's metadata; the
output manifest records all generated file hashes. Validation assets stay
outside the repository and are not productive exports.

### Exact regenerated outputs and hashes

Scratch full-run root: `/workspace/work/step14-second-review/synthetic-full`.
The following paths are relative to that root; every listed file, except the
manifest itself, has its exact SHA-256 and size in
`outputs/qc/blockA/14_risk_factors_progression/14_output_manifest.csv`:

```text
outputs/tables/blockA/14_risk_factors_progression/
  14_baseline_essdai_trajectory_models.csv
  14_baseline_ge5_cox_models.csv
  14_baseline_high_activity_models.csv
  14_baseline_new_domain_models.csv
  14_cross_domain_models.csv
  14_cross_domain_support.csv
  14_domain_sensitivity_comparison.csv
  14_model_interpretation_summary.csv
  14_predictor_feasibility.csv
  14_predictor_registry.csv
  14_sensitivity_models.csv
  14_timevarying_essdai_models.csv
  14_timevarying_ge5_models.csv
  14_timevarying_new_domain_models.csv
outputs/qc/blockA/14_risk_factors_progression/
  14_adjustment_selection_qc.csv
  14_analysis_metadata.json
  14_baseline_model_attrition.csv
  14_cox_failure_diagnostics.csv
  14_domain_composition_audit_summary.csv
  14_domain_composition_discrepancy_distribution.csv
  14_domain_composition_qc.csv
  14_domain_riskset_qc.csv
  14_interval_model_attrition.csv
  14_model_qc.csv
  14_multiple_testing_qc.csv
  14_outcome_wide_fdr_qc.csv
  14_output_manifest.csv
  14_predictor_availability.csv
  14_predictor_redundancy_qc.csv
  14_run_comparison.md
  14_run_comparison_summary.csv
  14_structural_qc.csv
  14_temporal_leakage_qc.csv
outputs/figures/blockA/14_risk_factors_progression/
  14_baseline_risk_factor_forest.pdf
  14_cross_domain_heatmap.pdf
  14_essdai_slope_difference_forest.pdf
  14_timevarying_progression_forest.pdf
outputs/logs/14_risk_factors_progression/14_risk_factors_progression.log
data/analytic/blockA/14_risk_factors_progression/
  14_baseline_predictor_dataset.parquet
  14_domain_incidence_risk_set.parquet
  14_lagged_interval_dataset.parquet
```

Backups preserve original relative paths under
`outputs/backups/14_risk_factors_progression/<UTC timestamp>/`; the exact
timestamped location is `output_backup_path` in `14_analysis_metadata.json`.
The backed-up manifest permits verification of preserved original bytes.
Input SHA-256 values, for these **synthetic** fixtures only:

| Input | SHA-256 |
| --- | --- |
| Integrated | `17c06bfb7722b05a881117ece1932e78012966aa2d62c5de2ad72520b9882793` |
| Baseline | `0c8b7829c11b8970b8c3c10a7e254c0d9a05942e3080259897654aed35ea93a5` |
| Progression | `b0835c05ed3f1483f0b8dd8ef0916008994f30ff3ac80af2aa23d641b3f5247d` |

Productive acceptance items requiring those absent files remain
`source_unavailable`: explanation of the 63/48 conflicts, diagnosis of the six
specific Cox failures, physical duration-source coverage, reference 159/497
invariance and scientific before/after inference. No upstream correction is
proposed without identifiable producer/episode evidence. Remaining scientific
limits include biological interval censoring, selection of evaluable episodes,
unadjusted treatment and other residual confounding, and accumulated
anti-Ro/SSA positivity rather than instantaneous serum-titer change.
