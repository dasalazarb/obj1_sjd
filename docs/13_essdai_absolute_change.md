# Step 13: observed absolute ESSDAI change

This extension complements continuous trajectories, adjacent-episode state
transitions and the existing first-observed categorical-event KM analyses.
It does not change Step 10/11, Pop definitions, models or legacy endpoints.
The implementation was audited on branch `work`, starting at
`91f9f120f9309b4173d2767b42cb391818ab88b5` (2026-10-19).

## Source and eligible patients

The sources remain `common.INTEGRATED_LONGITUDINAL_PARQUET` (Step 10,
`patient_id × clinical_episode_id`, official `essdai__total`) and
`common.INTEGRATED_BASELINE_PARQUET` (Step 11). `validate_inputs()` checks the
existing contracts before any calculation. Baseline is the official
`is_clinical_baseline` episode, never the first later nonmissing score.

The unit and denominator are unique patients with an observed official baseline
ESSDAI and at least one valid ESSDAI assessment on a date strictly after
`clinical_baseline_date`. Missing ESSDAI is never zero or imputed. Same-date
baseline companions do not count as follow-up. Missing follow-up scores are
skipped for these endpoints; they still break adjacent categorical transition
chains under the unchanged legacy policy.

Canonical episodes on the same follow-up date are retained. Ordering is stable
by patient, actual date and episode ID; the last valid episode in this order
supplies the last score. Ever flags consider all valid episodes, without
creating multiple patient events or denominators. QC reports both assessment
counts and distinct dates, including extra same-date assessments. This matches
the canonical last-observation policy; date ties are not independent temporal
events. First-event dates are the first **documented** qualifying assessment.

The eligibility QC separates the baseline cohort, baseline ESSDAI availability,
patients with later ESSDAI (also counted when baseline ESSDAI is missing), valid
baseline without follow-up, final evaluable patients and valid ESSDAI episode
counts. Follow-up years and numbers of visits have minimum, quartiles, median
and maximum. Patients with missing baseline are included only in Overall QC,
not assigned a clinical state. Eligibility is independent of model eligibility.

## Definitions and interpretation

For baseline score B, subsequent observed scores E(t), and last evaluable
subsequent score L:

- `delta_essdai_last = L - B`.
- `delta_essdai_max_from_baseline = max(E(t) - B)` over follow-up only.
- `delta_essdai_min_from_baseline = min(E(t) - B)` over follow-up only.
- For each prespecified k in `ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS = (3, 5)`,
  last worsening means L−B ≥ k, improvement means L−B ≤ −k, and stability means
  −k < L−B < k. These three last categories partition the evaluable population.
- Ever worsening/improvement means at least one later observed E(t)−B ≥ k /
  E(t)−B ≤ −k. These are independent marginal proportions: the same patient
  can meet both and their percentages need not sum to 100%.
- `still_worsened_at_last_ge{k}` means ever worsening and last worsening.
  It does not establish continuous or sustained deterioration.
- `not_worsened_at_last_after_ever_ge{k}` means ever worsening and no longer
  meeting the threshold at last assessment. It does not require return to
  baseline or imply a recovery rate. Reconciliation partitions patients into
  never, ever-not-last and ever-and-last.
- No observed crossing leaves `first_observed_*_date` missing (`NaT`).

ESSDAI activity categories stay Low <5, Moderate 5–13 (5 ≤ score <14), High ≥14.
Stratification always uses the **baseline** state. Absolute change and category
transitions are independent: 6→9 meets +3 within Moderate, while 4→5 crosses
Low→Moderate with only +1. Neither is a Pop classification.

Both ±3 and ±5 are prespecified operational thresholds for descriptive and
sensitivity analysis, not newly validated clinical MCIDs in this cohort. The
legacy `ESSDAI_IMPROVEMENT_MCID=-3`, `ESSDAI_LEGACY_WORSENING_DELTA=5` and their
`CHANGE_THRESHOLDS` keys remain unchanged. Existing `ever_worsened` and
`ever_improved` in `patient_progression()` continue to mean category changes.
Step 14 still imports the same classifier, models and +5 sensitivity threshold;
its optional episode-level Step 13 Parquet is unchanged.

## Products

All paths are relative to `--output-root` (repository root by default).
No individual records or generated clinical outputs belong in Git.

| Directory | New files |
| --- | --- |
| `data/analytic/blockA/13_disease_activity_progression/` | `13_essdai_patient_delta_endpoints.parquet` |
| `outputs/tables/blockA/13_disease_activity_progression/` | `13_essdai_delta_threshold_summary.csv`, `13_essdai_worsening_by_baseline_state.csv`, `13_essdai_ever_vs_last_reconciliation.csv`, `13_essdai_baseline_last_state_by_delta.csv` |
| `outputs/qc/blockA/13_disease_activity_progression/` | `13_essdai_delta_eligibility_qc.csv`, `13_essdai_delta_invariants_qc.csv` |
| `outputs/figures/blockA/13_disease_activity_progression/` | `13_essdai_delta_by_baseline_state.pdf`, `13_essdai_ever_vs_last_worsening.pdf`, `13_essdai_individual_trajectories_delta3.pdf`, `13_essdai_individual_trajectories_delta5.pdf` |

The patient Parquet records official/last episode IDs, dates, scores and states,
follow-up years, assessment and distinct-date counts, delta extrema, last
statuses and stable flags, ever/last worsening and improvement, reconciliation
flags and first documented worsening/improvement dates for both thresholds.

The tidy summary includes Overall and all three baseline states, with explicit
patient event counts and denominators. Baseline-to-last has three categories;
ever has two independent directions. Percentages are 100×events/evaluable and
Wilson 95% bounds use the percentage scale. Empty strata have explicit n=0 and
missing percentages/bounds. The compact table adds median follow-up and visits.
Reconciliation uses the same strata and denominators. The patient state/delta
crosstab uses `last_status` together with `threshold_abs` rather than separate
`last_status_pm3/pm5` columns; it is tidy and counts each patient once per
threshold, unlike the unchanged interval transition matrix.

Summary figures draw directly from the published summary table. The stacked
panels use identical scales and annotate n and non-small segments; all exact
counts and percentages are in the accompanying CSV. Empty groups are marked
not estimable. The ever/last figure uses identical patient denominators.
Trajectory figures include all evaluable patients and observed points, with
actual elapsed years, fixed baseline-state panels and activity cutoffs 5/14.
Color represents final absolute change and does not imply a monotonic path.
No patient labels are printed. The marginal `13_essdai_trajectory.pdf` remains.

## QC and execution

Pure endpoint/summary functions accept small fixtures; production calls follow
`validate_inputs()`. The invariants report pass/fail, failure counts and reasons,
including unique IDs, canonical endpoints/dates, legacy patient agreement,
partitions, nested thresholds, last-implies-ever, reconciliation, patient
crosstab totals, percentage bounds and table agreement. Integrity failures save
QC then raise before publishing the new tables or figures. Empty strata are
not estimable and do not interrupt other strata.

The existing overwrite policy is preserved: default / `--overwrite` regenerates
products, `--no-overwrite` protects all existing analysis outputs, and
`--dry-run` validates the contract without publishing patient endpoints, final
tables or figures (existing contract QC/log behavior remains).

```bash
python -m pytest -q tests/test_disease_activity_progression.py
python -m pytest -q tests/test_risk_factors_progression.py
python -m pytest -q
python src/block_A/13_disease_activity_progression.py --dry-run
```

Integration fixtures are synthetic, isolated under pytest temporary paths, and
verify both old/new products, nonempty PDFs, QC and overwrite/dry-run behavior.
They are not cohort estimates. At implementation time the canonical Step 10
and Step 11 Parquets are unavailable in this environment: **NOT RUN: input
unavailable** for the canonical dry run and real-cohort QC. No real clinical
results, patient counts or cohort estimates are claimed from these fixtures.

## Limitations

Visits are intermittent and irregular, and patients have unequal observation
length and numbers of assessments. Ever proportions may increase with follow-up
length or more visits; they are not incidence at a uniform horizon. First
observed threshold dates are not biological onset dates. The extension does
not add delta-specific Kaplan–Meier analyses or person-time rates. Last values
also refer to different times across patients.

Measurement variability, regression to the mean, and baseline floor/ceiling
effects can affect change, especially at very low or high baseline ESSDAI.
These observational descriptions support no causal treatment or biomarker
attribution. Report both thresholds; do not select one retrospectively based
on significance or observed proportions. Real results require execution and
reconciled QC on the authorized canonical cohort.

## Initial implementation review and validation record

Modified files and purpose:

- `config.py`: explicit ±3/±5 configuration, preserving all legacy keys/values.
- `src/block_A/13_disease_activity_progression.py`: pure patient endpoints,
  summaries, reconciliation, patient state/delta crosstab, aggregate QC and PDF
  rendering, integrated through the existing canonical output paths.
- `tests/test_disease_activity_progression.py`: exact boundaries, manual examples,
  missingness, dates/ties, empty strata, denominators, QC failure injection,
  all old/new output names and dry-run/overwrite behavior.
- `tests/test_risk_factors_progression.py`: explicit regression that +3/+4 do not
  trigger the downstream legacy +5 sensitivity and +5 still does.
- `README.md` and this guide: entrypoint, definitions, paths and limitations.

An AST comparison confirms every pre-existing Step 13 function except `main`
is unchanged; `main` only gains the new output writer call. No Step 10/11,
Step 14 production code, Pop definitions or longitudinal models were changed.
The four new vector PDFs were rasterized and visually inspected with synthetic
Low/Moderate/High examples; captions, axes, counts and legends are readable.
The layout was checked again after repairing label overlap. Fixtures and
rendered outputs remain in ignored temporary/scratch locations.

The full `python -m pytest -q` command cannot proceed past collection because
two pre-existing tests import files absent from the audited Git tree:

- `tests/test_btris_unmapped_lab_audit.py` expects
  `src/20_btris_visit_date_match_report.py`.
- `tests/test_pharma.py` expects `src/studies/pharma/01_run_pharma.py`.

Both paths are absent in the starting commit as well. These unrelated tests
and production modules were left unchanged. The canonical `--dry-run` was
attempted and cannot load the missing Step 11 input (Step 10 is also absent),
so it is recorded as **NOT RUN: input unavailable**, not a successful cohort run.

Final Step 13 verification: `python -m pytest -q
tests/test_disease_activity_progression.py` **PASS: 60 tests**. The synthetic
integration cohort has 24 evaluable patients, one row per patient, all 24
absolute-change invariant checks passing and all 12 expected PDFs generated
(eight existing, four new). A separate four-patient manual-example fixture
reconciles exact counts across baseline Low/Moderate/High, including overlapping
ever directions. These n values describe tests only, never the real cohort.
Dependency deprecation warnings remain; no clinical outputs are committed.

The Step 14 test
`test_weighted_other_domain_adjustments_aliases_missing_and_inconsistent_panels`
fails on its unchanged assertion that the raw articular contribution is 8,
while the canonical ordinal score 2 × configured weight 2 yields 4. The same
failure was reproduced in an isolated `git archive` of the starting commit
`91f9f120f9309b4173d2767b42cb391818ab88b5`, proving it predates this extension.
Clinical weights and the unrelated assertion were not changed. The new explicit
legacy +5 regression passes independently (1 passed, 72 deselected).

Final combined regression command:
`python -m pytest -q tests/test_disease_activity_progression.py tests/test_risk_factors_progression.py`
finished with **131 passed, 2 failed**: Step 13 has 60 passing tests, and Step 14
has 71 passing tests and the two pre-existing articular-weight assertions.
The second failure,
`test_to_negative_subtraction_and_discordance_fail_closed_without_clipping`,
also expects 8 instead of 4 and was independently reproduced on the same clean
starting-commit archive. No new regression was detected. The full-suite command
has the two collection blockers listed above; real-cohort validation remains
unavailable. `git diff --check` passes. Only source, tests and documentation are
included in the implementation commit on `codex/step13-essdai-delta3-delta5`.

## Follow-up opportunity and trajectory refinements (2026-10-19)

These additions preserve all primary ±3/±5 endpoints, their patient denominators,
legacy categorical analyses, Pop definitions, and Step 14 imports. The original
24 absolute-change invariants run unchanged alongside the new aggregate QC.
No clinical count from a reviewed run is hardcoded or used as a test fixture.

The existing `13_essdai_individual_trajectories_delta3.pdf` and `...delta5.pdf`
continue to color by **final absolute change**. The two additional files
`13_essdai_individual_trajectories_ever_worsening_delta3.pdf` and `...delta5.pdf`
color by existing worsening flags only:

- `never_observed`: no observed increase ≥k from official baseline.
- `ever_not_last`: observed ≥k before, but no longer ≥k at the last observed visit.
- `ever_and_last`: observed ≥k at some follow-up and at the last assessment.

Each figure retains every valid canonical episode, including date ties, in
patient/date/episode order. Times are actual years from official baseline;
activity cutoffs 5 and 14 are score references, not delta thresholds. All panels
share the ESSDAI scale, report n, median follow-up and category counts, and retain
a complete legend even when categories are absent. Background trajectories are
drawn first so minority event trajectories remain visible. No IDs or exact dates
are printed. Connecting lines join intermittent observations; they do not imply
monotonic change, continuous persistence, or recovery. Rare trajectories can
still be identifiable: any external disclosure requires institutional review.

The existing ever/last bar plot now shares a dynamic scale across both thresholds,
capped at 100%, with percentage and event/denominator labels. The final-change
stacked plot adds an exact-count/percentage table so small segments remain
readable without placing cramped labels inside them.

### New aggregate products

Paths remain relative to `--output-root`, under the canonical Step 13 directory.

| Location | File | Interpretation |
| --- | --- | --- |
| `outputs/tables/blockA/13_disease_activity_progression/` | `13_essdai_observation_opportunity_sensitivity.csv` | Overall and Low/Moderate/High, separately for each threshold and observation band; ever/last counts, percentages, Wilson 95% bounds, ever-not-last counts and median follow-up/assessment/date counts |
| Same tables directory | `13_essdai_baseline_change_opportunity.csv` | Original all-evaluable rates alongside counts and conditional percentages among patients mathematically able to improve/worsen by k |
| `outputs/qc/blockA/13_disease_activity_progression/` | `13_essdai_refinement_qc.csv` | Aggregate check/status/failure count/reason; no identifiers, dates or individual row values |
| `outputs/figures/blockA/13_disease_activity_progression/` | `13_essdai_individual_trajectories_ever_worsening_delta3.pdf`, `...delta5.pdf` | The complementary trajectory categories defined above |

Duration bands default to `(0, 2)`, `[2, 5)`, `[5, ∞)` years and are configured
through `ESSDAI_FOLLOWUP_DURATION_BANDS_YEARS`. Their CSV labels are `<2y`,
`2-<5y`, `>=5y`. Assessment-count bands are 1, 2 and ≥3. Counts refer to valid
canonical episodes; distinct-date medians are reported separately, because
multiple assessments can occur on one date. Each stratifier partitions the
entire evaluable population; thresholds have identical band denominators.
Empty cells retain n=0 and NaN percentages/intervals. `support_note` is
`not_estimable` for empty cells, `sparse` for n<10, and `descriptive_only`
otherwise. This is a descriptive support annotation, not an institutional
small-cell disclosure rule or permission to release aggregates. High remains
visible; no inferential comparisons are calculated.

Mathematical improvement requires baseline ≥k. Scores 0–2 cannot improve by 3;
all Low scores (<5) cannot improve by 5. Mathematical worsening requires
baseline+k ≤ `ESSDAI_MAX_VALID=123`, the unchanged repository validation ceiling.
High by itself does **not** imply a ceiling. Counts and rates over all evaluable
patients remain primary. Conditional rates use only the corresponding possible
count and are NaN if that count is zero. All actual events must be possible;
the conditional event count must therefore equal its original event count.

The new QC reconciles plot categories with the existing reconciliation table,
checks mathematical possibility for ever/last flags, validates positive visit
and date counts, partitions and event totals across bands and baseline states,
compares Δ5/Δ3 nesting and denominators, and rebuilds both aggregate summaries.
Failures write aggregate QC then stop the new output writer before publishing
its endpoint Parquet, tables or figures. The existing `--dry-run` early return
and `--no-overwrite` guard also protect all new products.

### Interpretation that accompanies these products

Observed follow-up duration and assessment counts are **post-baseline** strata.
The percentages are observed proportions conditional on that follow-up history,
not standardized cumulative incidences, risks at 2 or 5 years, or annual rates.
Longer or more frequent observation increases opportunity to detect change;
it may also reflect severity, clinical selection and treatment. Early worsening
followed by loss to follow-up can enter a short-duration band. No observed
event does not establish absence of progression. Comparisons across activity
states or observation bands support no causal conclusions.

Ever-not-last does not imply remission or a clinically acceptable last value;
ever-and-last does not estimate **persistent worsening** between assessments.
Visits are irregular and can miss events. Fixed-horizon risk would require a
separate censoring/first-event specification accounting for irregular visits
and possible interval censoring; no new simplified Kaplan–Meier analysis is
introduced here. Floor effects, regression to the mean, treatment and selection
can all influence observed improvements. A small High group cannot establish
spontaneous recovery or treatment efficacy.

### Methods text

We summarized observed changes in total ESSDAI relative to the official clinical
baseline using absolute thresholds of three and five points. In addition to
changes between baseline and the last observed assessment, we identified whether
worsening had been observed at any subsequent assessment and whether the
threshold was still met at the last assessment. Because patients differed in
follow-up duration and assessment frequency, we also provided descriptive
summaries stratified by observed follow-up duration and number of ESSDAI
assessments. These comparisons reflect opportunities for observing change and
are not standardized cumulative incidence estimates. We further documented
whether a change of a given magnitude was mathematically possible from each
patient's baseline ESSDAI score, while preserving the original evaluable-patient
denominators for primary summaries.

### Refinement validation record

On current `main` (`efeb23e`, before this refinement), the combined Step 13/14
regression command produced **131 passed, 2 failed**. After this change it
produces **152 passed, 2 failed**: Step 13 has **81 passing tests**, and Step 14
has **71 passing tests** plus the same two existing failures:

- `test_weighted_other_domain_adjustments_aliases_missing_and_inconsistent_panels`
- `test_to_negative_subtraction_and_discordance_fail_closed_without_clipping`

Both still expect articular contribution 8 where the configured score/weight
produces 4. Their failures were reproduced before editing; Step 14 source and
tests are unchanged. They are not new regressions. Dependency warnings remain.

New synthetic tests cover the three trajectory categories and drawing order,
canonical date ties and distinct-date counts, duration boundaries
1.999/2/4.999/5, assessment counts 1/2/3/4, configurable duration bands,
Wilson intervals, threshold nesting and totals, improvement floor and worsening
ceiling, High worsening, empty schemas/NaN percentages, aggregate CSV privacy,
QC failure injection, adaptive axes including 0%/100%, and protection of new
outputs under dry-run/no-overwrite. A separate synthetic preview passed all
**61 refinement checks**. All **24 original invariants** still pass.

An AST comparison with current `main` confirms all existing endpoint, summary,
reconciliation and model functions are unchanged. Existing function edits are
limited to naming the same validation ceiling in configuration, the two plot
presentation refinements, and wiring the new outputs into the existing writer.
Direct comparison against the prior implementation on four synthetic fixtures
confirms identical endpoint tables and all four pre-existing aggregate tables.
The four revised/new PDFs were rasterized and visually inspected, including
dense Low trajectories, all categories, counts, legends, axes and small cells.

The canonical `--dry-run` was attempted and stops at missing Step 11 baseline
input; Step 10 input is also unavailable. Thus real-cohort execution, the
provided 143-patient counts, and real-data QC remain **NOT RUN: input
unavailable**. No cohort results are inferred from synthetic tests, and no
patient data were requested, exported or committed. Generated outputs, previews
and logs remain ignored; `.gitignore` now also explicitly excludes `data/raw/`.
