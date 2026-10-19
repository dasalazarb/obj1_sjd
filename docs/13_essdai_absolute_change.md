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

## Implementation review and validation record

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
