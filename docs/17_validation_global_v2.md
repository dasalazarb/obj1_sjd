# Step 17 secondary validation V2

`validation_global_v2` consumes the four authenticated `temporal-spline-v3`
`main` references. The V3 analysis, FDR families, q-values, knots, eligibility,
fixed groups and support policy are unchanged. Existing V1 outputs and `work/`
checkpoints remain historical evidence. The current validation entrypoint writes
only V2; reproducing V1 requires its original code revision and environment.

## Bootstrap contract

Every numeric task uses the same stratified whole-patient draws and one fixed
family refit per draw for two policies:

- `strict_original_support`: requires every original locally supported grid
  point, as in V1.
- `fixed_original_grid_conditional`: evaluates the complete original estimand
  family after a valid refit even when local replicate support is lost. This is
  exploratory conditional sensitivity, not new empirical support. It cannot
  promote a result to `SUPPORTED_EXPLORATORY` when strict calibration fails.

Both require `max(100, ceil(0.8 * requested))` successful draws. No fallback
family, reduced grid, selected contrasts, filled gaps or reduced success
threshold is allowed. A retains four slopes and six slope contrasts; B retains
shared curvature and supported curves; C retains six pairs times the original
grid times level/change/velocity, with the original temporal anchor. Frozen
structural zero gradients remain defined anchors rather than studentized terms.

The maximum absolute deviation uses **original standard errors**. It is not a
replicate-studentized bootstrap-t/percentile-t procedure. Its critical value
compares simultaneous families; it does not validate external confidence coverage.

`07_patient_bootstrap_qc.csv` and the new
`07_bootstrap_failure_diagnostics.csv` have one row per task and policy. Each
policy separately reconciles `n_attempted = n_accepted + n_failed`, with failures
partitioned into refit, support, estimand and other/draw failures. Never sum the
two policy denominators: they describe identical draws. Stage counters and
support-loss histograms are secondary overlapping diagnostics, not additional
rejection causes. The strict primary cause follows V1's order: refit, then
support, then target. Explicit rejection of nonfinite targets prevents V1's
undefined maxima from being counted as calibrated; the regression oracle uses
finite comparable fits. Each policy's diagnosis and summary denominator are
reported separately. QC categories use an allowlist and deterministic priority;
raw exception text is never exported.

`SUPPORT_FILTER_DOMINANT` means more than half the strict rejections were valid
fits rejected by support. `REFIT_QC_DOMINANT` means more than half were failed
fixed-family/QC attempts. Remaining rejection patterns are `MIXED_FAILURES`.
There is no claim about which cause dominates the real cohort before execution.

A private replay of each draw's pre-selection RNG state observes original source
patients before identity is discarded. It does not change the production
resampler, its RNG sequence or patient pseudonyms. Only averages of source-unique
and pseudo-entry window fractions, per-group/grid coverage, draw counts and distance
outside the observed range survive, aggregated over conditionally accepted draws.
Original identities, selections, coefficients and individual predictions never
enter artifacts, logs or checkpoints. Temporal region support loss is aggregated
across comparable fits. These metrics describe support loss, not new support.

## Classification and disclosure

The classification inventory retains all registered outcomes times A/B/C,
including pending, inapplicable and unestimable models. It reports independent
`loo_stability_status`, `bootstrap_calibration_status` and
`scientific_validation_status`, three explicit validity/influence flags, reason
codes and coverage. Material LOO influence has precedence over an inconclusive
or unavailable bootstrap. Failure of every bootstrap draw alone does not imply
material influence. All-failed or uninterpretable LOO stays inconclusive.

A successful strict bootstrap plus LOO within the frozen material, longitudinal,
support and band limits can produce only exploratory support. Binary tasks retain
LOO but spend no bootstrap attempts and report `NOT_ESTIMABLE_METHOD`.

Conservative primary/complementary suppression covers numeric totals, fractions,
nested failure partitions, validity flags and diagnostic classifications. A small
cell (1–4) suppresses its linked outcome's exported diagnostics; totals cannot
reconstruct it by subtraction. Suppressed influence/calibration labels become
`DISCLOSURE_SUPPRESSED` and the scientific label remains conservatively
`INCONCLUSIVE`. No unsuppressed task status is cached beside a suppressed view.
Policy-level failure summaries omit counts if any contributing row is suppressed.
No domain aggregation is used as a disclosure workaround. Review unavailable
causes locally in the authorized protected Biowulf environment. Obtain the
institution's disclosure authorization before exporting real-data bundles.

## Commands

Run from the repository root with the analysis environment active:

```bash
python -m pytest -q tests/test_temporal_spline_validation.py tests/test_temporal_spline_validation_v2.py
python src/block_A/17_validate_temporal_heterogeneity.py --help
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth systematic --reference-root /authorized/obj1_sjd \
  --output-root /authorized/step17_validation_v2
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth deep --basis three_knots --threshold 3 \
  --bootstrap-replicates 200 --max-outcomes 5 --max-patient-refits 2000 \
  --reference-root /authorized/obj1_sjd \
  --output-root /authorized/step17_validation_v2_pilot
```

The frozen stratified queue uses all eligible models across domains and types,
without p/q filtering. Optional `--outcomes-file` supplies preregistered clinical
priorities, not a filtered inventory. Record pilot strata and clinical priorities
before reviewing validation results. A budgeted pilot validates a subset, not the
whole cohort. Both policies run automatically; no second bootstrap flag is needed.

Resume only unchanged V2 code, data, references, seed, policies and dependencies:

```bash
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth deep --basis three_knots --threshold 3 \
  --bootstrap-replicates 200 --max-outcomes 5 --max-patient-refits 2000 \
  --reference-root /authorized/obj1_sjd \
  --output-root /authorized/step17_validation_v2_pilot --resume \
  --backup-root /authorized/step17_validation_v2_pilot/outputs/backups/17_nonlinear_temporal_heterogeneity/validation_global_v2/next
```

Budgets apply to **each invocation**, counting new tasks only; cached completed
tasks consume no new budget. Reservation conservatively allows up to six
reference model attempts; observed accounting separately records actual reference
attempts (including failed gates/QC), LOO exclusion attempts, bootstrap draw
attempts and bootstrap refit attempts. Failures count. Observed counters and the reserved cost may be
suppressed to avoid reconstructing hidden task counts. A model task is the atomic
interruption boundary. Publication still uses staging, artifact hashes,
transactional rollback and an authorized backup inside the output root.

The manifest records both policy definitions, the original support policy,
calibration denominator, source/reference/code hashes, outcome sample
fingerprints, cutoff, seed, Git identity/worktree state and dependency versions.
Versioned checkpoint contracts reject V1 and changed V2 policies/context.
`software_qc_status=PASS` certifies integrity, not scientific calibration.

## Verification and remaining execution

Synthetic acceptance tests cover the strict V1 regression oracle, all A/B/C
target families, shared draws and whole trajectories, source duplicates, exclusive
failure partitions, nonfinite targets, fixed-family/covariance failures, threshold
ceilings, LOO precedence, privacy, all four V3/V1 bundle hashes, inventories,
q-values, resume, budgets and publication rollback. The preregistered synthetic pilot also exercises real fixed-family GEE fits
for linear A, common-curvature B and group-curvature C, each with 200 bootstrap
draws and complete LOO. Synthetic results do not
establish the real cohort's failure distribution.

The clinical references, diagnostic ZIP and NIH source data are not supplied in
the development environment. Authenticating the four real references, running a
preregistered stratified clinical pilot and reviewing its protected diagnostics
remain work for the authorized Biowulf run. Do not authorize a larger real-data
run solely from synthetic software tests.
