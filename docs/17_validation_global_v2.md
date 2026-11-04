# Step 17 validation_global_v2

V2 adds diagnostics and a paired exploratory sensitivity to the secondary validation
of `temporal-spline-v3`. It never rewrites the V3 analysis or V1 snapshots.
No primary analysis modules, model thresholds, knots, FDR families or q values change.

Both bootstrap policies use identical stratified whole-patient draws and exactly one
fixed-family refit per draw. `strict_original_support` retains V1's complete original
support rule and `max(100, ceil(0.8*n_requested))` requirement. The
`fixed_original_grid_conditional` sensitivity evaluates the same original estimands
when a valid fit loses local support. It establishes no new observational support.
Conditional calibration alone cannot confer `SUPPORTED_EXPLORATORY`.

Reported maxima use **original** standard errors, not replicate standard errors;
they are not bootstrap-t/percentile-t and do not establish external confidence coverage.
Nonfinite target values and changed families/covariance fail both policies. Strict
failure attribution is exclusive: refit, then support, then estimand. Secondary
support diagnostics are separate and must not be added to that partition.
Source-unique and pseudo local fractions distinguish repeated draws from independent
source trajectories. Source identities exist only in memory and are never attached
to the returned draw, checkpoint or published artifact.

`07_bootstrap_failure_diagnostics.csv` and the JSON summary contain disclosure-controlled
aggregate diagnostics. Linked flags, reason histograms and scientific classifications
follow complementary suppression. Resource totals are also withheld when they could
reconstruct suppressed patient denominators. Suppressed scientific status is conservatively
`INCONCLUSIVE`; no coarser disclosure bypass is provided. A suppressed breakdown cannot
justify a claim that support or fit failures dominate. Authorized protected-site review
remains necessary if those diagnostics cannot be released.

LOO and bootstrap have independent status columns. Material LOO influence takes
precedence over inconclusive/unavailable bootstrap. No successful LOO deletions alone
means inconclusive, even if every deletion failed. Binary bootstrap attempts are zero;
binary LOO is retained. The global inventory still includes every outcome and A/B/C.

## Commands

```bash
python -m pytest -q tests/test_temporal_spline_validation.py tests/test_temporal_spline_validation_v2.py
python src/block_A/17_validate_temporal_heterogeneity.py --help
python src/block_A/17_validate_temporal_heterogeneity.py --depth systematic --reference-root /protected/obj1_sjd --output-root /protected/step17_validation_v2
python src/block_A/17_validate_temporal_heterogeneity.py --depth deep --basis three_knots --threshold 3 --bootstrap-replicates 200 --max-outcomes 5 --max-patient-refits 2000 --reference-root /protected/obj1_sjd --output-root /protected/step17_validation_v2_pilot
```

Both policies always run together; no extra policy flag is needed. All four original
V3 references are authenticated even for a single-scenario pilot. Pilot prioritization
can be preregistered with `--outcomes-file` (JSON outcome IDs); the default queue is
global and does not select by p/q. A clinically representative stratification must be
specified before examining validation results.

V2 publishes under `temporal-spline-v3/validation_global_v2` and uses a separate work
namespace. Resume only unchanged V2 code/data/seed/policies/dependencies with `--resume`
and, for a persistent backup, `--backup-root` inside the chosen output root's
`outputs/backups/17_nonlinear_temporal_heterogeneity/` tree. Never point V2 at V1 work.
The existing automatic transactional replacement with temporary backups remains
available; explicit persistent backups are recommended for protected-site audits.
Budgets apply per invocation to newly executed whole model tasks; cached tasks do not
consume those budgets. Reservations conservatively include up to six reconstruction
attempts; summaries separately report actual reconstruction, LOO and bootstrap attempts.

## Remaining protected-site validation

Real NIH data and diagnostic ZIPs are not required for synthetic software checks and
are not included in commits. Run the complete systematic inventory and a preregistered
stratified deep pilot at the authorized site before broader deep validation. Confirm
all four authentic V3 and V1 snapshots remain byte-identical. Review disclosure before
exporting any clinical result; never export work/checkpoints or source data. Only then
assess whether failures are predominantly support-related, fit/QC-related or mixed.

## Software verification

Synthetic validation coverage: 75 distinct tests (27 initial contracts, 47 remaining
contracts, and one additional small-source-cell privacy contract). Five affected
privacy/determinism tests also passed after the last privacy change. The six other
Step 17 spline suites passed all 66 tests. CLI help, compilation and diff whitespace
checks passed. Strict V2 agrees numerically with the frozen V1 bootstrap for finite,
comparable synthetic fits; rejecting nonfinite targets is an intentional safety fix.

Source-support rates and distances are withheld if any contributing source-patient
window has a cell of 1–4, even when pseudo-patient counts look adequate. Replicate
counts and strict/conditional calibration are unchanged by this diagnostic suppression.

Full repository collection remains blocked by unrelated missing dependencies and
absent legacy scripts (`src/20_btris_visit_date_match_report.py` and
`src/studies/pharma/01_run_pharma.py`). No real clinical pilot was run or exported.
