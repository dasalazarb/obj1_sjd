# Step 17 V3 software validation

Implementation base: `87785f3dc94b6192d329a69450b5bcf16c67158f` (`main`).
Validation date: 2026-11-03. All runs below use synthetic inputs. No clinical
execution, institutional validation, or recovery count on patients is asserted.

V3 repairs A prediction when `FrozenBasis` is absent and runs both four- and
three-knot specifications for every applicable outcome. Three knots are a
post-hoc exploratory alternative. The four-knot reference, fixed retrospective
groups, sample rules, C support policy, fit QC and coherent fallback remain.

## Executed regression

```bash
MPLCONFIGDIR=/tmp/step17-mpl XDG_CACHE_HOME=/tmp/step17-cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/test_temporal_spline*.py \
  tests/test_secondary_comparison*.py \
  tests/test_secondary_trajectories_by_essdai_pattern.py \
  tests/test_secondary_trajectory_performance.py
```

Result: **176 passed**. The suite emitted existing dependency deprecation,
empty-slice and pandas concatenation warnings; no test failed. Final V3-focused
checks were also repeated after refining the failure-summary categorization:
**21 passed** (`test_temporal_spline_v3.py` and `test_temporal_spline_end_to_end.py`).

Environment: Python 3.12.14, NumPy 2.5.3, pandas 2.3.3, SciPy 1.18.1,
statsmodels 0.15.0. The committed V2 numerical reference uses deterministic
Gaussian GEE to avoid optimizer/platform convergence differences. Its source is
exactly the four-knot design, feasibility, fitting and characterization modules
at the implementation base above; it was generated before comparing V3.
`tests/fixtures/temporal_spline_v2_reference.json` contains synthetic aggregate
expectations, not patient data. Fits/covariances, tests and selected supported
levels/changes/velocities/pairwise simultaneous bands agree at `rtol=atol=1e-9`;
frozen transforms agree at `1e-12`. A/B predictions also have an independent
matrix comparison at `1e-12`. C3/C4 Wald tests use explicit independent index
lists and chi-square calculations.

The intended four-knot V2→V3 numerical difference is that previously omitted
supported A curves now exist when nonlinear basis construction fails. Output
version/namespace, metadata, final-family failure summaries and QC inventory are
also revised. No silent change to the four-knot estimands was detected.

## Synthetic bundle evidence

Each row below inventories one synthetic outcome at threshold ±3. These are
fixtures testing code, not estimates of institutional recovery or C success.

| Fixture / basis | A/B/C eligible | A/B/C fitted | Outcomes with A curves | Outcomes with C curves | Supported times | Curve rows |
|---|---|---|---|---|---|---|
| Dense different-curvature / four knots | 1/1/1 | 1/1/1 | 1 | 1 | 13 | 156 (A/B/C) |
| Dense different-curvature / three knots | 1/1/1 | 1/1/1 | 1 | 1 | 13 | 156 (A/B/C) |
| Repeated boundary knots / four knots | 1/0/0 | 1/0/0 | 1 | 0 | 6 | 24 (A only) |
| Repeated boundary knots / three knots | 1/0/0 | 1/0/0 | 1 | 0 | 6 | 24 (A only) |

The repeated-boundary fixture has 14 patients per group, all observed at year
zero; eight per group also have year-three records and one has a year-1.5 record.
Both nonlinear bases have nonunique knots while A remains estimable. The old
prediction guard would export zero curves; V3 exports 24 A rows, at the same six
shared supported times, with no C bands or C p/q values. Two independent output
roots produce identical table artifact hashes and unchanged input/code hashes.
A historical V2 manifest sentinel survives both publication and explicitly
backed-up V3 overwrite. Dense C bundles export all six pairs and three estimands.

Injected C3 covariance/convergence failures retain coherent GEE A/B and mark C
`FIT_FAILED`; C p/q and simultaneous outputs are absent. A separate test forces
MixedLM C3 failure after A/B attempts and verifies complete A/B/C GEE refitting.
Failures are not repaired. The final family determines the summarized failure
cause; covariance failures take priority over explicit nonconvergence, and
warning-only rejections remain other QC failures. Earlier attempts remain auditable in `01_fit_attempts.csv`.

## Acceptance coverage

| ID | Executed evidence |
|---|---|
| T01 | `test_A_without_basis_supported_grid_and_export`; genuine degenerate-basis end-to-end bundle exports 4 × 6 A rows |
| T02 | Same unit test: no shared grid → no curves, explicit `no_four_group_supported_grid` |
| T03 | `test_A_B_predictions_equal_original_basis_formula`; frozen V2 numerical reference |
| T04 | `test_A_known_functions_and_gradients_without_basis[False]`: independently known levels/slopes and finite-difference coefficient gradients |
| T05 | Binary variant: probabilities/logit intervals in [0,1], analytic probability velocity and gradients |
| T06 | Existing design tests and V2 reference: 8/10/16, frozen serialization, natural boundaries |
| T07 | Parameterized V3 design test: 8/9/12 with full rank and pooled median |
| T08 | Literal A→B→C prefix checks for both bases |
| T09 | Three-knot analytic derivatives versus interior finite differences; projection orthogonality and natural boundaries |
| T10 | Independent C3 indices/statistics, df 1/3/6 |
| T11 | Independent C4 indices/statistics, df 2/6/9, V2 numerical reference |
| T12 | `test_C3_failure_retains_coherent_AB_without_C_inference[covariance]`: no misleading p/q or bands |
| T13 | C3 full-family fallback test and existing C4 fallback regression |
| T14 | Cross-basis FDR/classification test, homonymous-outcome regressions, unique per-basis manifests |
| T15 | Patient-count support tests, unsupported-time omission, disconnected interval and plot segments |
| T16 | Known BH values and combined-base input: denominators separate by test/threshold/clock/scenario/basis; joint cross-basis BH intentionally not implemented |
| T17 | End-to-end privacy scan of every CSV; existing small-cell, percentage and reconstruction suppression tests |
| T18 | Repeated synthetic A-only publication has identical table hashes/input/code fingerprints; fixed-seed sup-t reproducibility test |
| T19 | Frozen V2 numerical reference and untouched V2 historical namespace sentinel |
| T20 | No-argument defaults test plus full synthetic execution producing both bases; no implicit dry run |
| T21 | Existing-destination refusal, missing-backup rejection, explicit V3 backup publication |
| T22 | A-only atlas test inspects all four A labels and eight disconnected line segments; A-only/C3 figures visually inspected |

All CSV schemas carry version, basis, knot count, nonlinear df, exploratory role,
support policy, sample fingerprint and unit (also on empty tables). Each bundle
has code/input/artifact hashes, commit and dirty state, model counts, C failure
causes, supported coverage, and FDR scope. Pointwise A intervals remain `ci95_*`.
Publication keeps existing atomic staging, privacy scanning, backup and rollback.
No changes were made to Steps 10/11/13/15/16 or upstream artifacts.

## Institutional work still pending

The reported 23 A-only outcomes at ±3 and 7 at ±5 cannot be certified by synthetic
runs. After review/integration and an institutional pull, run the default command
inside Biowulf with the identical authorized freeze and a new V3 output root.
Verify per-outcome shared support, QC PASS, and all 30 recovered A outcomes there.
Do not export individual records or overwrite V1/V2. Compare all C3/C4 outcomes
and failure causes, regardless of p/q, with separate exploratory BH denominators.
Do not select minimum p/q or claim joint confirmed discoveries across bases.
