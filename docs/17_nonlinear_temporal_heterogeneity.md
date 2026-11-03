# Step 17: nonlinear trajectories and temporal heterogeneity

Step 17 consumes the authenticated Step 10/11/13 contracts through the existing
Step 16 adapter. It never runs or changes Steps 10/11/13/15/16. Fixed retrospective
ESSDAI memberships at ±3 and ±5 are fitted separately. These are associations
conditional on the observed history, not prediction, causation, onset or latent
patient states. Nonestimability is a result, not evidence of equal trajectories.

## V3: A curve repair and exploratory three-knot alternative

A previously fitted linear model could lose every curve when spline freezing
failed. V3 uses `secondary_comparison_models.design_row` for A levels and the
exact difference between its rows at years 1 and 0 for its derivative. Pointwise
Wald limits remain `ci95_*`; binary limits transform a logit interval with expit.
Only points marked `four_group_supported=True` are published. If none exist,
`no_four_group_supported_grid` is recorded; no replacement grid is invented.
Changes, velocities and simultaneous bands specific to C still require valid C.
An A-only outcome receives an atlas with the existing supported segments.

The three-knot alternative was proposed after observing failures of four-knot C.
It is **exploratory post hoc**, even when q < .05, and cannot replace the reference
based on significance. Both bases use exactly the same longitudinal rows, groups,
three-year main horizon and local support; the runner checks sample digests and
support equality before fitting. Different rank/fit eligibility is allowed.

| Specification | Frozen normalized knots | A / B / C parameters | B–A / C–B / C global df |
|---|---|---|---|
| `four_knots` | 0, pooled 1/3, pooled 2/3, 1 | 8 / 10 / 16 | 2 / 6 / 9 |
| `three_knots` | 0, pooled median, 1 | 8 / 9 / 12 | 1 / 3 / 6 |

Methods are `restricted_cubic_pooled_quantiles_1_3_2_3_projected` and
`restricted_cubic_pooled_median_projected`. The alternative reduces dimension;
it does not guarantee identifiability, convergence, valid covariance or evidence.
`C_RELAXED_SUPPORT_V1` and every model/covariance check remain fixed. There is
no pseudoinverse, covariance repair, ad hoc regularization or relaxed fitting.

BH families and figure panels are separate for each basis. Their tests have
different degrees of freedom and are not direct replications. This implementation
**does not implement joint cross-basis FDR**: do not select the minimum p/q or claim
confirmed discoveries from comparing bases. `family_n_tests` and the manifest
state the denominator and the separate exploratory family policy.

## Design and implementation

- `temporal_spline_config`: versioned, response-independent limits, priority list
  (including real catalog IDs `iga`, `igm`) and frozen RNG settings.
- `temporal_spline_contract`: reuse input authentication, dates, units and group
  identities; fingerprint every upstream artifact and both old and new code.
- `temporal_spline_feasibility.evaluate`: freeze one common sample per
  variable/threshold/clock/scenario; count independent patients, repeated dates,
  three-date patients with sufficient span and local representation.
- `temporal_spline_design.FrozenBasis`: pooled-time natural restricted cubic
  spline with four knots (boundaries, pooled 1/3 and 2/3 quantiles), giving time
  plus two nonlinear columns. Project nonlinear terms off intercept/time once.
  The serialized transform is reused in analytic predictions and derivatives.
  Four knots remain the reference (A/B/C: 8/10/16 columns). Three pooled knots
  (boundaries and median) provide the alternative (8/9/12 columns). A is a
  literal prefix of B and B of C in both designs; default `freeze(time)` stays
  four knots. Projection, scaling, prediction and derivatives use one frozen object.
- `temporal_spline_models.fit_family`: ML random-intercept A/B/C with QC; if any
  fails, discard the partial family and refit all eligible models with robust
  Gaussian GEE. Binary repeated outcomes use robust binomial GEE. No jumps over
  a failed A/B. A/B remain possible when C support fails. Degenerate spline knots
  leave A only within that specification; A curves use the exact eight-column
  linear design and time gradient independently of `FrozenBasis`. There is no
  nonlinear basis fallback inside a specification.
- `temporal_spline_contrasts` / `inference`: supported group means, common-reference
  changes, analytic velocities, and six response-scale pairwise differences.
  Reproducible coefficient simulations provide a joint 95% sup-t band across
  **six pairs × all supported grid points × all three estimands**. A separate
  joint family covers four group curves/changes/velocities. Binary bands use an
  explicitly asymptotic response-scale delta approximation, including the
  probability-velocity gradient; no logit coefficient is called probability/year.
- `temporal_spline_sensitivity`: independent scenario fits, patient bootstrap
  refinement, threshold/horizon/sign/composition comparisons.
- `temporal_spline_outputs`: isolated staging, aggregate export, overwrite/backup
  guards and the existing transactional install/rollback. Publication occurs only
  after final input/code verification; QC directories install last.

## One command: full analysis by default

Use the institutional Python environment containing numpy, pandas, scipy,
statsmodels, pyarrow, matplotlib and pytest. Running Step 17 **without flags**
runs the entire canonical analysis, saves tables, priority figures and QC
artifacts. `--dry-run` is optional and is not the default.

```bash
python src/block_A/17_nonlinear_temporal_heterogeneity.py
```

The no-argument invocation freezes **data cutoff at 2026-05-31**. This date
is a reproducible default selected for the current workflow, **not** inferred
from source data and **not** a claim that the approved institutional freeze
is 2026-05-31. If the canonical data freeze differs, explicitly override it:

```bash
python src/block_A/17_nonlinear_temporal_heterogeneity.py --data-cutoff YYYY-MM-DD
```

Default parameters:

- Input origin: `canonical`; output root: `common.PROJECT_ROOT`.
- `--threshold both`, `--time-axis clinical_baseline`, `--analysis all`.
- `--sensitivity standard`, `--max-primary-years 3` (via SplineConfig).
- Both bases, in frozen order: `four_knots`, then `three_knots`, for every applicable outcome.
  There is no failure-triggered or p-value-triggered basis selection.
- `--figures priority`, `--progress` (enabled), 250 DPI.
- `--dry-run` disabled; outputs are written and existing output paths
  remain overwrite-protected.

When successful, the final JSON contains `qc_status: PASS`, the bundle count,
the cutoff and output directory paths. CSV/PNG/manifest files live under
`<project>/outputs/{tables,figures,qc}/blockA/17_nonlinear_temporal_heterogeneity/temporal-spline-v3/`
with `<basis_specification>/<time_axis>/delta<3|5>/<scenario>/` suffixes. Previous published results
are never overwritten silently. `--overwrite` still requires a valid
`--backup-root`.

The `--analysis all` default fits eligible A/B/C models, tests temporal
heterogeneity, applies multiplicity control, makes priority figures and
publishes both bases at both thresholds. `--analysis temporal` also fits its own models;
`feasibility` and `models` get distinct partial-output namespaces.

Optional diagnostics / custom runs:

```bash
# Validate authenticated inputs, support, ranks and knots; no writes.
python src/block_A/17_nonlinear_temporal_heterogeneity.py --dry-run

# First real-data smoke in an isolated location; verify IDs in catalog.
python src/block_A/17_nonlinear_temporal_heterogeneity.py \
  --variables iga igg complement_c3 \
  --output-root /authorized/step17_smoke --sensitivity none --figures all

# Optional frozen-basis whole-patient bootstrap on a priority variable.
python src/block_A/17_nonlinear_temporal_heterogeneity.py \
  --variables iga --bootstrap-replicates 200 \
  --output-root /authorized/step17_bootstrap
```

Other optional controls: `--threshold 3|5|both`, `--time-axis
clinical_baseline|essdai_index`, `--sensitivity none|standard|full`,
`--config path.json`, `--seed`, `--simulations` (default 4000,
minimum 500), `--figures none|priority|all`, `--dpi` (minimum 200).
Configuration is a JSON object of `SplineConfig` fields. Example:

```json
{"version": "temporal-spline-v3", "max_primary_years": 3,
 "grid_step": 0.25, "c_support_policy": "C_RELAXED_SUPPORT_V1",
 "minimum_c_supported_regions": 2, "minimum_c_supported_annual_windows": 2,
 "minimum_c_supported_grid_points": 4, "minimum_c_supported_grid_span_years": 1.0,
 "minimum_prediction_patients": 5,
 "simulations": 4000, "seed": 1701}
```

The original analysis command has no checkpoint/resume. The separate global
validation command below implements aggregate checkpoints. In-memory patient rows are discarded
after each variable. Existing destinations are blocked unless both
`--overwrite` and `--backup-root` are provided. Backups must be under
`<output-root>/outputs/backups/17_nonlinear_temporal_heterogeneity/temporal-spline-v3/`
(and under `synthetic_validation/` for synthetic inputs).

## Support and estimands

All fits use the intersection of observed group ranges, with nonnegative time
and a primary maximum of three years. At least 10 patients and 8 repeated
patients per group, three distinct group times and a common span of at least
one year are required even for A. Policy **C_RELAXED_SUPPORT_V1** requires
four-group local support at **at least four distinct grid times spanning at least
one year**, covering **at least two of three reporting-horizon regions** and
**at least two annual windows**. Every published point still requires **five
distinct patients per group**. The number of patients with at least three dates
and a one-year span remains a diagnostic (`n_three_dates`, its components, and
`limited_three_date_histories`), and is no longer a C eligibility barrier.
These are operational limits, not power guarantees.

This uniform, versioned revision was selected after inspecting prior C
feasibility failures. The analysis role is therefore
`exploratory_posthoc_basis_specification` for both bases, thresholds and every outcome.
C results, including p/q and simultaneous bands, are exploratory. Significance,
BH-FDR within each base, the four-knot reference estimands, A/B requirements, rank, condition, convergence,
model-family and covariance QC remain unchanged. Failed C fits do not yield p/q.
Support thresholds for this named policy cannot be overridden individually in
configuration; another policy requires an explicit specification/version change.
The three-year primary horizon and retrospective group definitions are unchanged.
V3 destinations include a mandatory basis directory under `temporal-spline-v3/`; V1/V2 artifacts and manifests remain untouched. Use the historical V2 commit to reproduce a V2 run.

The fixed grid is 0, .25, .5, … within the common range. A point is supported
only if all four groups have at least five distinct patients in ±.5 years.
Unsupported points are omitted from estimates, retained in the support table,
and never bridged in interval detection or figures. Models can still fit across
such gaps; those gaps receive no interpretable inference. The reference is the
first four-group-supported grid point (zero when supported), recorded per result.
It is a modeled common mean reference, never an individual's biomarker baseline.

Historic ever, cumulative, ordinal, count and complex discrete outcomes stay in
the inventory with explicit reasons. ESSDAI/overlap-derived outcomes are
mechanically related and descriptive only. Numeric types alone do not override
known ordinal/count metadata (including ocular Oxford/van Bijsterveld scores).

## Tests and multiplicity

- A: joint group × time Wald test (3 df).
- B vs A: common nonlinear terms (2 df for four knots; 1 for three), ML LRT for regular comparable mixed
  models or robust Wald for GEE.
- C vs B: group-specific curvature (6 df for four knots; 3 for three), appropriate LRT/Wald as above.
- Primary C test: **joint linear and nonlinear group interactions (9 df for four knots; 6 for three)**,
  excluding group intercept differences.

The same records, family, random structure, knots and scales are used for all
comparisons within a scenario. All attempted fits retain convergence, warning
classes, rank, condition, covariance/information regularity, residual diagnostics
and random-intercept boundary checks. Failed fits/tests have no p-values.
AIC/BIC/log-likelihood are exported only for ML models; GEE fields are undefined.

BH-FDR covers **all estimable independent variables** separately for each
test, threshold, scenario, clock and basis specification. It does not split the primary
family into favorable priority subsets. Domain-specific BH is supplementary;
both denominators are recorded. There are no discoveries selected by AIC or by
a Step 16 p-value. Pairwise separation uses the simultaneous bands, never the
first significant pointwise CI. Separation segments do not cross unsupported
regions or a change in sign. Grid, model and domain dependence remain explicit.

The Gaussian coefficient bands are asymptotic, not a claim of empirically
verified 95% coverage. Optional patient bootstrap resamples complete patient
histories within fixed groups, freezes knots, compares the same validated family
and reports failures. It provides an aggregate sup-t critical-value sensitivity;
it does not silently replace the primary bands. Binary patient-bootstrap timing
calibration is not implemented and is labeled nonestimable. Maximum-velocity
windows and T50 remain `NOT_ESTIMABLE_TIMING`: no stable interior timing with
validated uncertainty is asserted by this version.

## Export identity and evidence classification

Canonical `outcome_id` encodes domain and canonical source column; variable,
domain, outcome type, unit, threshold, clock, scenario and basis specification
are checked together in result joins. Homonymous
lab/serology variables cannot exchange tests, comparisons or figures. Duplicate
outcome/test rows raise a contract error before publication. Primary BH families
continue across domains; supplementary domain q-values never drive classification.

`n_patients` retains the table's local meaning (group, or group and annual
window); `n_patients_analysis` separately describes the entire analysis sample.
Metadata uses `variable_reference_policy` and `inference_family`; temporal rows
use `inference_reference_policy`, and bootstrap uses `bootstrap_family`.
Unexpected annotation collisions raise an informative error. Patient counts
1–4 and their derived percentages are suppressed in CSVs and figures; analysis
totals are withheld when they could reconstruct a suppressed group. Identifying
columns and patient cells remain prohibited by `safe_table()`.

A/B/C eligibility, fit and test states are separate. Classification reuses the
four global tests and their original across-domain q-values/denominators:

- A describes linear group-slope evidence even when C is unavailable.
- B versus A describes common curvature, without claiming group-specific curvature.
- C versus B receives `EXPLORATORY_NONLINEAR_GROUP_HETEROGENEITY` only for a valid
  specific test with q < .05; otherwise `NO_EXPLORATORY_NONLINEAR_EVIDENCE` or
  `C_NOT_ESTIMABLE`.
- The global C interaction has a separate exploratory global classification;
  it cannot establish nonlinear group evidence by itself.

All CSV schemas include version, basis specification, number of knots, nonlinear
df, analysis role, support policy, sample fingerprint and unit. The run manifest
includes policy/version, basis execution mode, exploratory role, effective parameters,
cutoff, commit/code and input hashes, counts of eligible/estimated/failed models
A/B/C, valid C tests, estimator families, reasons and supported temporal coverage.
Per-bundle QC also reports inventory totals, covariance/convergence/other C failures,
A/C curve coverage and supported grid times per outcome. Fit failure summaries
use the final coherent family; all earlier ML attempts remain in the attempts CSV.
QC PASS certifies software and provenance checks, not confirmatory or clinical validity.

## Sensitivities and Step 16 reconciliation

`standard` runs primary capped common support, full common horizon, the **same
full estimation sample on the primary reporting domain**, official-variable
baseline only, delayed-variable baseline only, ≥3 variable dates, ≥3 ESSDAI dates,
and the alternate clock (excluding pre-index observations). `full` adds official
vs delayed ESSDAI index subsets. Every scenario recomputes eligibility/support
without changing group identity. No groups are dropped to manufacture overlap.
The identical full-sample comparison freezes estimation rows and basis; a
separate main-vs-full comparison changes composition. Sample digests and counts
make that distinction inspectable. Optional observation-intensity weighting is
not fit: verified measurement-opportunity denominators are absent.

`07_step16_reconciliation.csv` calls Step 16's linear **function** on the exact
Step 17 main sample and verifies the design and coefficients when families
match. It never reruns the Step 16 command or changes its products. A different
validated family is explicitly labeled. Published full-horizon Step 16 results
are not treated as comparable to capped Step 17 results without shared rows.

The supplied proposal describes a previous IgA Worsening–Stable sign reversal
between full linear and 0–3-year fits. That is context supplied by the proposal,
**not independently reproduced clinical evidence in this implementation**.
`01_linear_pairwise_slopes.csv` and the consistency table check this for IgA and
every other variable using identical rules, with sample digests and patient
counts to distinguish composition from domain changes. Neither spline choice
nor horizon depends on the observed sign of IgA.

## Outputs

Every bundle has:

```text
outputs/{tables,figures,qc}/blockA/17_nonlinear_temporal_heterogeneity/temporal-spline-v3/
  <four_knots|three_knots>/<time_axis>/delta<3|5>/<scenario>/
```

The `main` scenario uses the frozen capped horizon; the three-knot specification remains post-hoc exploratory. Synthetic inputs always
get an additional `synthetic_validation/` root. Empty tables have readable headers.

| Table | Contents |
|---|---|
| `00_variable_eligibility` | A/B/C decisions, reasons, semantic type, sample counts and serialized basis |
| `00_local_temporal_support` | Local unique counts, percent, span and four-group support at every grid point |
| `00_design_diagnostics` | Per-model rank, number of columns and design condition, including rejected designs |
| `00_group_support` | Per-group patient/repeated/three-date/date counts |
| `00_aggregated_observations` | Patient-balanced annual descriptive means/quantiles; small cells suppressed |
| `01_model_comparison` | A/B/C family, status, parameters, ML criteria and basis |
| `01_fit_attempts` | Every attempted model/family with QC and safe warning classes |
| `01_model_coefficients` | Ordered coefficient estimates and SEs; index follows A/B/C design |
| `01_linear_pairwise_slopes` | Six model A slope differences for horizon/composition audit |
| `02_global_temporal_tests` | Four prespecified tests, p, overall/domain q and denominators |
| `03_group_curves` | Supported means for every valid A/B/C; C simultaneous bands |
| `03_within_group_change` | C change from recorded common supported reference |
| `04_temporal_derivatives` | C velocities with units/year and uncertainty |
| `04_pairwise_temporal_differences` | Six pairs × level/change/velocity with pointwise and simultaneous limits |
| `04_simultaneous_bands` | Explicit sup-t method, critical value, RNG and joint family |
| `05_separation_intervals` | Contiguous supported separation; not biological onset |
| `05_temporal_characteristics` | Explicit nonestimable timing registry |
| `05_heterogeneity_classification` | One canonical outcome × test row; eligibility, fit and test states, p/q, FDR denominator, reason, support policy and exploratory role |
| `06_sensitivity_and_threshold_consistency` | Global tests and signs across thresholds/scenarios with composition metadata |
| `06_non_estimable_registry` | Missing model/function components and exact reasons |
| `06_patient_bootstrap_validation` | Optional patient-level bootstrap summary and failures |
| `07_step16_reconciliation` | Shared-sample linear verification or explicit noncomparability |

`qc/run_manifest.json` records input/code hashes, Git identity and dirty flag,
cutoff, parameters, software versions, artifacts, QC status and inferential
limits. `PASS` certifies successful software/contract processing, **not clinical
approval, sufficient support for every variable or calibrated real-data coverage**.
A dirty worktree remains explicitly preliminary with code hashes.

Figures are fixed-palette four-group A/B/C atlases, velocity panels, all six
contrast panels for each estimand, local-n QC plots, a heterogeneity heatmap and
a domain evidence dashboard. Grey nonestimability is distinct from an estimated
absence of evidence. No patient trajectories or individual points are exported.

## Privacy and validation status

Publication staging/directories use 0700. Exports forbid identifying columns and
patient-identity cells; warning messages are reduced to classes. Small nonzero
count cells below five and their local percentages are suppressed; group-count
JSON is suppressed similarly. Aggregate outputs remain access controlled and
need the institutional disclosure policy before external release. Synthetic
figures and outputs must never be interpreted as patient findings or committed
as clinical artifacts.

Automated tests cover nested designs, frozen transforms, natural boundaries,
analytic derivatives, null/common/group-specific/crossing synthetic signals,
coherent fallback, A failure, real repeated binary probability gradients, sparse
groups, unsupported reference/gaps, known BH values, immutable groups/thresholds,
semantic exclusions, zero-write dry-run, both-threshold end-to-end publication,
privacy, hashes, figures and shared-sample Step 16 reconciliation.

```bash
python -m pytest tests/test_temporal_spline_*.py -q
```

No canonical patient data are present in the implementation environment. The
real-data dry run and 1–3-variable institutional smoke above are therefore still
required before a full clinical run. No upstream study or real cohort analysis
has been executed or claimed here.

Implementation validation on 2026-10-28: **64 tests passed** (27 Step 17 tests
and 37 Step 16 contract/model/output/feasibility regression tests), using Python
3.12 and statsmodels 0.15.0. The synthetic end-to-end test renders all figure
families, exports all tables and checks artifact hashes. Gaussian null-band
simulation and a 100-replicate whole-patient GEE bootstrap also pass. Dependency
deprecation warnings and existing empty-slice warnings remain nonfatal.

```bash
python -m pytest tests/test_temporal_spline_*.py tests/test_secondary_comparison_*.py -q
```

Revision validation (2026-10-29): see the [C_RELAXED_SUPPORT_V1 software QC report](17_c_relaxed_support_v1_validation.md) for the executed 88-test regression, synthetic A/B/C counts and blocked institutional prerequisite check.

V3 acceptance coverage and numerical V2 compatibility are recorded in
[the V3 software validation report](17_temporal_spline_v3_validation.md). Clinical
recovery of the previously reported 23 + 7 A-only outcomes is pending execution
on identical authorized inputs in Biowulf; synthetic QC cannot certify that count.

## Global longitudinal validation supplement (`validation_global_v1`)

Historical V1 contract. The current entrypoint writes V2; see
[Step 17 validation V2](17_validation_global_v2.md) for current policies and commands.

The supplement evaluates **every registered outcome**, including missing,
noneligible, ordinal/count, mechanically derived and historical outcomes. It
has its own command and namespace; the ordinary Step 17 command and its published
V3 tables/manifests are unchanged. ALT, IgA and MCV have no special validation
rules. Software validation below uses synthetic fixtures only; it does not assert
that the clinical cohort has been validated or clinically reviewed.

```bash
# Read/authenticate the four COMPLETE original V3 main bundles and cover all outcomes.
python src/block_A/17_validate_temporal_heterogeneity.py

# Intensive validation for all fitted A/B/C candidates (potentially expensive).
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth deep --all-fitted --bootstrap-replicates 200

# First budgeted batch. This limit includes full-trajectory LOO and bootstrap refits.
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth deep --max-patient-refits 1000 --max-outcomes 8

# Continue the IDENTICAL plan; preserve the previous validation snapshot in backup.
python src/block_A/17_validate_temporal_heterogeneity.py \
  --depth deep --bootstrap-replicates 200 --max-patient-refits 1000 --resume \
  --backup-root /authorized/project/outputs/backups/17_nonlinear_temporal_heterogeneity/validation_global_v1
```

If a systematic validation snapshot already exists, start intensive work with
`--depth deep --resume --backup-root ...` to preserve that snapshot, even for
the first intensive batch. Each checkpoint scope has one writer at a time.

Run canonical data only inside the authorized NIH environment. The no-argument
cutoff comes from the four authenticated reference manifests, never the system
date or a hard-coded assumption. An explicit `--data-cutoff` must match them.
`--reference-root` can point to a different authorized project containing V3;
by default it equals `--output-root`. Synthetic inputs/references/outputs are
separated under each root's `synthetic_validation/` directory. The original
reference runs must have `analysis=all` and
`fdr_selection_scope=all_registered_variables`; subset runs cannot substitute
for the complete reference. Missing or hash-invalid bundles fail closed.

`--threshold {3,5,both}` and `--basis {four_knots,three_knots,both}` explicitly
scope the new snapshot. Both default to `both`; all four source references are
still authenticated. `--outcomes-file` is an optional JSON list of **canonical
`outcome_id` strings**, used as a predeclared clinical priority for intensive
work, not as a filter on the systematic inventory. `--max-outcomes` limits the
number of new distinct intensive outcomes per invocation; `--max-patient-refits`
limits reserved refits. Neither changes the complete systematic denominator.
`--seed` controls deterministic queue ties and validation RNG only. Original
model parameters, grid, basis, projection/scaling and QC limits come from V3.

### Comparability and systematic diagnostics

The runner authenticates every artifact in all four reference manifests and
checks their common cutoff/parameters, origin, version and named support policy.
It requires an exact catalog/inventory match and A/B/C model rows per outcome.
For each `outcome_id × threshold × basis × clinical_baseline × main`, it
reproduces the analytic sample fingerprint, unit/type, eligibility, frozen basis,
counts, local support and rank/condition diagnostics. Suppressed original counts
remain suppressed; they are not treated as observed zeros. Discrepancies produce
`NOT_COMPARABLE_REFERENCE`, exclude intensive evaluation and set software QC to
FAIL. An incomplete/corrupt reference blocks publication entirely. The intensive
mode additionally reproduces the original coherent model family and compares all
published coefficients and their standard errors before validation refits.

Systematic mode makes **zero model fits**. For every outcome/group it reports
coverage on `[0,3]`, distinct measurement dates, repeated-patient counts,
follow-up-span quartiles and disjoint early `[0,.50]` / late `[1.75,2.50]` windows
(inclusive endpoints). `n_neither` includes fixed-group members with no analytic
observations in either window. `pct_both_of_group` uses all fixed-group patients;
`pct_both_of_early` uses unique early contributors. The four-way early/late
partition is distinct from the original overlapping one-year prediction windows.
If a diagnostic window is entirely unobserved, its absence does not by itself
invalidate a trajectory: the predefined continuity gate uses the full-horizon
repeated-patient fraction for that group instead of an undefined paired overlap.
Both fractions and the rule remain explicit in the inventory.
It reports supported grid-point counts, span, segment count and the sum of
adjacent supported intervals; no gap is filled to inflate covered duration.

For applicable numeric/binary outcomes, one arithmetic mean per patient/window
precedes population descriptives and late-minus-early paired summaries. Binary
change is the difference in each patient's mean binary response (probability
scale), with the paired-patient denominator. Ordinal/count/complex scores,
historical/nonlongitudinal and ESSDAI-derived outcomes retain coverage rows but
receive `NOT_APPLICABLE_TYPE` for numeric change/distribution inference. Medians,
IQR, MAD and skewness describe distributions; asymmetry is an audit signal, not
an exclusion rule. Units are checked against the authenticated catalog. No
individual extremes, exact dates or patient traces are exported.

### Intensive plan, estimands and resumption

Every comparable fitted A/B/C is a candidate, regardless of its p/q. A frozen
queue cycles one task from each nonempty **model × type × threshold × basis ×
domain** stratum per round. Within a stratum, predeclared clinical priorities
precede `sha256(seed, semantic task key)` ties. This supplies explicit diversity
quotas without selecting only significant findings. Both the plan and its hash
are persisted **before refits**. Every unexecuted candidate is
`DEEP_PENDING_RESOURCE_BUDGET`, with its queue position. No robustness percentage
is attributed to the whole inventory from a selected intensive subset.

A task consists of one outcome/model/threshold/basis. LOO removes a patient's
**entire trajectory**, rebuilds the frozen design, and fits exactly the original
family. `fit_fixed_family()` applies the original rank, condition, convergence,
covariance, separation, residual, repeated-patient/event support and warning checks
and never selects another
estimator. Failed and noncomparable refits stay in the denominator. Comparisons
use only the intersection of original/deletion-supported points; C change
contrasts also require the original reference point. No temporal interpolation
or gap bridging is performed. Influence exports pool across deletions and report
robust quantiles by estimand/group/contrast; no omitted-patient rows or rankings
are persisted.

The patient bootstrap shares the original resampling implementation. It draws
complete trajectories with replacement within each ESSDAI group and gives each
duplicate an internal fresh cluster ID. Refits freeze model, family, basis and
original inference grid. A replicate losing any original supported point fails
instead of shrinking its inferential family. Joint sup-t families are:

| Model | Simultaneously calibrated family |
|---|---|
| A | Four group slopes plus six pairwise slope differences |
| B | Shared curvature on the supported grid plus four group trajectories |
| C | Six pairs × all original supported times × level/change/velocity |

The default is 200 requested bootstrap replicates. Calibration requires
`max(100, ceil(.8 × requested))` comparable successes; failures and reasons are
retained. Analytic coefficient simulations retain the reference configuration seed; the
validation seed controls patient resampling. Analytic/bootstrapped critical values, global sup-t quantiles and
simultaneous-band sign agreement are reported. Binary response-scale bootstrap
calibration is explicitly `NOT_ESTIMABLE_METHOD`; binary LOO describes link-scale
A slopes/B curvature and response-scale trajectories, but cannot establish
`SUPPORTED_EXPLORATORY` without a validated bootstrap method.

Checkpoint files live only at
`<root>/work/17_nonlinear_temporal_heterogeneity/temporal-spline-v3/validation_global_v1/<scope>/`.
They contain **disclosure-controlled aggregates only**, a frozen plan/context
hash, and payload hashes. Temp-file/fsync/atomic rename guards interrupted writes.
Resume rejects changes to inputs, references, code, parameters, software versions,
seed or requested bootstrap count. Budgets may increase. Completed tasks are
immutable and skipped, and cached results republish idempotently. A model task is
the interruption boundary: interruption mid-task restarts that task; neither
patient-level checkpoints nor partial models are published. A budget too small
for an entire task leaves it pending. Initial-family reconstruction reserves up
to six fits in addition to its LOO/bootstrap cost. There is no wall-clock timeout
or per-replicate resume; bound scheduled job runtime using the institutional
scheduler and choose an appropriate whole-task budget.

Resume of an already published snapshot requires an explicit `--backup-root`
below `outputs/backups/17_nonlinear_temporal_heterogeneity/`. It replaces only the
new validation namespace with a transactional snapshot after all selected tasks
finish, preserving pending tasks as pending. Staging/rollback and final input,
reference and code hash checks protect the previous snapshots and V3 originals.

### Scientific states, disclosure and output contracts

`software_qc_status` and `scientific_validation_status` are independent. The
predeclared materiality rules in the frozen plan are descriptive standardized
rules, **not clinical MCIDs**: a >1 SE LOO shift at any supported estimand (including an isolated influential
trajectory), or a 95th-percentile standardized shift >1 SE,
material direction-change fraction >.20, or LOO failure fraction >.20 flag
fragility when bootstrap is valid. Insufficient comparable refits/calibration
are inconclusive. Continuity <.20 in any fixed group (paired-window overlap, or the
full-horizon repeated fraction when either diagnostic window is absent), LOO supported-grid retention
25th percentile <.80, simultaneous-band
agreement <.80 or bootstrap/analytic critical ratio >1.50 also prevent support.
Continuous diagnostics accompany labels. `SUPPORTED_EXPLORATORY` requires both
intensive components complete and adequate overlap; p/q never enters that rule.
These thresholds are fixed globally and cannot be changed per outcome.

Patient IDs are used in memory only. CSV columns and values plus nested JSON
keys/values are audited before writing. Linked diagnostic patient totals,
subtotals, percentages and distribution/change summaries are conservatively
suppressed **across tables for the entire outcome** if any patient partition or
sign cell contains 1–4. Bootstrap-replicate counts describe draws rather than individual patients and
remain visible when no patient cell is suppressed. LOO count totals and failure histograms are suppressed
together when small cells could reveal one deletion. This may hide otherwise
usable aggregates; suppression never alters internal scientific classification.
Task/outcome denominators are software coverage counts, not patient counts.
Institutional disclosure review remains necessary before releasing any clinical
CSV/JSON. The command does not publish a ZIP or send clinical outputs elsewhere.

Each selected bundle publishes eight `07_*.csv` tables and QC JSON under:

```text
outputs/{tables,qc}/blockA/17_nonlinear_temporal_heterogeneity/
  temporal-spline-v3/validation_global_v1/<basis>/clinical_baseline/delta<3|5>/main/
```

Tables are `07_outcome_validation_inventory`, `07_patient_temporal_coverage`,
`07_within_patient_change`, `07_distribution_diagnostics`, `07_influence_summary`,
`07_fit_stability`, `07_patient_bootstrap_qc`, and
`07_scientific_validation_classification`. Inventory and scientific rows cover
all outcomes × A/B/C, even when intensive work is inapplicable/pending; diagnostics
cover all outcomes × groups. All rows retain canonical identity, domain/type/unit,
threshold/basis/clock/scenario, sample fingerprint and validation version/depth.
`reference_q_value` and `reference_family_n_tests` come unchanged from the original
**full** V3 test family; C's curvature test is retained separately. Validation
never recomputes FDR on a selected subset.

`run_manifest.json` records provenance, code/upstream/reference/artifact hashes,
UTC timestamp, versions and seed. `validation_summary.json` gives full inventory,
model/domain denominators, task states, plan/budgets, limitations and a stability
fraction whose visible denominator is **only intensively completed models**.
Neither software PASS nor the same-cohort bootstrap constitutes independent
clinical validation.

Reproduce software checks with synthetic data only:

```bash
python -m pytest -q tests/test_temporal_spline_validation.py
python -m pytest -q tests/test_temporal_spline_*.py \
  tests/test_secondary_comparison_*.py tests/test_secondary_trajectories_by_essdai_pattern.py \
  tests/test_observed_patterns_v2.py
```

