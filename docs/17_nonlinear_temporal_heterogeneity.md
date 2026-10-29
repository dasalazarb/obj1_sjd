# Step 17: nonlinear trajectories and temporal heterogeneity

Step 17 consumes the authenticated Step 10/11/13 contracts through the existing
Step 16 adapter. It never runs or changes Steps 10/11/13/15/16. Fixed retrospective
ESSDAI memberships at ±3 and ±5 are fitted separately. These are associations
conditional on the observed history, not prediction, causation, onset or latent
patient states. Nonestimability is a result, not evidence of equal trajectories.

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
  A (8 columns) is a literal prefix of B (10), itself a prefix of C (16).
- `temporal_spline_models.fit_family`: ML random-intercept A/B/C with QC; if any
  fails, discard the partial family and refit all eligible models with robust
  Gaussian GEE. Binary repeated outcomes use robust binomial GEE. No jumps over
  a failed A/B. A/B remain possible when C support fails. Degenerate spline knots
  leave A only; no silent change to a different spline family.
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
- `--figures priority`, `--progress` (enabled), 250 DPI.
- `--dry-run` disabled; outputs are written and existing output paths
  remain overwrite-protected.

When successful, the final JSON contains `qc_status: PASS`, the bundle count,
the cutoff and output directory paths. CSV/PNG/manifest files live under
`<project>/outputs/{tables,figures,qc}/blockA/17_nonlinear_temporal_heterogeneity/`
with `<time_axis>/delta<3|5>/<scenario>/` suffixes. Previous published results
are never overwritten silently. `--overwrite` still requires a valid
`--backup-root`.

The `--analysis all` default fits eligible A/B/C models, tests temporal
heterogeneity, applies multiplicity control, makes priority figures and
publishes both thresholds. `--analysis temporal` also fits its own models;
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
{"version": "temporal-spline-v1", "max_primary_years": 3,
 "grid_step": 0.25, "minimum_three_date_patients": 5,
 "simulations": 4000, "seed": 1701}
```

No checkpoint/resume is implemented. In-memory patient rows are discarded
after each variable. Existing destinations are blocked unless both
`--overwrite` and `--backup-root` are provided. Backups must be under
`<output-root>/outputs/backups/17_nonlinear_temporal_heterogeneity/`
(and under `synthetic_validation/` for synthetic inputs).

## Support and estimands

All fits use the intersection of observed group ranges, with nonnegative time
and a primary maximum of three years. At least 10 patients and 8 repeated
patients per group, three distinct group times and a common span of at least
one year are required even for A. For C, each group also needs at least five
patients with three distinct dates spanning at least one year, and four-group
local support in three annual windows and all three thirds of the reporting
horizon. These are operational limits, not power guarantees.

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
- B vs A: common nonlinear terms (2 df), ML LRT for regular comparable mixed
  models or robust Wald for GEE.
- C vs B: group-specific curvature (6 df), appropriate LRT/Wald as above.
- Primary C test: **joint linear and nonlinear group interactions (9 df)**,
  excluding group intercept differences.

The same records, family, random structure, knots and scales are used for all
comparisons within a scenario. All attempted fits retain convergence, warning
classes, rank, condition, covariance/information regularity, residual diagnostics
and random-intercept boundary checks. Failed fits/tests have no p-values.
AIC/BIC/log-likelihood are exported only for ML models; GEE fields are undefined.

BH-FDR covers **all estimable independent variables** separately for each
prespecified test, threshold, scenario and clock. It does not split the primary
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
outputs/{tables,figures,qc}/blockA/17_nonlinear_temporal_heterogeneity/
  <time_axis>/delta<3|5>/<scenario>/
```

The `main` scenario is the prespecified capped analysis. Synthetic inputs always
get an additional `synthetic_validation/` root. Empty tables have readable headers.

| Table | Contents |
|---|---|
| `00_variable_eligibility` | A/B/C decisions, reasons, semantic type, sample counts and serialized basis |
| `00_local_temporal_support` | Local unique counts, percent, span and four-group support at every grid point |
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
| `05_heterogeneity_classification` | Variable-level evidence/QC; not patient phenotypes |
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
