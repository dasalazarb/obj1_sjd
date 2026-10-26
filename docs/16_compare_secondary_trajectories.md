# Step 16 — comparative secondary trajectories across observed ESSDAI patterns

Step 16 asks whether independently measured symptoms, laboratory values and glandular function have different observed trajectories across the four **fixed retrospective ESSDAI v2 groups**. It does not classify patients, estimate progression risk or establish causality. Synthetic software validation is distinct from clinical validation.

## Interface and protected inputs

```bash
PY=/usr/local/Anaconda/envs/py3.10/bin/python

# Inspect feasibility first: no model fitting, figures or output directories.
$PY src/block_A/16_compare_secondary_trajectories.py \
  --threshold both --analysis feasibility --dry-run

# Export only feasibility/QC in a separate namespace.
$PY src/block_A/16_compare_secondary_trajectories.py \
  --threshold both --analysis feasibility

# After scientific review of support, freeze variables and configuration.
$PY src/block_A/16_compare_secondary_trajectories.py \
  --threshold both --analysis all --priority-only \
  --config docs/16_secondary_comparison_config.json

# Explicit, limited fitting check with no writes (distinct from dry-run).
$PY src/block_A/16_compare_secondary_trajectories.py \
  --variables esspri_total complement_c3 --validate-models --validation-limit 2
```

The default Step 10 integrated parquet and Step 11 official-baseline parquet are resolved through `common.py`. Step 13 v2 assignment, eligibility, selected-date and manifest files must already exist. All manifest-listed artifacts and the consumed Step 10/11 bytes are authenticated with Step 15's `verified_manifest` and `validate_inputs`. Unknown groups, duplicates, incompatible baseline identities, unverified calendar dates, future dates after `--data-cutoff`, mismatched origins or altered hashes abort the run. **No upstream producer is executed by Step 16.**

Optional `--context` and `--source-registry` default to Step 10 sibling files when present. They supply dates, units, assay/protocol metadata and catalog inventory; context cannot add a previously excluded clinical outcome. All consumed inputs, every Step 13 manifest-listed artifact and relevant local code are fingerprinted and checked again after analysis/rendering and immediately before installation. Code hash keys use resolved paths relative to the resolved repository root, including Biowulf aliases such as `/data` and `/vf/users`. Python, NumPy, pandas and statsmodels versions, git revision/dirty state and full config are recorded.

`--input-origin synthetic_validation` must match the authenticated Step 13 manifest and isolates output under `OUTPUT_ROOT/synthetic_validation/`. It cannot consume a canonical manifest or publish a real-cohort claim.

## Fixed groups and measurement semantics

Group labels, order and colors are imported from `observed_patterns_v2_figures.LABELS/COLORS`. Within threshold means no observed excursion reaching the ESSDAI threshold, not clinically inactive disease. The ±3 and ±5 assignments are analyzed independently. No inference pools their rows, group maps or denominators.

Step 15 supplies the typed catalog, measurement-date rules, valid scales, laboratory units, assay/version checks, duplicate resolution, official-episode precedence, references and observation extraction. Null measurements never become zero or negative tests. Unknown/incompatible dates, scales, units or assay versions prevent that outcome's use. Ocular side/scale and CRP assay variants retain separate catalog identities. Explicit overlap counts remain numeric; all `ovl__*` and `essdai__*` outcomes are conservatively flagged mechanically related by this consumer. They receive descriptive tables, no independent-discovery inference or FDR ranking.

`ever_positive_through_episode` is cumulative historical knowledge. It receives first-available and measured-official-baseline profiles where dates exist, not seroconversion or repeated logistic modeling. Undated baseline serology is excluded, not assigned a fabricated baseline date. Ordinal/nominal outcomes receive category counts and evaluable denominators, with `NOT_ESTIMABLE` for repeated inference until a validated model is explicitly prespecified. Counts are not forced into Gaussian or binary models.

## Feasibility before models

Each selected variable and threshold has all four group rows, including zero-support groups. `00_group_observation_support.csv` preserves Step 15's exact `N_group/n_any/n_repeated/n_ref_official/n_ref_delayed` fields for reconciliation and adds analysis-axis support: N with any, official baseline, first delayed, ≥2 or ≥3 dates; measurement counts; duration quartiles; ESSDAI-assessment counts; delayed index counts; preindex exclusions; observation rates and reference delay. Coverage by `[start,end)` windows contains one patient contribution per window, including zero coverage for groups in observed windows. Protocol availability counts unique patients within protocol/window; protocols are not additive patient denominators.

The gate initially requires, **in every group**, ≥10 evaluable patients, ≥8 repeated patients, ≥3 distinct analysis times and ≥1 year of common group time range. These are conservative software eligibility rules, not a power calculation. All minima are explicit in the versioned [JSON configuration](16_secondary_comparison_config.json); common gates have CLI overrides. No group is silently removed. Smaller or incompatible cohorts retain descriptions and an explicit reason for `NOT_ESTIMABLE`.

Absent an explicit, producer-verified measurement-opportunity denominator, `00_followup_observation_opportunities.csv` records `NOT_ESTIMABLE` and null missing percentages. It reports observed measurements, without equating missing episode rows to unperformed/negative tests. Informative visits and indication bias remain limitations; observation rates do not automatically supply valid inverse-intensity weights.

## Profiles, change, variability and burden

Official-baseline profiles require both the official episode and its exact measurement date. First-available profiles are separate and carry delay summaries. Numeric profiles show n, mean, SD, median and IQR; binary/category profiles show actual evaluable counts and Wilson 95% intervals. Supported independent numeric profiles have a four-group Kruskal–Wallis test and pairwise Welch mean differences/95% intervals and within-variable standardized mean differences. Categorical and mechanical profiles remain descriptive.

Change is `value-reference_value` **only for numeric outcomes and `is_delta_followup=True`**. Step 15 delta windows can include the reference Δ=0; Step 16 deliberately excludes it and records this distinction in `estimand_difference_log.csv`. Singletons never have change. Late-reference changes use `time_since_variable_reference_years` and remain separated from official-reference changes; references are never shifted to clinical baseline. Window summaries select the last actual measurement per patient, not the mean of visits. Window-change contrasts remain explicitly `NOT_ESTIMABLE`: heterogeneous exact reference times do not justify a naive independent test. Native-value repeated models are the primary inferential comparison.

Individual OLS slopes are descriptive estimates, with two-date slopes labeled provisional. Exploratory SD, range, successive absolute changes/rates, reversal count and residual SD require ≥3 dates and the configured span. Aggregates are stratified by exact visit count and duration band; rates can be unstable for short intervals. No individual slope is treated as an error-free response in inference.

Normalized AUC uses a configured **common fixed horizon**, initially 0–3 years. A patient must bracket the horizon with actual measurements, meet the minimum dates and maximum gap (initially 2 years), and have ≥3 points on the integration grid. Boundary values can be linearly interpolated **inside** observed support, with trapezoidal integration, never extrapolated. Insufficient patients have zero eligible AUC support rather than invented burden. These burden/variability tables are descriptive and exploratory, without unsupported bootstrap precision claims.

## Repeated models and diagnostics

The primary analysis uses native observed values and actual years since clinical baseline:

`value ~ group + time + group:time + patient random intercept`.

Within threshold is the fixed reference. The numeric model is ML MixedLM, not REML. It must converge, have finite coefficients, positive/nondegenerate fixed-effect covariance, acceptable design conditioning, random-intercept variance above its prespecified boundary and no convergence/runtime warning. A failed MixedLM is recorded as a failed attempt and falls back to Gaussian GEE with patient clusters, independence working correlation and robust sandwich covariance. GEE also must pass convergence, coefficient/covariance checks and warning diagnostics. Residual skewness and group residual-variance ratio have explicit conservative limits; failed models produce no valid p-values. These checks are diagnostics, not proof that the model or observation process is clinically adequate.

Binary outcomes use logistic GEE with unique-patient event/non-event gates, linear-programming checks for complete/quasi separation, coefficient/probability stability and robust covariance checks. Predictions are marginal probabilities with link-transformed intervals; time contrasts include absolute probability differences using the delta method. Rare/constant/separated outcomes produce `NOT_ESTIMABLE`.

Every valid primary repeated model reports a joint Wald test of its three group×time terms (df=3), fixed coefficients, six slope contrasts and support-limited marginal predictions. Three Within-threshold comparisons are prespecified; the other three pairs are exploratory. Binary slopes use log-odds/year; numeric slopes use native units/year. Numeric standardization uses the pooled SD of actually observed official-baseline values and is absent if that SD is unavailable or zero. A theoretical intercept is explicitly **not** labeled an observed-baseline comparison.

Predictions lie in the shared group time range and additionally require the configured number of **unique observed patients in every group** within a centered local window. Unsupported grid points remain null gaps in plots. Reporting times 1/2/3/5 years receive mean/probability contrasts only when supported; no point beyond the observed common range is published. The main model is descriptive and unadjusted. No age/sex/treatment covariate is guessed: baseline-adjusted sensitivity remains `NOT_ESTIMABLE` until an approved baseline mapping is supplied. No spline, random slope, unapproved transformation or inverse-visit weighting is silently fitted.

Model attempt, diagnostic and coefficient tables preserve the exact family, counts, support rule, warnings (class names, avoiding identifying error text), fallback reason, scenario and adjustment set. A local fit failure is visible; a global provenance failure blocks installation.

## Sensitivity and multiplicity

Frozen sensitivities include an official-variable-baseline subcohort, ≥3 variable dates, ≥3 ESSDAI dates, official/delayed ESSDAI-index source subcohorts, the alternate time axis, and a fixed horizon truncation. Each receives a newly computed four-group gate and effective denominator. The ESSDAI-index axis filters negative times for inference and reports excluded counts; change additionally excludes preindex references. Raw negative times stay in memory for support auditing. Separate ±3/±5 primary slope consistency is a comparison of signs/intervals across different group memberships, not an equivalence test.

Priority variables are prespecified in the versioned config and editable before analysis. Global valid p-values receive BH-FDR **within domain/priority family, threshold, time axis, scenario and profile**. The variable→family mapping can be frozen in JSON. Failed/nonestimable/nonfinite p-values and mechanically related outcomes never enter BH denominators. Pairwise trajectories have separate Holm families per variable, threshold, axis, scenario, contrast type/time and prespecified/exploratory role; baseline pairs have separate per-variable/profile Holm families. `family_n_tests` accompanies q-values. No selection by p-value, automatic confirmatory refit or fabricated q=1 for failures occurs. The discovery matrix orders clinical priority/domain/variable and retains effect size, uncertainty, support and QC, rather than ranking by p alone.

## Aggregate output and installation

Bundles are isolated under:

```
outputs/{tables,figures,qc}/blockA/16_compare_secondary_trajectories/
  {clinical_baseline,essdai_index}/delta{3,5}/{analysis}/
```

`--analysis feasibility` has its own namespace and never emits a false inferential bundle. Other modes (`baseline`, `change`, `longitudinal`, `sensitivity`, `figures`, `all`) likewise coexist. `--analysis figures` runs the supported primary models to build plots; it does not reconstruct individuals from Step 15 aggregates. `--dry-run` always stops at feasibility regardless of analysis mode. `--validate-models` explicitly allows bounded fitting without writes.

Tables include `00_*` support/exclusions, `01_*` profiles/contrasts, `02_*` change/slopes/variability/AUC, `03_*` attempts/QC/global tests/coefficients/contrasts/predictions/support, `04_*` FDR/discovery/sensitivity/threshold consistency, registry, analysis status, estimand differences and plot manifest. The run manifest contains artifact hashes, input/code hashes, fixed group denominators, config, versions, runtime and limitations. Effective model counts are in model/support tables; primary and sensitivity rows are distinct.

Figures use official colors and ≥200 DPI PNG or vector PDF. Availability heatmaps show n/N and percentages, including a window-specific panel; effect heatmaps show standardized slope effects with NE/M symbols. Supported priority outcomes have marginal curves with 95% intervals and local patient-n panels and forests of slope differences. `--figures all` extends outcome figures beyond the priority list; `--figures none` disables rendering. Unsupported variables still appear as NE in the comparison heatmap. Sparse distributions/individual trajectories are not exposed as points.

No patient-level CSV, identifiers or individual dates are exported. Patient intermediates are in memory and released per outcome. **Aggregate does not imply institutionally cleared:** these are restricted internal preliminary bundles, marked `NOT_CLINICALLY_REVIEWED`, and can contain small cells. Institutional disclosure review/suppression and clinical review are required before external circulation. This implementation does not claim to automate institutional approval or create public-ready results.

All exports stage under the output root's filesystem. Inputs/code are rechecked after rendering and before same-filesystem installation; QC is installed last as the completion marker. On any installation failure, all newly installed directories are removed and previous directories restored from retained backups. Existing output is rejected by default. `--overwrite` requires an explicit `--backup-root` below the effective root's `outputs/backups/16_compare_secondary_trajectories/`; backups are timestamped. Input overlap and output symlink redirection are rejected. Thresholds and analysis modes are not overwritten by unrelated runs. A model-level `NOT_ESTIMABLE` does not invalidate otherwise honest tables; QC PASS means provenance/software checks passed, **not** clinically valid inference.

## Validation and remaining scientific work

Run the synthetic/regression suite:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/test_secondary_comparison_*.py \
  tests/test_secondary_trajectories_by_essdai_pattern.py \
  tests/test_secondary_trajectory_performance.py
```

Fixtures cover known numeric slopes, zero interaction/unequal visits, binary probabilities/separation, protected v2 producers, both thresholds, exact baseline/delayed clocks, singletons, reference-Δ exclusion, preindex filtering, sparse/constant/ordinal/cumulative outcomes, multiplicity, provenance tampering, symlink hash keys, no-write feasibility, real fitting/figure export on fictional data, output collision, export failure, installation rollback and backups. Upstream tests cover producer/date/duplicate/unit rules without modifying their implementation.

The protected NIH cohort was not available in the implementation workspace. No real-cohort feasibility matrix, biomarker finding, clinical p-value or significant group difference is asserted. Before a real full run, inspect its feasibility matrix, freeze supported outcomes/families, validate assay/scale and model assumptions, examine residual diagnostics and visit patterns, review sensitivity populations and obtain institutional privacy/clinical clearance. Ordinal/nominal repeated models, clinically approved covariate adjustment, flexible time effects and justified visit-process weighting remain explicitly unsupported, not silently approximated.

The [implementation validation record](16_secondary_comparison_validation.md) lists the executed checks, synthetic cohorts, test counts and unavailable canonical validation.
