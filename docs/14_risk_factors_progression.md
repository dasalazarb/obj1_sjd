# Step 14: risk-factor associations with subsequent progression

Run after the canonical integrated dataset (Step 10), official baseline (Step 11),
and disease-activity definitions (Step 13):

```bash
python src/block_A/14_risk_factors_progression.py --dry-run
python src/block_A/14_risk_factors_progression.py
```

The script consumes canonical products only. It imports the activity classifier,
first-observed Low → Moderate/High and Moderate → High risk-set builders, domain
aliases, and mixed-model acceptance checks from Step 13. Its optional Step 13
Parquet is checked against Step 10, including the actual `essdai_total` export
name. A missing optional progression product is logged; definitions still come
from Step 13. No raw clinical forms, BTRIS records, new baseline, serologic
thresholds, phenotype reconstruction, or imputation are used.

The Step 10 `10_integrated_longitudinal_context.parquet` companion is discovered
beside `--integrated`, or supplied with `--context`. Only lab dates, days from
anchor, units, and evaluability metadata are joined. Episode keys must match
exactly and shared fields must agree. Official Step 11 baseline rows receive
metadata through their existing episode keys; their predictors are preserved.
Labs without embedded or companion as-of timing evidence are unavailable for
prospective analysis. The registry records canonical names and explicit aliases;
absent glandular features remain unavailable. Complement C3/C4 and rheumatoid
factor spellings follow the repository, and high-sensitivity CRP is a separate
sensitivity predictor.

## Temporal and outcome contracts

Baseline characteristics are related to assessments strictly after official
clinical baseline. A slope model requires two subsequent distinct assessments.
Time-varying predictors come from the FROM episode of an adjacent canonical
clinical interval; TO supplies outcomes. Missing middle episodes and same-date
pairs never cause earlier and later episodes to be bridged. State-transition
fields retain missingness when either ESSDAI is unknown.

ESSDAI ≥5 means Moderate/High activity; ≥14 means High. No Pop2 progression
endpoint is used. Total ESSDAI +5 is a sensitivity endpoint, using the upstream
change threshold; it is never applied to 0–3 ordinal domains.

Incident target domains must be confirmed inactive at official baseline and at
FROM, with an evaluable TO assessment. First-event risk stops at the first
observed positive visit even if that interval lacks a predictor, so recurrence
and later recovery cannot reset eligibility. Baseline domain survival datasets
censor at the last evaluable target assessment. Global `any_new_domain` uses
only domains confirmed inactive at baseline; a negative global censor requires
all those domains to be evaluable. A positive target suffices to establish the
global event even when other targets are missing.

Dated upstream lab selections must be known at/before FROM. The upstream
nearest-value median uses tied nearest measurements and prefers pre-anchor
evidence; it retains the selected measurement date's temporal scope. Future and
undated measurements and invalid selections are excluded, with availability
counts. Raw values remain intact. Different reported units are not combined
without a canonical conversion; such models are skipped. No retrospective
patient-consensus serology enters the time-varying registry.

Cox uses the first observed positive visit time. Biological onset is interval
censored between the last evaluable negative and first positive assessment.
Non-evaluable adjacent intervals are excluded from interval models and never
bridged, so their estimates apply to observed evaluable intervals.

## Models and support

Each association uses one prespecified predictor plus adjustment, rather than
an automatically selected multivariable predictor panel.

- Baseline ESSDAI trajectory: random-intercept MixedLM, predictor × time effect.
  Singular, boundary, non-converged or invalid inference falls back to Gaussian
  GEE with exchangeable correlation and patient clustering. Attempted
  coefficients and diagnostics remain in model QC.
- Baseline first-event endpoints: unpenalized Cox PH with a predictor-specific
  rank-time PH test. PH warnings retain estimates with caution labels.
- Subsequent ESSDAI: Gaussian GEE, adjusted for starting ESSDAI, interval length
  and available baseline covariates. A constant starting-state or spacing field
  is redundant with the intercept and is explicitly recorded as constant.
- Time-varying events: piecewise-exponential Poisson GEE with robust patient
  clustering, independence working correlation, and log(interval years) offset.
  Fixed FROM-time bands are <1, 1–3 and ≥3 years. This explicit fallback avoids
  relying on lifelines' unsupported robust time-varying Cox variance; IRR and HR
  are labeled separately.
- Cross-domain pairs: secondary FROM active A → first incident B models, A ≠ B.
  Every pair receives support counts; only supported pairs are fitted.

Default safeguards are 20 patients, 10 repeated patients for trajectories,
20 intervals, 10 events, 5 exposed and 5 unexposed patients, 3 exposed and 3
unexposed events, and 5 events per fitted parameter. Cross-domain pairs require
5 exposed events. `--minimum-events` may raise the event gate but cannot lower
it. These are conservative safeguards, not inferential guarantees.

Full adjustment is baseline ESSDAI, age, sex and disease duration. If full
complete-case support fails, the prespecified reduced set is baseline ESSDAI
only; no-total sensitivities and lagged models use age only in the reduced set.
Unavailable/constant adjustments, model rank, event support and exact attrition
are reported. Starting ESSDAI remains in lagged models when it varies. No
stepwise selection or silent threshold reduction occurs. Domain predictor
models flag their structural relationship to baseline ESSDAI total.

Continuous predictors use 1-SD effects, preserving their raw values. Baseline
scaling uses one row per patient in the outcome-eligible source cohort; interval
scaling uses available FROM observations. The mean, SD and cohort are recorded.
Ordinal domain effects use +1 level; active/inactive comparisons are separate
sensitivities. BH-FDR includes only valid selected inference within distinct
analysis, predictor-family and outcome families; sensitivities are separated.

Sensitivities include total ESSDAI +5, domain ordinal versus binary, domain
models without baseline total, at least 0.5 years between the first and last
subsequent trajectory assessments, continuous laboratories, and an upstream
anchor-day lab restriction when days-from-anchor exists. No new generic lab
recency window is imposed. Treatment adjustment is explicitly not implemented;
individual comorbidity analyses remain in Step 07. All interpretation describes
associations, not causal effects or a deployable prediction model.

## Outputs and CLI

All outputs are isolated under `14_risk_factors_progression` in the standard
analytic, Block A tables/figures/QC, and logs directories. `--output-root` changes
the project root for those same relative paths. The three Parquet products are:

- `14_baseline_predictor_dataset.parquet`: one official baseline row per patient.
- `14_lagged_interval_dataset.parquet`: adjacent FROM/TO intervals and predictors.
- `14_domain_incidence_risk_set.parquet`: patient/target-domain first-event rows
  with wide baseline predictors and eligibility metadata.

All required result and QC tables are emitted, including unavailable and sparse
rows with support reasons. Main forest plots contain valid selected models; an
unsupported analysis receives a labeled empty plot. The secondary cross-domain
heatmap appears only with at least four valid pairs; unsupported cells are gray,
not zero effects. Metadata records gates, inputs, methods and software versions.
The log records availability, feasibility, attempts, fallbacks, warnings and
output locations. Review structural/temporal QC before feasibility and models.

CLI overrides: `--integrated`, `--baseline`, `--context`, `--progression`,
`--output-root`, `--minimum-events`, `--overwrite`/`--no-overwrite`, `--dry-run`.
Outputs regenerate by default. `--no-overwrite` protects all existing dedicated
outputs before any write. Dry run validates datasets, builds risk sets and
feasibility, prints planned models, and writes only QC/log diagnostics. Structural
or temporal disagreements fail loudly, with available diagnostics saved.

Verification:

```bash
python -m pytest -q tests/test_risk_factors_progression.py
```
