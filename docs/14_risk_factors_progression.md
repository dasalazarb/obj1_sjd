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

Adjustment is selected before fitting, in the fixed order Full → Reduced A →
Reduced B → Minimal. Each level checks its own complete cases, design rank,
patient/interval/event support and events per parameter. Effect sizes, p-values
and convergence are never used to choose the level. Numerical failure preserves
the chosen specification; only the existing estimator fallback is allowed.

| Level | Baseline trajectory/Cox and interval event models | Lagged ESSDAI |
| --- | --- | --- |
| Full | baseline_essdai + age + C(sex) + duration | from_essdai + interval_years + age + C(sex) + duration |
| Reduced A | baseline_essdai + age + C(sex) | from_essdai + interval_years + age + C(sex) |
| Reduced B | baseline_essdai + age | from_essdai + interval_years + age |
| Minimal | baseline_essdai | from_essdai + interval_years |

The predictor, trajectory time and interaction, Poisson FROM-time bands and
log(interval_years) offset remain constitutive terms at every level. Strictly
constant state/spacing fields are omitted as intercept-redundant and recorded;
constant demographic adjustments reject that candidate level. Sex uses observed
categories only; missing sex is excluded when that level requires it. Static age,
sex and disease duration always come from the official Step 11 baseline, joined
by patient_id. An unsupported Minimal row remains not estimable with exact counts.

Step 11 owns `disease_duration`: diagnosis to official clinical baseline, in
years (days / 365.25). This refactor does not derive duration from follow-up or
later diagnosis records. Negative values, values exceeding observed baseline age,
and diagnoses after baseline are excluded with availability counts, never clipped.
Pharma names `time_since_diagnosis_years`/`disease_duration_years` are schema hints,
not verified cohort aliases; the previously accepted `disease_duration_years`
alias, along with the unverified `demo__disease_duration` alias, is removed under
the versioned source policy. The legacy
`disease_duration_yrs_model` is clipped upstream and is not used. No additional
alias or Step 11 derivation is introduced without canonical schema evidence.

Every model records `adjustment_status`, `selected_adjustment_level`, covariates,
effective formula, selection rule/reason and JSON diagnostics for all four levels.
The detailed adjustment QC flattens those candidates. Historical full-model
support fields remain; `adjustment_status_legacy` maps reduced levels to the old
`prespecified_reduced` label. When no level is supported, selection is `none`,
and counts/formula describe the unsupported Minimal candidate.

Continuous predictors use 1-SD effects, preserving their raw values. Baseline
scaling uses one row per patient in the outcome-eligible source cohort; interval
scaling uses available FROM observations. The mean, SD and cohort are recorded.
Ordinal domain effects use +1 level; active/inactive comparisons are separate
sensitivities. `q_value` preserves historical BH-FDR within analysis,
predictor-family and outcome families; `q_value_family` is its compatibility
alias. Sensitivities retain their own historical BH families.

`q_value_outcome_wide` is a separate multiplicity sensitivity over valid primary
canonical hypotheses, grouped by analysis_type × outcome × effect_measure.
Baseline, time-varying and cross-domain analyses, and HR/IRR/beta scales, never
pool. Duplicate construct representations, sensitivities and invalid/nonfinite
p-values receive NaN and an exclusion reason. Duplicate policy precedes fitting
and FDR eligibility; it is never selected by observed correlation or significance.
The outcome-wide QC lists planned/valid/duplicate/test counts, predictor IDs and
rule versions. Report both corrections consistently; do not choose whichever
procedure yields a more favorable conclusion.

`domain_glandular` ordinal is the canonical glandular hypothesis. Salivary gland
swelling remains in all descriptive/result CSVs as an alternate representation,
with `alternative_to`, construct ID, evidence and FDR eligibility in the registry.
It remains in historical family BH for comparability, but is excluded from the
primary outcome-wide BH and primary forest plots. Redundancy QC requires at least
20 complete pairs and nonconstant variables before flagging high correlation;
small-pair correlations can be reported without establishing redundancy or
independence. Binary pairs use phi/contingency; ordinal/continuous pairs use
Spearman. Glandular activity concordance and the versioned construct policy are
reported separately. QC never automatically drops a predictor.

Sensitivities include total ESSDAI +5, domain ordinal versus binary, domain
baseline models `without_baseline_total`, at least 0.5 years between the first and last
subsequent trajectory assessments, continuous laboratories, and an upstream
anchor-day lab restriction when days-from-anchor exists. No new generic lab
recency window is imposed. Treatment adjustment is explicitly not implemented;
individual comorbidity analyses remain in Step 07. All interpretation describes
associations, not causal effects or a deployable prediction model.

For every ESSDAI domain, the required lagged sensitivity is
`domain_from_other_domains_adjusted`:

```text
Primary:     ESSDAI_TO ~ domain_d_FROM + ESSDAI_FROM + interval_years + selected demographic adjustments
Sensitivity: ESSDAI_TO ~ domain_d_FROM + ESSDAI_FROM_other_domains_d + interval_years + selected demographic adjustments
ESSDAI_FROM_other_domains_d = ESSDAI_FROM - weight[d] * ordinal_d_FROM
```

The domain remains an unweighted ordinal predictor. Weights come exclusively
from config.ESSDAI_DOMAIN_WEIGHTS; verified aliases are gland_swell→glandular
and neuro_periph→pns. Known, evaluable domain/total measurements and a
nonnegative difference are required. Complete panels must reconstruct the total;
mismatches/invalid subtractions become missing sensitivity values with QC, never
silent corrections. Partial panels permit only subtraction of the known target
contribution and are identified in QC. Source columns, weights and counts are
recorded. Numerical equality with the original state covariate is flagged as
nonindependent evidence. The outcome stays total subsequent ESSDAI; this does
not estimate propagation exclusively into other domains. The optional TO
other-domains outcome sensitivity is not implemented. The redundant lagged
`without_baseline_essdai` sensitivity is removed; the baseline omission is
renamed `without_baseline_total` and still omits only baseline total.

## Outputs and CLI

All outputs are isolated under `14_risk_factors_progression` in the standard
analytic, Block A tables/figures/QC, and logs directories. `--output-root` changes
the project root for those same relative paths. The three Parquet products are:

- `14_baseline_predictor_dataset.parquet`: one official baseline row per patient.
- `14_lagged_interval_dataset.parquet`: adjacent FROM/TO intervals and predictors.
- `14_domain_incidence_risk_set.parquet`: patient/target-domain first-event rows
  with wide baseline predictors and eligibility metadata.

All required result and QC tables are emitted, including unavailable and sparse
rows with support reasons. `claim_status`/`inference_tier` distinguish primary,
sensitivity, alternative_representation, unsupported and failed rows. Main forest
plots contain valid canonical primary representations, label the adjustment level
and retain caution labels; an
unsupported analysis receives a labeled empty plot. The secondary cross-domain
heatmap appears only with at least four valid pairs; unsupported cells are gray,
not zero effects. Metadata records gates, inputs, methods and software versions.
Metadata records versioned adjustment/composition/hypothesis/FDR rules, input
SHA-256 and paths (including context), Git commit/worktree state, UTC run time,
dependencies, supported trajectory counts and actual MixedLM/GEE use. QC adds
`14_adjustment_selection_qc.csv`, `14_domain_composition_qc.csv` and
`14_outcome_wide_fdr_qc.csv` without removing previous output columns.
The log records availability, feasibility, attempts, fallbacks, warnings and
output locations. Review structural/temporal QC before feasibility and models.

CLI overrides: `--integrated`, `--baseline`, `--context`, `--progression`,
`--output-root`, `--minimum-events`, `--overwrite`/`--no-overwrite`, `--dry-run`.
Outputs regenerate by default after backing up every existing dedicated file to
`outputs/backups/14_risk_factors_progression/<UTC timestamp>/` with its original
relative path. Backup failure aborts before replacement. `--no-overwrite` protects all existing dedicated
outputs before any write. Dry run validates datasets, builds risk sets and
feasibility, prints planned models, and writes only QC/log diagnostics (including
metadata and adjustment/composition QC). Structural
or temporal disagreements fail loudly, with available diagnostics saved.

Verification:

```bash
python -m pytest -q tests/test_risk_factors_progression.py
```
