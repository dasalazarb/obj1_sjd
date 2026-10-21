# Step 14: isolated inclusive ESSDAI-index extension

The [2026-10-21 composition provenance review](14_composition_provenance_review.md)
documents the source semantics, new QC, safe version gate and figure changes.

The 2026-10-20 inclusive specification supersedes the earlier official-only
policy **for this new extension**. Both Step 13 v2 index routes enter the primary
ESSDAI cohort when longitudinally eligible. Cohort size is reconstructed, never
forced to the investigator's reference counts. Official-only is a robustness
comparison. Inclusive v2 is the no-flags default; historical Step 14 remains
available with `--historical-main`, preserving its scientific functions,
thresholds, estimators, formulas, FDR and outputs.

**IMPLEMENTED / TESTED_SYNTHETIC; canonical execution BLOCKED.** Canonical Step
10/11, context/registry and Step 13 v2 clinical products were not present in the
implementation environment. No real-cohort N, event count, HR, beta, feasibility
or clinical validation is claimed. Institutional clinical/privacy review remains
pending even when software QC passes.

Implementation base: live `main` `dc56d303a2c9379e8b3193accb615fd9ae8f91ee`,
verified using connected GitHub, including its Git tree. Prechange scientific
source hashes and clean branch/status were captured in ignored scratch storage.
There were no historical clinical outputs to physically compare in this
workspace. Each future run records available physical inputs, old products,
branch, commit, working-tree status, scientific diff, code fingerprint and
package versions in its own prechange snapshot.

## One-command clinical run (new default)

From the repository root, with the existing canonical Step 10/11 and Step 13 v2 inputs:

```bash
python src/block_A/14_risk_factors_progression.py
```

No flags are required. This invokes the inclusive v2 extension and resolves
all seven companion input locations using the repository layout. Inputs are
still verified against the Step 13 v2 producer manifest, including SHA-256
hashes and QC; missing or incompatible files **block** the run instead of
silently changing the cohort. The script writes the
`14_explainwhy_inclusive_v2_*` output family under the normal Step 14
outputs directories.

For the original historical Step 14 models, use
`python src/block_A/14_risk_factors_progression.py --historical-main`.
The `--explainwhy-audit-only` and `--explainwhy-inclusive-main` entrypoints
remain available for reproducibility and custom explicit paths. The simple
command does **not** imply clinical or institutional review is complete.

## Execution

The no-flags command resolves canonical locations automatically. Explicit paths
remain available for controlled reproduction. The extension never discovers an
unversioned pattern file or reconstructs an index
as a substitute for missing products. The following paths illustrate the
repository layout; replace them with authorized input locations as needed:

```bash
python src/block_A/14_risk_factors_progression.py \
  --explainwhy-inclusive-main \
  --integrated data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet \
  --baseline data/analytic/blockA/11_integrated_baseline_characterization/11_integrated_baseline_patient_level.parquet \
  --context data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_context.parquet \
  --variable-registry data/analytic/10_build_integrated_longitudinal_dataset/10_variable_registry.csv \
  --patterns-v2 data/analytic/blockA/13_disease_activity_progression/13_essdai_patient_observed_patterns_v2.parquet \
  --selected-dates-v2 data/analytic/blockA/13_disease_activity_progression/13_essdai_pattern_selected_dates_v2.parquet \
  --eligibility-v2 data/analytic/blockA/13_disease_activity_progression/13_essdai_pattern_patient_eligibility_v2.parquet \
  --pattern-manifest-v2 outputs/qc/blockA/13_disease_activity_progression/13_essdai_observed_pattern_manifest_v2.json \
  --legacy-endpoints data/analytic/blockA/13_disease_activity_progression/13_essdai_patient_delta_endpoints.parquet \
  --output-root /authorized/research/run --no-overwrite
```

Use the same command with `--dry-run` first: validates hashes, exact canonical
reconstruction, the legacy bridge and temporal QC, with **no fits or writes**.
`--overwrite` makes a timestamped backup of this family only. The historical
script's overwrite default is retained; `--no-overwrite` is recommended for new
runs. `--explainwhy-audit-only` and `--explainwhy-inclusive-main` are mutually
exclusive. `--input-origin synthetic_validation` labels fixtures; it cannot
consume a canonical manifest, or certify canonical execution.

Primary laboratory values require same-date evidence. Optional
`--lab-max-age-days 30 90` adds **sensitivity** index-lab windows only when these
windows have been explicitly prespecified/approved by the investigator. The
example is not a clinical recommendation or automatic carry-forward rule.
No window is enabled by default. Lagged labs keep the strict FROM-date rule.

## Blocking inputs and index contract

Before any join, validate the physical Step 10/11 hashes against producer input
roles, the three physical v2 artifact hashes, definition version, thresholds,
selected-date policy, producer commit, origin and QC PASS. Current config/v2/v1/
Step 13 definition hashes must match the producer manifest. A declaration in a
manifest without the actual bytes does not pass.

Use the upstream v2 builder to independently reconcile **all columns and keys**
of eligibility, eligible patterns and selected dates against Step 10/11. This
is a validator of received products, not a copied indexing algorithm. The
clinical baseline episode/date and official Pop are never replaced. The all-
patient cohort preserves exclusions (zero/one evaluable ESSDAI date, unreliable
baseline date, etc.). No `.dropna()` on the official baseline ESSDAI is used to
select the new primary cohort.

Time zero for every new model is `essdai_index_date`; reference score is
`essdai_index_score`. First +3/+5 is reconstructed from strictly later selected
valid dated scores, independently of retrospective pattern labels. Censor at
the last evaluable ESSDAI date. Store both detection and the preceding known
non-event date, including index. Detection is not exact biological onset.
Missing-score episodes do not extend censoring. No pre-index interval is assumed
event free. An alternative index is not disease onset or diagnosis.

The original all-episode endpoint Parquet is exactly reconciled to the original
Step 13 endpoint builder. Every official-route event/date disagreement must
have its earliest all-episode crossing on a genuinely discordant date; an
unexplained disagreement blocks the bundle. Aggregate disagreement counts and
separate v1 all-episode sensitivity models preserve the definition difference.

## Predictors, model support and comparisons

Index predictors come from the exact selected index episode with registry
producer/temporal-scope evidence. Labs also require explicit measurement dates,
known compatible units, evaluability and the prespecified age window. There is
no automatic reuse of the historical baseline-predictor dataset for fallback.
Static sex and age come from Step 11; age is labeled **age at official clinical
baseline**, not age at alternative ESSDAI entry. Disease duration is used only
where an existing Step 11 duration and verified diagnosis date support the
same official index; fallback duration remains unavailable. Treatment remains
`not_implemented`, even if a field happens to be present. PRO versions/validity
and serology temporal scope use the stricter feasibility-audit contracts.

Models reuse Step 14's unchanged Full → Reduced A → Reduced B → Minimal support
hierarchy, minimum patients/events/intervals, rank checks, events per parameter,
Cox covariance/convergence diagnostics and predictor PH test. Numerical failure
never chooses a new adjustment. The new `baseline_essdai` **adapter field** is
the ESSDAI index score; model metadata explicitly names this change of clock.

Implemented families are first +3 Cox, +5 sensitivity Cox, post-index slope
MixedLM with the existing Gaussian GEE fallback, adjacent lagged Gaussian GEE,
and observed interval first-+3 Poisson GEE with offset/time bands/patient robust
clustering. Low→≥5 and Moderate→High risk starts from index state. Incident
individual/global domains start only from observed inactive index domains.
`domain_index_support.csv` reports unavailable fallback entry states as
`not_estimable_for_fallback`; those subjects remain in the descriptive cohort.
Cross-domain models require observed source/target states and the existing
stronger exposed-event threshold; unsupported pairs retain structured rows.

Shift the full Step 10 spine before filtering: missing intermediate visits
never create direct FROM→TO links. Positive-time intervals require both endpoints
to be selected v2 dates and evaluable for total-score association. First-event
risk stops on the first selected observed event independently of missing FROM
or predictor data; later intervals cannot rescue it. Predictor-specific
`event_not_modelable_due_to_missing_from_or_predictor` counts are exported.
The original temporal QC runs on its unchanged official-baseline estimand;
additional QC reconstructs the new index predictors, event risk sets and
adjacent intervals without weakening the original checks.

Official-only comparisons use the same selected-date outcomes, estimator,
index clock, inclusive standardization mean/SD and **frozen inclusive adjustment
level**. If that formula cannot be retained, report not estimable. This avoids
changing specification or predictor scale when removing alternative-index
subjects. Output tables separate cohort N, at-risk N, available predictor N,
complete-case N, fitted N and fallback attrition.

Domain-total composition is explicit. Leave-source-domain lagged sensitivities
reuse the original subtraction only on complete, arithmetic-concordant FROM/TO
panels with semantically certified compatible scoring versions. Equal source
strings do not certify this. They are not a +3 extra-
domain event. Recorded total ESSDAI is never reconstructed or replaced.

Each new family has its own BH-FDR; canonical primary hypotheses additionally
receive new outcome-wide BH. Official-only, +5, excluded-date, lab-window,
legacy-policy and composition analyses are labeled sensitivities. Historical
q-values are never consumed or replaced. No p-value controls model selection.

## Sensitivities and outputs

S1: matched-definition official-only; S2: +5 under the same v2 index; S3: upstream
whole-discordant-date exclusion (with fallback reselection and eligibility
recalculation); S4: 3+/4+ dates; S5: <2 / 2–<5 / ≥5 observed years; S6: official
v1 all-episodes bridge/models; S7: source-specific counts/delay distribution;
S8: strict labs plus explicitly configured age-window sensitivities. S4/S5 are
descriptive observation strata, not comparable incidence or causal estimates.

All products use `14_explainwhy_inclusive_v2_` under the existing Step 14 stem:

| Area | Products |
| --- | --- |
| Protected Parquet | `index_cohort`, `first_delta3_riskset`, `first_delta5_riskset`, `lagged_intervals`, `index_predictors` |
| Aggregate CSV | `cohort_flow`, `first_event_summary`, `index_source_comparison`, `predictor_availability`, `domain_index_support`, `model_feasibility`, `delta3_models`, `delta5_sensitivity_models`, `lagged_models`, `trajectory_models`, `index_state_domain_models`, `official_only_comparison`, `pattern_characterization`, `observation_sensitivities` |
| QC | `index_reconciliation_qc`, `legacy_bridge_qc`, `temporal_leakage_qc`, `attrition_qc`, `model_attempts_qc`, `fdr_qc`, `domain_composition_qc`, `marker_qc`, `prechange_snapshot.json`, `closure_summary.md`, `manifest.json` |
| Controlled PDFs | `essdai_spaghetti_index_markers`, `essdai_clinical_clock_entry`, `patient_change_scatter`, `first_event_comparison`, `delta3_forest`, `pattern_by_index_source` |

Individual spaghetti marks **only the index** of each alternative-index subject
with ×. Legends use run counts. Two explicit clocks show index-relative follow-
up and clinical-baseline entry delay. Scatter marks one point per person by
source; forest plots use cohort notes, not individual markers. Pattern bars
separate 2-date direction from 3+-date intermittence denominators. Marker QC
counts plotted artists, not a hardcoded cohort N. Missing model support yields
`not_estimable` figures and structured rows, never a zero effect.

All individual Parquets and PDFs remain controlled. Relative-time trajectories,
rare subgroups and unsuppressed aggregate cells require institutional disclosure
review; × is not anonymization. No clinical product is committed or uploaded to
GitHub. The writer stages after QC, hashes all artifacts, installs the manifest
last and rolls back installation exceptions. A blocked overwrite backs up the
old inclusive bundle and removes stale estimates, preserving historical outputs.
This is exception rollback, not filesystem-wide crash atomicity.

A canonical result is labeled `EXECUTED_CANONICAL` only with canonical inputs
and a clean identifiable commit; dirty runs remain conditional. Technical PASS
never grants institutional closure. Remaining limitations include delayed entry,
unequal follow-up/opportunity to observe changes, interval-censored onset,
regression to the mean, domain/biomarker composition and unmeasured treatment.

## Software validation record

The required regression suite plus the new inclusive tests passed: **336 tests**,
including **28 synthetic inclusive cases**. The latter cover fallback entry,
first-event bounds/censoring, same-date ties, strict labs, missing middle visits,
independent event-risk stopping, matched adjustment/scaling (including the v1
policy sensitivity), actual supported Cox inference, marker artists, blocked
inputs, no-overwrite, rollback, zero eligible patients and historical AST/bytes.
The full suite also exercises the unchanged GEE, MixedLM and Cox diagnostics.

Four stale expectations also failed on an archived, unmodified implementation
base. Tests now recognize the existing v2 companion figures and use the current
canonical articular weight/consistent synthetic panel total. No scientific
weight, historical estimator or original output was changed to satisfy them.
The former whole-file Step 14 equality assertion now permits isolated CLI
dispatch and verifies the scientific function AST; additional tests compare all
non-entrypoint module definitions and protected source bytes.

See `14_explainwhy_inclusive_v2_validation.json` for the test scope, source hashes
and explicit blocked canonical/review states. Deprecation warnings from the
existing scientific environment and empty upstream-figure layout warnings do
not establish clinical validation.
