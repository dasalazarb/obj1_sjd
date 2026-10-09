# SjD Analysis Project

This repository supports a Sjögren's disease (SjD) analysis workflow.
It is meant to keep the project organized from raw data to final tables and figures.

## Project goal

Analyze patient data, prepare clean datasets, and generate reproducible results for reporting.

## Main workflow

1. Store original files in `data/raw/`.
2. Prepare analysis-ready data with the project scripts.
3. Save intermediate files in `data/intermediate/` when needed.
4. Save final datasets in `data/analytic/`.
5. Export tables, figures, and logs to `outputs/`.

Block A has a single directional characterization boundary: Step 10 publishes
the canonical `patient_id × clinical_episode_id` integrated dataset; Step 11
selects only `is_clinical_baseline == True` for the canonical patient-level
baseline, Table 1, availability, and baseline QC; Step 12 independently
describes longitudinal follow-up from the same Step 10 product. Downstream
analyses must consume the Step 11 baseline rather than reconstructing one.

Step 13 consumes the frozen integrated longitudinal clinical-episode dataset
and produces the canonical Block A disease-activity progression analyses. It
must not rebuild clinical episodes, baseline, ESSDAI, ESSPRI, or Pop states.

Step 13 requires all 12 ESSDAI ordinal domain columns, including
`essdai__glandular_ordinal_score`. Step 06 derives that score exclusively from
`essdai__gland_swell`, independently of the aggregate glandular phenotype;
Step 10 publishes it. Existing inputs without it must be regenerated in order:
Step 06, Step 10, then Step 13. Refresh Step 11 if its baseline schema must
include the new fields; Step 12 does not use the new glandular score.

Continuous ESSDAI and ESSPRI models select a valid random-intercept MixedLM or
fall back to patient-clustered Gaussian GEE when convergence, covariance,
boundary or Hessian checks fail. Both attempts are preserved in QC, and
ESSDAI sensitivities attempt random slopes and quadratic time when supported.
Domain inference uses patient-clustered OrdinalGEE, then active/inactive
binomial GEE if estimable; BH-FDR includes only valid domain models. Trajectory
figures show the selected model's marginal prediction and covariance-based
95% CI.

Descriptive windows use the official baseline and one episode per patient per
follow-up window, selected nearest its center (first eligible after five years).
ESSPRI uses all repeated patients as primary; a span of at least six months is
reported as sensitivity. KM describes first-observed events, reports event
proportions separately from person-time rates, and suppresses the median if
there are fewer than five events or fewer than five patients at risk when the
curve first crosses 0.5. These are explicit descriptive support rules.
Step 13 regenerates its outputs by default, so the standard command can be
rerun after updating upstream inputs. Use `--no-overwrite` to protect existing
outputs; `--dry-run` validates inputs and reports the domain contract without
fitting the final analyses.

Step 14 evaluates associations between earlier clinical, serological, glandular,
laboratory and ESSDAI-domain characteristics and subsequent progression. It uses
the official Step 11 baseline, adjacent Step 10 clinical intervals and Step 13
progression definitions. Baseline Cox/trajectory, lagged Gaussian GEE, incident
domain risk sets and support-gated secondary cross-domain models have separate
outputs, temporal QC and FDR families. See
[Step 14 implementation and review guide](docs/14_risk_factors_progression.md).
Run `python src/block_A/14_risk_factors_progression.py --dry-run` to validate
contracts and inspect feasibility before fitting the associations.

## Key folders

- `data/`: project data files
- `src/`: analysis and processing scripts
- `outputs/`: generated results
- `dashboard/`: local Streamlit explorer for published aggregate results

## Important note

Do not edit the original raw data. Keep all changes reproducible through scripts.

## Longitudinal Graph / Model A

Model A asks how observed patients move through a visit-level clinical landscape.
It consumes the canonical Step 10 `patient_id × clinical_episode_id` integrated
longitudinal dataset and the canonical consecutive-episode interval spine. Unlike
the baseline `studies/graph` analysis (one baseline observation per patient), it
first builds a time-blind reference Mapper from eligible clinical visits and only
then overlays actual consecutive visits. Mapper proximity is never treated as a
temporal transition.

Run preparation and the reference Mapper first:

```bash
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py
```

Script 02 preserves the S3 primary and the S2, full80, urine and valid hsCRP-log1p
sensitivities. Its explicit scenario registry additionally refits without hsCRP,
without PRO, with equal patient mass, with local DBSCAN epsilon, with a small
prespecified cover grid, and with a robust lens range. PRO-summary-only and
comparable-visit-feature alternatives require documented clinical whitelists;
empty/unavailable whitelists produce `not_applicable`, with a reason.

Each scenario records exact features, transformation, PCA and Mapper parameters,
status, support and coverage. PC1/PC2 remain the lens and all retained PCs are used
for clustering. Loading fractions use the sum of squared loadings **within a PC**;
they are not percentages of total variance. RobustScaler does not clip extremes.

Before temporal analysis, inspect `02_longitudinal_mapper_summary.json`, the
scenario/representation manifests, loading/family/scale/lens diagnostics, cover
occupancy/noise tables, coverage strata and original interval endpoint QC.
`02_mapper_admin_availability.csv` explicitly marks missing confounders, and
`02_mapper_admin_association.csv` adds patient-vector permutations to descriptive
visit-level V. Missing variables or absence of a flag do not prove no confounding.
01 now retains raw values and an observation mask; 02 contrasts original-pair PRO
correlations against the historical imputed-state pruning, and audits serology
eligibility without changing the eligible upstream feature set.

`02_mapper_bootstrap_replicates.csv` refits preprocessing, representation and the
Mapper after resampling complete patient series. Support counts original unique
patients even for duplicate draws. Matches use patient and then visit Jaccard on
common sampled patients; splits, merges and unmatched structures remain visible.
ARI includes comparable hard visits, patients and label counts; one label or a
small subset cannot establish stability. Components, modularity communities and
macrostates have separate identities. `branch_id` remains undefined until the
researcher documents the topology definition; the branch stability file explicitly
labels its current component QC. See the bootstrap summary for pending checks.

The membership contract is unchanged: `mapper_graph_covered` means any graph node,
while `mapper_covered` means a supported macrostate for Script 03. Weights use
`equal_weight_per_supported_node`, rather than a silently introduced temperature
rule. Exclusive visit/patient coverage categories reconcile to the input universe;
macrostate patient sets may overlap. Endpoint QC uses original consecutive
intervals and never connects across a missing intermediate visit.

The two reference PNGs now show all nodes with the same geometry: one colors
connected components, the other colors communities and labels support. Edges are
shared visits, not temporal transitions. Additional diagnostic figures cover lens
distributions, occupancy, PCA dominance, coverage and sensitivity comparability.
04 records whether each variable was actually used in the post-pruning primary
Mapper; a held-out variable still requires proxy-dependence review.

Both 01 and 02 support `--output-root` for isolated review bundles, and true
`--dry-run` validates/describes without fitting the Mapper or writing canonical
outputs. Missing clinical data returns `not_run_missing_data`. Successful 02
bundles archive prior `02_*` files outside canonical directories before replacement,
including obsolete sensitivity files. `02_run_manifest.json` records Git/script,
configuration/input hashes, seed, date, dependencies and check states.

The scientific gate never autoapproves flow. Missing branch evidence blocks branch
inference; other unavailable validations remain explicit. Review coverage,
comparability, patient stability, administration, observability and circularity,
and document the investigator's decision, date, SHA and justification before
using 03/04 for scientific inference. A technically completed run is not approval.
No all-in-one command runs those stages. After documented review, run manually:

```bash
python src/studies/longitudinal_graph/03_analyze_temporal_flow.py
python src/studies/longitudinal_graph/04_characterize_transitions.py
```

Study outputs remain under the standard `studies/longitudinal_graph/` roots.
[Implementation, validation and real-data limitations](docs/longitudinal_mapper_validation.md)
include isolated-run commands, historical provenance and the ten requested
scientific questions, explicitly `not_run_missing_data` in this review.


## Section 5 comorbidity analysis

Section 5 now treats the project Codebook as the source of truth for variable semantics.
`rheumatological_comorbidities__` fields are summarized at baseline with mutually exclusive documented-status categories: confirmed/present, history only, documented with unspecified status, and not documented. Only confirmed/present records contribute to rheumatological prevalence summaries or non-causal progression associations.

`past_medical_history__` fields and `sjogren's_syndrome_history__` fields are summarized separately as documented historical information. They are not used as baseline rheumatological prevalence inputs, longitudinal comorbidity events, risk-set definitions, event dates, cumulative histories, or progression-model exposures.

The required Section 5 outputs are grouped by producing script under `outputs/tables/blockA/<script>`, `outputs/figures/blockA/<script>`, `outputs/qc/blockA/<script>`, and `outputs/logs/<script>`. Generated intermediate and analytic data follow the same `<script>` subdirectory convention under `data/`. Scripts whose names begin with `00_` retain their shared bootstrap paths. The legacy comorbidity event-rate outputs are intentionally not produced.


## Local research dashboard

The Streamlit app lives in [dashboard/](dashboard/README.md). Install its separate
Python environment and run it on your PC using the Windows or macOS/Linux commands
in that README. With the dashboard environment active, run from the repository
root:

```bash
python dashboard/run_dashboard.py
```

Open http://localhost:8501. The app reads this checkout's registered aggregate
outputs and curated figures without executing analysis scripts. Generated results
are not included in GitHub; missing files are reported explicitly.

Repository-root `pytest` remains scoped to the analysis tests in `tests/`.
Run the dashboard's separate software suite from `dashboard/` using its environment.
