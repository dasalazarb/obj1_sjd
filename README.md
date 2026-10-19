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

Step 13 also reports observed patient-level absolute ESSDAI changes using both
±3 and ±5 prespecified descriptive/sensitivity thresholds. Separate ever-observed
and baseline-to-last endpoints, baseline Low/Moderate/High strata, patient
reconciliation/crosstabs, eligibility/integrity QC and four complementary PDF
figures preserve the existing category-transition analyses and Step 14 legacy
+5 sensitivity. See [absolute ESSDAI change definitions and outputs](docs/13_essdai_absolute_change.md)
for denominators, same-date selection and observation limitations.

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

The execution order stays **01 → 02 → 03 → 04**. Run only the numbered scripts:
01 prepares inputs, 02 builds Mapper and runs its QC, 03 analyzes temporal flow
after scientific approval, and 04 characterizes transitions. The `_review/`
directory contains internal modules imported by 02/03; it adds no execution steps.

Prepare inputs with Script 01; use Script 02's versioned review commands documented
below to consume the frozen reference and evaluate candidates:

```bash
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py --help
```

Historical S3/S2 names and results remain historical references. New review runs
use S4 names for an unchanged visit comparator, hsCRP transformation/winsorization
and ablation, equal patient mass, no PRO, approved PRO-summary/common feature
whitelists, observed-pair redundancy pruning and full80 PCA. Empty or unapproved
clinical whitelists produce `not_applicable`, with a reason. Legacy reproduction
retains its prior S3 sensitivity names in a separate create-only bundle.

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

Script 02 defaults to `candidate-qc`, reads a sealed historical S3-primary and
publishes S4 alternatives under `--output-root/runs/<run-id>`. A bare invocation
generates a unique UTC run ID and uses the only available seal or historical
archive. An archive is audited in place and sealed externally on its first full
run. If there are several references, select one with `--reference-run-id` or
`--reference-root`; it never chooses the latest silently. The seal lives outside
the archive. Existing runs and
historical outputs cannot be overwritten. Missing G0 reference evidence blocks
candidate execution. `--dry-run` does not fit, seal or write outputs. The original
algorithm can be reproduced under `--mode legacy-reproduction` in a new run.
Candidate policy is in `methodology_review.yaml`; historical `config.yaml` remains
unchanged. Run manifests record code/config/input/artifact hashes and seed.

The scientific gate never autoapproves flow. Missing branch evidence blocks branch
inference; other unavailable validations remain explicit. Review coverage,
comparability, patient stability, administration, observability and circularity,
and document the investigator's decision, date, SHA and justification before
using 03/04 for scientific inference. A technically completed run is not approval.
No all-in-one command runs those stages. Script 03 defaults to `reference-only`.
After documented scientific approval, its `approved-temporal` mode requires the
exact reviewed run manifest, scenario readiness and approval record before it
writes an isolated temporal run. An explicit `--allow-blocked-preview` writes a
separate bundle marked `EXPLORATORY_NOT_FOR_INFERENCE` and retains the official gate.

[Current contract and commands](docs/02_current_pipeline_contract.md) and
[validation evidence](docs/02_validation_test_report.md) document full/supported
graph exports, coverage, patient refits, administrative availability, circularity
and open scientific decisions. Clinical data and the historical archive were
unavailable in this workspace; published experiment aggregates are synthetic only.


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
