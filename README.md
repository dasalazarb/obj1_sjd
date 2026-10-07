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

## Key folders

- `data/`: project data files
- `src/`: analysis and processing scripts
- `outputs/`: generated results

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

Script 02 runs the complete S3 primary and S2 unbalanced Mappers, plus S3
sensitivities with enough PCs to reach the configured variance target, without
urine squamous cells, and with hsCRP transformed by `log1p` when the original
observed values are nonnegative. All share the fixed post-redundancy feature set
(except the specified exclusion) and fixed Mapper hyperparameters. Full80 removes
only the PCA component cap; the primary retains its configured cap. Every scenario
uses PC1/PC2 as its lens and all retained PCs for local clustering. Family weights
are applied after RobustScaler, preserving the primary representation.

Before running temporal flow, review these tables and JSON under
`outputs/tables/studies/longitudinal_graph/`:

- `02_mapper_sensitivity_summary.csv`: five scenarios, PCA variance, support,
  coverage, and ARI with comparable visit and patient counts.
- `02_s2_s3_macrostate_stability.csv` and `02_s2_s3_stability_summary.json`:
  visit/patient overlap and ARI. Each primary macrostate matches independently by
  maximal patient Jaccard, then visit Jaccard; many-to-one matches expose mergers.
  Numeric macrostate IDs are not used as evidence of correspondence. Sets include
  all visits in macrostate nodes, including ties. Scalar Jaccard summaries are
  unweighted means over supported primary macrostates; review each macrostate,
  including the core, rather than relying on the mean.
- `02_full80_macrostate_stability.csv`,
  `02_no_urine_squamous_macrostate_stability.csv`, and
  `02_hscrp_log1p_macrostate_stability.csv`: per-macrostate sensitivity matches.
- `02_feature_distribution_diagnostics.csv`: original finite observations on the
  Script-01 cohort, before imputation; extremes are strictly below Q1 minus three
  IQRs or above Q3 plus three IQRs. Missing raw data or negative hsCRP observations
  are documented; ineligible sensitivities have `status: not_applicable` and no
  invented metrics.
- `02_mapper_pc_feature_diagnostics.csv`: hsCRP and urine squamous loadings, squared
  loading fractions, and loading ranks on PC1--PC4 in the fitted scenarios.
- `02_longitudinal_mapper_summary.json`: coverage reconciliation, administrative
  association flag, interval endpoint coverage, and representation-stability gate.

ARI excludes ambiguous or missing supported hard assignments. Its interpretation
requires enough common visits/patients and represented labels; high ARI on a small
subset or a single macrostate does not establish stability. No ARI cutoff or
automatic parameter tuning is used. Administrative associations below the flag
threshold do not establish absence of confounding.

The summary's Mapper coverage counts any graph node. In membership files,
`mapper_graph_covered` exposes that measure, while the existing `mapper_covered`
continues to mean membership in at least one supported macrostate for Script 03.
Graph-covered visits partition into hard assignments, ties, and covered visits
with neither a supported hard assignment nor a tie. Percentages use all input
visits or unique patients as denominators, on a 0--100 scale. Visits/patients in
unsupported macrostates can also belong to supported macrostates. Unsupported
macrostates remain descriptive and contribute no primary temporal weights.
The endpoint counts use only the original consecutive intervals, with positive
time and the configured maximum interval; missing endpoints never create skips.

The gate reports objective blockers and otherwise remains `requires_review`.
Proceed only after reviewing useful supported endpoint coverage, at least two
supported primary macrostates, recognition of the core across representations,
per-macrostate overlap, hsCRP/urine diagnostics, and administrative QC. Clinical
macrostate labels remain deferred to held-out characterization in Script 04.
Script 02 does not execute Script 03 or approve the scientific gate automatically.
After that review, run:

```bash
python src/studies/longitudinal_graph/03_analyze_temporal_flow.py
python src/studies/longitudinal_graph/04_characterize_transitions.py
```

Study datasets, tables, figures, QC, and logs are written under the standard
`studies/longitudinal_graph/` roots in `data/analytic/` and `outputs/`. Each
command also supports `--config`, `--integrated`, and `--dry-run`; dependent
stages expose arguments for their preceding-stage inputs.


## Section 5 comorbidity analysis

Section 5 now treats the project Codebook as the source of truth for variable semantics.
`rheumatological_comorbidities__` fields are summarized at baseline with mutually exclusive documented-status categories: confirmed/present, history only, documented with unspecified status, and not documented. Only confirmed/present records contribute to rheumatological prevalence summaries or non-causal progression associations.

`past_medical_history__` fields and `sjogren's_syndrome_history__` fields are summarized separately as documented historical information. They are not used as baseline rheumatological prevalence inputs, longitudinal comorbidity events, risk-set definitions, event dates, cumulative histories, or progression-model exposures.

The required Section 5 outputs are grouped by producing script under `outputs/tables/blockA/<script>`, `outputs/figures/blockA/<script>`, `outputs/qc/blockA/<script>`, and `outputs/logs/<script>`. Generated intermediate and analytic data follow the same `<script>` subdirectory convention under `data/`. Scripts whose names begin with `00_` retain their shared bootstrap paths. The legacy comorbidity event-rate outputs are intentionally not produced.
