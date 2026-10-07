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
Use `--overwrite` to replace existing Step 13 outputs; `--dry-run` validates
inputs and reports the domain contract without fitting the final analyses.

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

Run the four stages in order:

```bash
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py
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
