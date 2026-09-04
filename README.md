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

## Key folders

- `data/`: project data files
- `src/`: analysis and processing scripts
- `outputs/`: generated results

## Important note

Do not edit the original raw data. Keep all changes reproducible through scripts.


## Section 5 comorbidity analysis

Section 5 now treats the project Codebook as the source of truth for variable semantics.
`rheumatological_comorbidities__` fields are summarized at baseline with mutually exclusive documented-status categories: confirmed/present, history only, documented with unspecified status, and not documented. Only confirmed/present records contribute to rheumatological prevalence summaries or non-causal progression associations.

`past_medical_history__` fields and `sjogren's_syndrome_history__` fields are summarized separately as documented historical information. They are not used as baseline rheumatological prevalence inputs, longitudinal comorbidity events, risk-set definitions, event dates, cumulative histories, or progression-model exposures.

The required Section 5 outputs are grouped by producing script under `outputs/tables/blockA/<script>`, `outputs/figures/blockA/<script>`, `outputs/qc/blockA/<script>`, and `outputs/logs/<script>`. Generated intermediate and analytic data follow the same `<script>` subdirectory convention under `data/`. Scripts whose names begin with `00_` retain their shared bootstrap paths. The legacy comorbidity event-rate outputs are intentionally not produced.
