# Graph representation sensitivity: execution QC and interpretation

## Column audit follow-up (2026-10-16)

Graph 01 and Graph 02 now share `canonical_lab_name()` from Script 01. It
strips whitespace, lowercases the metadata key, and removes only the leading
`lab__` namespace; compound lab stems, original feature names and source columns
are preserved. Script 01 uses these keys both for the four characterization
extras (`protein_total`, `mcv`, `bun`, `lymphocyte_count`) and for warnings about
approved clinically expected labs excluded from the embedding.

The five S4 assignments proposed in the supplied column audit are implemented
in response to the user's request to apply that audit:

| LAB | Group | Subgroup |
| --- | --- | --- |
| `acute_care_panel__anion_gap_blood` | `METABOLIC_ENDOCRINE` | `ELECTROLYTES` |
| `urinalysis_includes_microscopic__hyalin_casts_urinalysis` | `RENAL_URINARY` | `URINALYSIS` |
| `urinalysis_includes_microscopic__renal_epithelial_cells_urinalysis` | `RENAL_URINARY` | `URINALYSIS` |
| `urinalysis_includes_microscopic__transitional_epith_urinalysis` | `RENAL_URINARY` | `URINALYSIS` |
| `urinalysis_includes_microscopic__unclassified_casts_urinalysis` | `RENAL_URINARY` | `URINALYSIS` |

The anion gap follows the existing electrolyte panel group; the four urinary
microscopy findings follow the existing urinalysis group. These are fixed
clinical metadata assignments, not numerical optimization or permission to
include a lab. Other unmapped schema columns are not assigned automatically.
The LAB selector, coverage thresholds, redundancy rules, PCA/Mapper settings,
bootstrap and clinical input data are unchanged.

`02_graph_feature_selection_audit.csv` records one row per manifest feature
and scenario, including rejected candidates and availability-only records for
uncatalogued LAB result columns. It separates presence in the integrated parquet,
source availability at baseline (including the separate Block A diagnosis-age
source), Pharma catalog membership and approval, observed baseline coverage,
encoding in the expanded manifest, scenario eligibility, coverage filtering,
selection after redundancy, S4 mapping status, and actual presence among the
PCA input columns. Mapping is reported independently of selection. S0 eligibility
uses its conventional manifest, including its existing categorical exceptions;
S1–S4 eligibility uses the expanded manifest. False downstream stage flags for
an excluded candidate do not imply that its data are absent. The file is saved
after each successful full-sample PCA fit. Existing audit outputs remain.

Regression validation uses synthetic fixtures, including namespaced and compound
labs, sparse mapped labs, rejected candidates, constant labs, redundancy and
uncatalogued source columns. The integrated clinical parquet and real Pharma /
Graph execution CSVs are absent from this checkout. Consequently, actual
missingness, units, text/value concordance, candidate coverage and authentic
diagnosis ages must still be checked on Biowulf. Before rerunning, record
`git rev-parse HEAD` and `git status --short`, verify that Step 10 and Pharma 00
outputs are from the same cycle, and archive the previous S4 mapping audit before
Graph 02 writes its replacement. Review the new selection audit alongside both
S4 audits; no clinical before/after results are inferred from synthetic tests.

Validation on 2026-10-16: the four focused Graph test modules passed all 61
tests; both scripts compiled and `git diff --check` passed. Additional Pharma
checks encountered existing issues in untouched files: `test_pharma.py` still
imports the absent `01_run_pharma.py`, and
`test_text_fallback_and_numeric_sufficiency_are_independent` expects `numeric`
where Pharma 00 currently returns `unknown` for one isolated numeric result.

Age at diagnosis stays missing without an authentic diagnosis-age/date source.
Adding ocular/salivary `ext__` variables, ESSDAI ordinal domains, or serology
anchors requires a separate methodological decision and manifest version with
circularity controls. Outcomes and Pop remain annotations.

## Earlier execution-QC corrections (2026-10-15)

The Spanish correction specification is preserved in
[graph_sensitivity_correction_spec_es.md](../../../docs/graph_sensitivity_correction_spec_es.md).
This change applies to `02_graph_representation_sensitivity.py`; it does not
change Block A, Pharma, Script 01, community detection, or the scientific
Mapper/PCA settings. Pop remains an external annotation.
The executed checks, synthetic example and clinical-data limitation are recorded
in [graph_sensitivity_qc_validation.md](../../../docs/graph_sensitivity_qc_validation.md).

## Fixed mapping before resampling

Before S0 starts, S4 mapping is checked against every LAB feature in
`full_values`, including features excluded by full-sample coverage or redundancy
that might be selected in a subsample. Only the leading `lab__` namespace is
normalized; compound stems remain intact. A missing, pending, or non-`MAPPED`
entry aborts before any bootstrap. The mapping audit is written even on failure,
with exact labs, features, eligibility, mapping status, and validation result.
`included_S4` is populated once the full-sample S4 selection is known. Features
are not added to S4 merely because they are in the map.

The two newly reported mappings are explicit metadata decisions:

| LAB | Group | Subgroup | Decision |
| --- | --- | --- | --- |
| `angiotensin_converting_enzyme` | `GRANULOMATOUS_DISEASE` | `ACE` | Group explicitly approved by the investigator in the 2026-10-15 conversation; the subgroup identifies the analyte. ACE is a differential-diagnostic biomarker here, not a diagnosis or optimized numerical grouping. |
| `hepatitis_b_surface_antibody` | `INFECTIOUS_SEROLOGY` | `HEPATITIS` | Hepatitis B serology, consistent with the existing hepatitis antibody group and the supplied specification. |

The fixed map is the inspectable table of group assignments. Clinical coverage
of a particular run is established by its preflight audit, not by the number of
rows in the map or by the three-lab synthetic fixture.

## MFA blocks with no sample variation

Imputation and scaling are refitted on each patient subsample. A LAB subgroup
whose scaled first singular value is numerically zero retains all columns,
positions, and dimensions, but contributes a zero block with effective weight
zero. Its status is `zero_variance`; it contributes no signal to the outer LAB
normalization. Active subgroup blocks are normalized by their positive first
singular value, combined, and normalized at LAB-family level. There is no
`1 / epsilon` substitution. Returned transformed columns preserve raw order.

The zero test uses `machine_epsilon * max(block.shape) * max(abs(block))`, with
no absolute floor that could erase small real variation. Non-finite values or
non-finite inverse weights are invalid. If all LAB groups are zero, or a present
family lacks an estimable representation, a bootstrap attempt is
`not_estimable`; the same defect in the full-sample fit aborts the scenario.
The audit records subgroup/family status, sigma1, sample size, nominal and
effective weights, and active group counts. For a LAB subgroup the effective
weight includes the outer LAB-family normalization.

## Bootstrap denominators and operational QC

Sampling remains patient subsampling without replacement, fraction 0.80. Every
attempt refits feature selection, imputation, scaling/MFA, PCA, Mapper and
communities. The configured budget remains 60; attempts are never replenished.
Seeds, sample positions/hashes, stages, causes, and counts are recorded per
attempt. Unexpected schema/programming errors such as `KeyError` propagate;
they are not silently converted into non-matches. Mapping errors are blocking
input-contract failures, not biological instability.

For a reference community, a successful match means best Jaccard >= 0.50
between its **Mapper-node patient union restricted to sampled patients** and
a refitted candidate node union. With M matched successful attempts, S successful
attempts, and B requested attempts:

* `stability_conditional = M / S` (NA when S=0).
* `stability_conservative = M / B` (diagnostic, not a direct topological estimate).
* `bootstrap_stability` is the conditional estimator **only after QC passes**;
  otherwise it is NA.

The new operational gate is S/B >= 0.95. This is an execution-QC policy, not a
biological validity cutoff and not a claim about an earlier preregistration.
With no successful attempts the status is `INVALID_BOOTSTRAP`; below 95% it is
`BOOTSTRAP_QC_FAIL`. In both cases stable community count is NA, not zero, and
no community is called stable or unstable from that bootstrap. After QC passes,
the unchanged stability cutoff 0.80 applies to the conditional estimate.
Conditional estimates may still be biased by composition-dependent failures;
both estimates and all failures remain visible even when QC passes.

| Synthetic denominator example | Conditional | Conservative | QC | Stable declaration |
| --- | ---: | ---: | --- | --- |
| 60 requested, 27 successes, 27 matches | 1.00 | 0.45 | FAIL | Not evaluable |
| 60 requested, 60 successes, 50 matches | 0.8333 | 0.8333 | PASS | Meets unchanged 0.80 threshold |
| 60 requested, 0 successes | NA | 0 | INVALID | Not evaluable |

These are regression examples, **not audited counts from the 159-patient run**.

## Three separate interpretation questions

`02_graph_interpretation_qc.json` separates computational QC, core evidence,
and global partition robustness. A stable community in two scenarios does not
demonstrate reproducible multiple branches. S1 changes coverage, S2 adds
redundancy filtering, S3 changes family weighting, and S4 adds hierarchical LAB
weighting. S3/S4 raw frames, column order and missingness are asserted equal.

Existing core Jaccard uses **unambiguous hard-assigned patient sets**, not
overlapping Mapper-node unions. A second explicitly named node-union table is
now emitted; neither definition replaces the primary bootstrap object.
Community numbers are not treated as semantic labels across scenarios.

Pairwise ARI retains the original supported, covered, non-tied intersection
for each pair and adds n/total and coverage fraction. The additional common-
intersection ARI uses exactly the same eligible patients in all five scenarios.
Its operational sample floor is at least 20 patients and at least 50% of the
cohort, to avoid interpreting a tiny selected subset. Single-cluster partitions
are `not_estimable` for this multi-branch comparison. These are descriptive
sample-quality rules, not similarity thresholds. ARI near zero is approximately
chance-level concordance; negative ARI is possible. A high pairwise ARI does not
prove reproducibility for all patients or all representations.

No scientific Jaccard/ARI claim threshold is inferred from the known results.
The defaults are null, so `CORE_EVIDENCE` and `GLOBAL_PARTITION_ROBUSTNESS` remain
`NOT_EVALUABLE` as claims, while the pair metrics, denominators, coverage and
stable counts are reported. If investigators prospectively specify thresholds
for a future analysis, a nonempty `threshold_justification` is mandatory.
Scientific interpretation requires all five bootstrap QC gates to pass, all six
cleaned-scenario pairs to be evaluable, and (for global support) at least two
stable communities in every cleaned scenario. No output says `YES` or
`CONFIRMED` solely from connectivity or one stable core.

## Output additions and maintenance

All prior CSVs and three PNGs remain. Additional outputs are:

* `02_graph_bootstrap_replicate_audit.csv` — one row per attempt.
* `02_graph_bootstrap_qc_summary.csv` — requested, attempted, valid, failed,
  success fraction, gate, and failures by cause for every scenario.
* `02_graph_bootstrap_balance_audit.csv` — per-attempt MFA block diagnostics.
* `02_graph_common_intersection_ari.csv` — one fixed patient intersection.
* `02_graph_stable_core_node_union_overlap.csv` — explicitly separate object.
* `02_graph_interpretation_qc.json` — execution and interpretation status.

The mapping, community stability, scenario summary and S4 full-fit weight audits
are extended without removing their old columns. Completed-scenario bootstrap
audits are written as the run progresses. Redundancy audit concatenation uses an
explicit schema and omits empty frames without dropping populated audit rows.

Diagnosis age remains missing when no actual `age_dx` / `age_at_diagnosis` is
available from Block A; visit or baseline age is never substituted. The summary
reports observed diagnosis-age count. No patient ages/dates are invented.

The pandas `numexpr` warning is environment maintenance, not a cause of S4
mapping or bootstrap errors. In the Python 3.10 environment that actually runs
pandas, inspect dependency compatibility first, for example:

```bash
/usr/local/Anaconda/envs/py3.10/bin/python -m pip show pandas numpy numexpr
/usr/local/Anaconda/envs/py3.10/bin/python -m pip install --dry-run 'numexpr>=2.8.4'
```

An environment owner must approve any actual package update. This code does not
modify the NIH environment, user packages, or pandas warning filters.
# Contrato Pharma 00 v2 y preflight

Graph 01 y 02 ahora requieren el contrato `pharma_lab_profile_v2` y una liberación
clínica externa para ajustar modelos. Empiece con `--preflight-only` en ambos
scripts. La plantilla `graph_lab_release.yaml` está pendiente y no autoriza análisis.
Las políticas `legacy_with_overlap` y `without_overlap` generan outputs separados
por política y corrida; Graph 02 acepta `--canonical-config` explícito.
Consulte [la guía del refactor](../../../docs/graph_lab_contract_refactor.md) para
comandos, hashes, pruebas, linaje y bloqueos conocidos del mapa clínico.
