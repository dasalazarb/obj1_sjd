# Graph03 Phase C: clinical endpoint documentary review

Phase C creates a separate, restricted audit of six endpoint families. It never
runs Graph02, Phase B, or longitudinal models, and never promotes a source to
`N3_REVIEWED`. Passing audit QC means that the audit is internally consistent;
it does not establish endpoint independence, clinical validity, or model readiness.

## Explicit input and execution contract

The default template pins Graph03 `20261104T180702279234Z_961ffc6f`, input
implementation commit `585876b3c27355d5ab3e462c783026d5d7b18b9d`, and frozen
Graph02 `20261103T141918515545Z_fd43fde0`, whose manifest SHA-256 is
`02151f4cfec041cd30ddfea1a10d1df175f3d47d06f8a922335eb40be3d9a6fc`.
These are provenance expectations, not approvals. A different run requires an
explicitly revised review YAML. No latest pointer is read or written.

Run from the repository root with the real, authorized local paths:

```bash
python src/studies/graph/03c_graph_n3_clinical_adjudication.py \
  --graph03-run /authorized/graph03/runs/20261104T180702279234Z_961ffc6f \
  --graph02-release /authorized/graph02/runs/20261103T141918515545Z_fd43fde0/02_graph_release_manifest.json \
  --config src/studies/graph/configs/n3_phasec_review.yaml \
  --output-dir /authorized/phasec/new_audit_directory
```

The output directory must not exist and must be separate from both input runs.
The CLI requires the ten aggregate Phase A inputs listed in the implementation,
the original frozen Graph02 manifest and its checksum-verified artifacts. A ZIP
of aggregates alone cannot establish patient grain, protocol availability,
N1 inclusion, or actual baseline-category counts. Do not extract or reconstruct
patient records from an aggregate ZIP. No patient data is required in Git.

Before creating outputs, the reader verifies Phase A completion, schema, commit,
run identity, provenance agreement, all Phase A hashes in the Graph03 summary,
Graph02 manifest/artifact hashes, unique keys, frozen patient membership and
support/stability denominators. For the pinned reference only, it reconciles
2,525 inventory rows (505 sources × five representations), 420 candidate pairs,
88 distinct candidate sources, and the reported R0/R2 endpoint coverage. Future
runs are not required to retain these counts. Any mismatch raises an error before
creating the output directory; do not interpret a divergent release as comparable.

The command does not import a pipeline runner or call an estimator. Its only
reads of Stage02 are existing contract/reader utilities, frozen feature inclusion
and documentary source code; nothing in Stage02 or Block A is modified.

## Family review and default disposition

Coverage below is the reference plan's expected aggregate reconciliation, not an
execution result or independent clinical approval. Each pair is observed / hard
supported; the clinical status remains pending for R0 and R2 until evidence is
reviewed. Actual code-detected dependence takes precedence and blocks eligibility.

| Family and exact variants | R0 | R2 | Documentary source | Main unresolved review |
|---|---:|---:|---|---|
| Systemic activity: `essdai__total` | 86/92 | 84/87 | `01_essdai_esspri.py:derive_essdai_total` | Upstream total, ESSDAI domains/overlap N1 proxies, baseline timing; Pop prohibited |
| Symptoms: `esspri__total` | 66/92 | 67/87 | `01_essdai_esspri.py:derive_esspri_total` | Complete pain/fatigue/dryness triples, SF36/PROFAD/MDAFS construct overlap; Pop prohibited |
| Histopathology: `ext__biopsy_focus_score` | 58/92 | 55/87 | `01_extended_clinical_phenotype.py:derive_pathology` | `biopsy_pathology__f_score`, discordance masking, selection by indication, units and procedure/report date |
| ANA: `lab__ana_status__text`; `lab__ana_hep2_status__text` | 63/92; 42/92 | 66/87; 45/87 | `01_serological_profile.py:select_episode_analytes`, `build_wide` | Original EDA20/20b assay mapping, technique/category equivalence, date and N1 serology proxy |
| RF: `lab__rheumatoid_factor__text`; `lab__rheumatoid_factor__value` | 46/92; 44/92 | 48/87; 37/87 | Same laboratory selection and wide mapping | Same sample vs separate assays; units, censoring and thresholds; no merging or summation |
| Qualitative proteinuria: `lab__urine_protein_qualitative__text` | 88/92 | 82/87 | Same laboratory selection and wide mapping | Negative/trace/unknown/not-tested categories, repeated samples, contamination; not SjD renal attribution |

ESSDAI total copies the upstream canonical total with range QC; ESSPRI is the
mean of three observed, range-checked components. The laboratory layer selects
among already matched episode measurements by minimum absolute anchor distance,
prioritizes pre-anchor evidence in a distance tie, and handles discordance and
unit conflicts. This rule can select a post-anchor measurement; it is not evidence
that an endpoint was known at baseline. `build_wide` carries selected laboratory
measurement dates separately from clinical episode anchors. Actual assay-specific
source mapping and measurement windows require authorized evidence beyond code.

For every representation/source, the reviewer must document the source table,
original field, patient × episode/assay/instrument grain, transformations, baseline
window, real instrument-date semantics, possible future derivation, units or
categories, indication/selection bias, technical dependency paths and conceptual
N1 proxies. Code references carry SHA-256 checksums. Shared domains and absence
from `feature_sets` never prove clinical independence. R1/R3/R4 remain traceable
without changing their roles.

`PENDING_DOCUMENTARY_EVIDENCE`, `BLOCKED_RELATED_OR_PROXY`, and
`BLOCKED_UNAVAILABLE_OR_INVALID` are distinct dispositions. Evidence statuses
separate `NO_EVIDENCE`, `SOURCE_UNAVAILABLE`, `PROXY_REVIEW_PENDING`, and
`NOT_ELIGIBLE`. A file alone leaves clinical status `PENDING_REVIEW`.

A `candidate_reviews` entry is specific to representation and exact source.
Its `source_reference` must point inside repository `src/`, `docs/`, or `tests/`,
and `source_sha256` must match that document. `status: DOCUMENTED` plus structured
baseline/date/unit-or-category/independence evidence can produce
`HUMAN_REVIEW_REQUIRED` / `ELIGIBLE_FOR_HUMAN_REVIEW`. This requests an expert
clinical decision; it never creates final approval. The template records the
required fields and starts empty. `REJECTED` and documented future derivation
block eligibility. Neither names, coverage nor association results select assays.

## Administrative, era and Pop reviews

Protocol auditing distinguishes an absent canonical field, an all-null field,
available aliases with unreviewed equivalence, documented non-equivalence and
an available verified canonical mapping. It reports aggregate alias coverage and
canonical cardinality without copying category values. No alias is joined,
filled, or substituted. If the frozen adjustment is unavailable, propose a
protocol amendment instead of silently rebuilding Graph02 or dropping it.

Optional `protocol_reviews` and `baseline_era_reviews` each allow one record per
representation. To observe an existing human signoff, the record must carry
`status: DOCUMENTED`, `human_signoff: true`, a nonempty `reviewer`, a local
`source_reference`, and its matching `source_sha256`. The reviewer must supply
that evidence; Phase C never creates it. Reviewer identities/free text are not
exported. A protocol record also requires `canonical_source` and
`equivalent_to_prespecified_confounder: true` (or `false` for a documented
non-equivalent alias). A verified mapping records availability only and does not
repair the model inputs.

Era review also requires `date_semantics: official_baseline_anchor`,
`timezone_review_status: DOCUMENTED`, `lookahead_review_status: DOCUMENTED`, and
`official_baseline_contract_review_status: DOCUMENTED`,
valid frozen anchors and no observed official-baseline mismatch. Only then can
the audit observe `BASELINE_ERA_VERIFIED`. The separate endpoint date status stays
`PENDING_REVIEW`, and longitudinal stays `BLOCKED`. Date parsing for diagnostics
uses UTC without asserting that the original timezone is verified. No exact dates
or era category values are exported. Missing official-baseline date evidence and
temporal order are still explicit documentary limitations of the frozen inputs.

Pop auditing reports exact counts for Pop1/2/3, Unclassifiable, Unclassified,
missing and other-invalid, separately for baseline, Mapper-covered and supported
hard assignments. Unknown category strings are never serialized. It records the
Step11/observed-patterns `Unclassifiable` versus Graph03 `Unclassified` contract
mismatch without treating them as equivalent or recoding anything. Comparison
against raw Step11 values is explicitly unavailable; do not claim the previous
35 unrecognized values have been identified without that comparison.

## Feasibility and estimand

The implementation reads the existing `validation_03.yaml` thresholds and
confounders without changing them. Actual contracts in `_pop_status`,
`_source_basis` and `n3_incremental_models` define:

| Model | Existing design |
|---|---|
| A | Prespecified clinical covariates plus reviewed baseline era; eligible, specifically reviewed Pop only when requested |
| B | A plus frozen community |
| C | A plus a prespecified, verified reduced N1 basis |
| D | C plus frozen community only when explicitly requested |

Admitted contrasts are B vs A, C vs A and optional D vs C, conditional on frozen
memberships. Pop is prohibited for ESSDAI, ESSPRI, `eg_*`, `ovl__status` and their
documented derivatives. All planned comparisons require a single complete-case
population for the union of required outcome, adjustment, Pop and N1 terms.
`n_Y_nonmissing_with_all_covariates` from Phase A is not that final union; Phase C
leaves `n_common_complete_case` blank and marks it unverified. Zero covariate
complete cases does not mean Y was never measured.

Readiness rows retain observed/hard/stable/provisional denominators and separate
documentary, adjustment, support, topology, design, outcome-class, event and
multiplicity gates. No coefficients, effect intervals or p-values are produced.
Residual df/collinearity and binary events per parameter remain unassessed until
independent statistical review. Source-basis selection cannot use Y associations.
R0's reference 62 stable / 30 provisional and R2's 0 stable / 87 supported retain
the topology block. Approval of documents alone cannot remove it; a stable-only
R0 sensitivity requires a separate prespecified amendment.

## Outputs, QC and disclosure

The new output directory contains exactly these eleven artifacts, each prefixed
`03_graph_n3_phasec_`:

- `candidate_adjudication.csv`, `source_lineage.csv`, `measurement_review.csv`
- `covariate_audit.csv`, `baseline_era_audit.csv`, `pop_vocabulary_audit.csv`
- `model_readiness.csv`, `human_review_queue.csv`
- `qc.json`, `summary.md`, `artifact_inventory.csv`

CSV outputs include disposition, evidence status, reviewer action, blockers,
representation/source, input Graph03 run, frozen Graph02 SHA and privacy class.
Run ID, implementation commit, configuration hash and input/documentary-source
hashes establish provenance. The artifact inventory hashes the other ten files;
it deliberately does not hash itself. Outputs are `INTERNAL_RESTRICTED`, never
automatically shareable. Counts 1–4 trigger complementary suppression of all
patient counts on that row; category strings are restricted to the Pop allowlist.
IDs, episode keys, exact clinical dates and free clinical/reviewer text are not
exported. Suppression is conservative and does not replace institutional review.
The audit never packs, publishes or pushes these restricted runtime artifacts.

After institutional approval of the selected aggregates only, package the explicit
allowlist below, replacing `phasec_dir` with the reviewed directory and placing the
archive outside it. This command is documentation, not an automatically run step:

```bash
phasec_dir=/authorized/phasec/reviewed_audit_directory
zip -j /authorized/phasec/Graph03_PhaseC_reviewed_aggregates.zip \
  "$phasec_dir"/03_graph_n3_phasec_candidate_adjudication.csv \
  "$phasec_dir"/03_graph_n3_phasec_source_lineage.csv \
  "$phasec_dir"/03_graph_n3_phasec_measurement_review.csv \
  "$phasec_dir"/03_graph_n3_phasec_covariate_audit.csv \
  "$phasec_dir"/03_graph_n3_phasec_baseline_era_audit.csv \
  "$phasec_dir"/03_graph_n3_phasec_pop_vocabulary_audit.csv \
  "$phasec_dir"/03_graph_n3_phasec_model_readiness.csv \
  "$phasec_dir"/03_graph_n3_phasec_human_review_queue.csv \
  "$phasec_dir"/03_graph_n3_phasec_qc.json \
  "$phasec_dir"/03_graph_n3_phasec_summary.md \
  "$phasec_dir"/03_graph_n3_phasec_artifact_inventory.csv
```

Synthetic acceptance and existing regression tests:

```bash
python -m pytest -q \
  tests/test_graph03_n3_candidate_audit.py \
  tests/test_graph03_n3_incremental.py \
  tests/test_graph_characterization_tiers.py \
  tests/test_graph03_n3_phasec_adjudication.py
```

Before changing existing production contracts, obtain separate human signoffs for
`characterization.independence_reviews`, `n3_validation.measurement_policies`,
`baseline_era_reviews`, `pop_comparator_reviews` and `source_comparators`, and an
independent statistical/disclosure review. Phase C never edits them or
`temporal_resolution.endpoint_policies`.

Mapper communities describe exploratory multidimensional heterogeneity in SjD.
Independent incremental clinical utility remains unproven while measurement,
conceptual independence and stability are unresolved. This audit supports no
external-validation, prognostic, causal or latent-progression claim.
