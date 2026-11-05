# Graph03 Phase D: frozen resolution preflight

Phase D adds an auditable evidence package after Phase C. Integrity QC may PASS
while scientific status remains `COMPLETE_WITH_BLOCKERS`. It never estimates a
model, trains/projects Mapper, applies a crosswalk, imputes, drops a confounder,
alters a frozen artifact or edits either existing Graph03 configuration.

The historical A/B/C/D records are preparation rows, not fitted models. Forty
representation/source candidates produce 160 preparation rows when the explicit
release contains R0–R4. Missing candidate inventory keys are reported in QC;
there is still an explicit unavailable-candidate record, not a patient record.

## Implementation map

| File | Responsibility |
|---|---|
| `src/studies/graph/03d_graph_n3_resolution_preflight.py` | Explicit-input CLI, always preflight only |
| `src/studies/graph/stage03/s03_n3_resolution.py` | Frozen reader, evidence reconciliation, D1–D5, artifacts and disclosure |
| `src/studies/graph/stage03/s03_n3_resolution_support.py` | D6 complete-case union, design rank/df, events and separation feasibility |
| `src/studies/graph/configs/n3_phase_d_resolution.yaml` | Separate v1 pins, prohibitions and scoped documentary proposals |
| `tests/test_graph03_n3_resolution.py` | Synthetic positive, negative, integrity, privacy and regression contracts |

The implementation reuses the Phase C frozen-input reader, `safe_artifact`,
`hard_members`, typed dependency traversal and observed-value missingness rules.
It implements the existing Pop comparator guard without importing the module
that contains model estimators; a parity test checks ESSDAI/ESSPRI derivatives.

## Provenance and production differences

The initial local checkout was `583bac20fa6d3ff4d7dbed29412e807e6e35a363`.
The verified production/main revision was
`d770387b4ee6356c036c78a38dd22a758adcf416`; its additional changes concern Step17,
not Graph03. The work branch is based on that production revision. Phase A's
separate implementation pin remains
`585876b3c27355d5ab3e462c783026d5d7b18b9d`. Production execution must check its own
exact inputs rather than assume a branch name establishes provenance.

Three trust anchors were not supplied in the modification plan and cannot be
invented. Independently verify and fill these in a restricted copy of the new
config before production use:

- `graph03_summary_sha256`: `03_graph_results_summary.json` in the explicit Phase A run.
- `phasec_qc_sha256`: `03_graph_n3_phasec_qc.json` in the explicit Phase C run.
- `phasec_inventory_sha256`: `03_graph_n3_phasec_artifact_inventory.csv` in that run.

The defaults are null and fail closed. The command does not compute-and-trust
missing anchors. An operator may inspect hashes with `sha256sum` in the protected
site and reconcile them against independently approved release evidence before
recording pins. No real release, review ZIP, Step11 table or original clinical
form was available for the synthetic software checks.

Phase C is identified by its pinned QC run_id, not its directory name. QC must
be PASS with documentary blockers, no approvals, no estimation and unchanged
inputs. Every inventoried Phase C file, all Phase A summary hashes and all
Graph02 manifest hashes are verified. Candidate and A/B/C/D keys and provenance
must reconcile. The source manifest inventories all three entire frozen trees.
Symlinks inside frozen trees and traversal/escaping paths are rejected.

`code_drift: report_and_block` records current versus Phase C documentary hashes
and blocks review readiness on changed files; `reject` aborts. Newly added
Phase D modules are distinguished from changed historical code. Current code
hashes and the implementation commit are recorded separately. Code and review
documents are checked for changes during execution, including error paths.

## Running at the protected site

```bash
BASE=/data/salazarda/data/obj1_sjd
PY=/usr/local/Anaconda/envs/py3.10/bin/python
GRAPH03_RUN="$BASE/outputs/tables/studies/graph/03_graph_community_characterization/runs/20261104T180702279234Z_961ffc6f"
GRAPH02_RELEASE="$BASE/outputs/tables/studies/graph/02_graph_representation_sensitivity/runs/20261103T141918515545Z_fd43fde0/02_graph_release_manifest.json"
PHASEC_RUN=/operator/verified/phasec/run
PHASED_CONFIG=/protected/reviewed/n3_phase_d_resolution.yaml
OUTPUT_DIR="$BASE/outputs/tables/studies/graph/03d_graph_n3_resolution_preflight/operator_unique_run"

"$PY" "$BASE/src/studies/graph/03d_graph_n3_resolution_preflight.py" \
  --graph03-run "$GRAPH03_RUN" \
  --graph02-release "$GRAPH02_RELEASE" \
  --phasec-run "$PHASEC_RUN" \
  --config "$PHASED_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --preflight-only
```

Supply the actual verified Phase C path and a new output directory. No latest-run
selection exists. `--preflight-only` documents intent; omitting it still cannot
enable estimation. The output path cannot exist, contain a frozen tree, or be
inside one (including through a symlink). Each package receives its own run_id,
mode 0700 directory and mode 0600 files. Incomplete packages are removed on
failure; frozen checks run in `finally` on both success and failure.

## Documentary signoffs

Each config section is scoped to representation_id. Measurement, independence
and statistical-design decisions also require the exact source_column. Duplicate,
unscoped and final `APPROVED` records are rejected. Empty reviews stay pending.

Use a separately reviewed YAML/JSON decision document under `docs/`, `src/` or
`tests/` in the protected repository. The document must equal the config record
apart from source_reference/source_sha256, plus its review_section. For example,
this **illustrative pending record is not a signoff**:

```yaml
review_section: protocol
representation_id: R0
status: PENDING_REVIEW
review_id: institution_assigned_review_id
reviewer: null
human_signoff: false
original_source: ids__protocol
canonical_target: demo__protocol
cohort_scope: R0
meaning: research_protocol
code_dictionary_review: PENDING_REVIEW
multiple_protocol_rule_review: PENDING_REVIEW
equivalent_to_prespecified_confounder: false
```

After human review, the config record contains those same fields except
review_section, and the document's repository-relative source_reference and
verified source_sha256. `DOCUMENTED` requires human_signoff true, nonempty
reviewer/review_id, the matching content and scoped review. `REJECTED` is observed
without promoting eligibility. A modified document, mismatched signoff or file
existence alone cannot establish evidence. Cryptographic file consistency does
not authenticate reviewer identity or competence; institutional verification
remains required. Reviewer names and arbitrary review prose are not exported.

Measurement records require `date_semantics: instrument_measurement`,
`instrument_measurement_date_verified: true`, `derived_from_future: false`,
`timezone_review: DOCUMENTED`, `baseline_temporal_order_review: DOCUMENTED`, plus
nonempty baseline_definition, lookback_rule, measurement_source_reference and
repeated_measurement_rule. Unit/category reviews are separately required.
An episode anchor or an era review cannot substitute for these fields.

Official-baseline and baseline-era decisions are separate sections. Each needs
a verified date_column, official_baseline_definition_review, timezone_review
and lookahead_review. Era additionally needs `era_rule: fixed_five_year_strata`.
No era is synthesized from an unreviewed date. Protocol aliases never enter
the current design; a verified equivalence only proposes a new derived release.

Independence records need separate documented clinical, origin, proxy and
discovery-exclusion reviews. Code dependencies remain blocking independently.
Statistical records require reviewed covariate categories, hypothesis family,
multiplicity and uncertainty; reduced N1 columns must be unique, prespecified
and defining in that representation. Records must include hypothesis_family,
`multiplicity_method: fdr_bh`, uncertainty_plan and
`missingness_policy: common_complete_case`. D requires an explicit include_model_d.
Text-to-binary assay conventions are left for a future versioned contract;
this preflight only handles reviewed numeric_0_1 binary encoding. Unknown
category conventions never enter a design matrix.

A Pop equivalence proposal requires a signed pop_semantics document with
`equivalence_review: DOCUMENTED` and an explicit crosswalk restricted to
Unclassifiable/Unclassified. It is reported as an unapplied proposal; clinical
categories, missing and invalid values cannot be reassigned. Canonical Pop
origin is `01_pop_distribution.py`; `observed_patterns.py` verifies its output.
Step10 keeps protocol-related context outside the analytic master, which
explains why code aliases alone cannot repair missing demo__protocol.

## Outputs and disclosure

The package includes every required `03_graph_n3_phased_` artifact: qc.json,
summary.md, source_manifest.csv, protocol_source_inventory.csv,
baseline_date_lineage.csv, endpoint_date_readiness.csv,
candidate_evidence_matrix.csv, circularity_dependency_report.csv,
pop_crosswalk_proposal.csv, pop_reconciliation_qc.json,
topology_stability_audit.csv, model_preflight.csv, human_review_queue.csv,
proposed_contract_changes.md and tests_summary.json.

Additional artifacts are chronology_qc.json, clinical_review_template.md,
pop_reviewer_decision_template.md and artifact_inventory.csv. The latter hashes
every other output; it cannot contain its own hash. Every artifact carries the
three input identities/hashes, implementation/config provenance, restriction
flags and gate states. Runtime tests_summary records integrity assertions;
it explicitly does not claim to have executed pytest.

**Conservative disclosure choice:** patient partitions overlap across endpoints,
Pop universes, topology and complete-case tables. This version withholds all
patient counts, zeros, denominators, design dimensions and residual df across
the entire package. It calculates exact support and reconciliations in memory
before withholding them. This is stronger than row-only suppression of cells
1–4, which can leak through subtraction across tables. Counts cannot be
recovered from percentages because no such percentages are exported. Scientific
gates retain conservative statuses; institutional review is still mandatory
before any external sharing. No public-export or packaging route is provided.

A synthetic run's expected status excerpt (not clinical evidence):

```text
integrity_gate: PASS
phased_status: COMPLETE_WITH_BLOCKERS
historical_phasec_status: NOT_ESTIMABLE
status: NOT_ESTIMABLE
decision_status: ESTIMATION_BLOCKED
clinical_approval_granted: false
shareable_export_allowed: false
no_inference: true
no_refit: true
```

## Verification and remaining decisions

```bash
python -m pytest -q tests/test_graph03_n3_resolution.py \
  tests/test_graph03_n3_phasec_adjudication.py \
  tests/test_graph03_n3_candidate_audit.py tests/test_graph03_n3_incremental.py
python src/studies/graph/03d_graph_n3_resolution_preflight.py --help
```

The synthetic suite exercises five-representation key reconciliation, hash/run/
commit corruption, incomplete Phase A, failed/incongruent Phase C, collisions,
symlinks/traversal, frozen identity on errors, partial/rejected/invalid signoffs,
changed documents, protocol aliases, separate dates, future-derived outcomes,
Pop distinctions, assay variants, real dependency paths, ESSDAI/ESSPRI comparator
parity, R2/R0 restrictions, unknown categories, union complete cases, insufficient
community support, collinearity, residual df, separation feasibility and linked
disclosure. No fitted coefficients, p-values or predictions are produced.

Real-release regression must be run at the authorized protected site. Original
forms/assays, research protocol equivalence, multiple-protocol adjudication,
instrument dates/timezones, clinical independence, selection bias, hypothesis
families and topology sensitivity require independent human evidence. Neither
QC PASS nor numerical support establishes clinical validity, external validation
or prognostic utility. A separately approved new contract/release is required
before any later Graph03 analysis.

## Synthetic acceptance result

The final targeted run passed all 185 tests (zero failures, errors or skips):

| Suite | PASS | FAIL |
|---|---:|---:|
| `test_graph03_n3_candidate_audit` | 24 | 0 |
| `test_graph03_n3_incremental` | 17 | 0 |
| `test_graph03_n3_phasec_adjudication` | 24 | 0 |
| `test_graph03_n3_resolution` | 58 | 0 |
| `test_graph03_pop_denominators` | 2 | 0 |
| `test_graph_characterization_tiers` | 60 | 0 |

The standalone CLI also completed against a synthetic five-representation
bundle, producing all required restricted artifacts with integrity PASS and
COMPLETE_WITH_BLOCKERS. Compilation, CLI help and Git whitespace checks passed.
Real HPC regression was not run because the protected releases were unavailable.
