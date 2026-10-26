> Historical documentation. Superseded for production by [the Step08 episode-name contract](08_medications_analytic_contract.md). Old count fields, month windows and audit-root CLI commands below describe archived products only.

# Step 08: CTDB + BTRIS audit v2.1

`src/block_A/audit/08_medications/legacy/08_treatment_audit_v2.py` is the only entrypoint. Its private helper
`_treatment_classification_v21.py` extends the existing v2 audit with conservative
product axes, observed order-context matrices and patient coverage by evidence.
It creates no therapeutic episodes or integrated clinical dataset.
`downstream_integration` remains exactly `NO_GO`. Step 08 v1, Step 10/14/15,
Block B, studies, common paths and master sources remain unchanged.

## Run and permitted writes

Use the project's Python environment with pandas and pyarrow:

```bash
python src/block_A/audit/08_medications/legacy/08_treatment_audit_v2.py
```

Defaults read only:

- CTDB: `common.DEFAULT_ANALYTIC_DATASET`, currently
  `/data/salazarda/data/obj1_sjd/data/raw/visits_long_collapsed_by_interval_codebook_corrected.parquet`.
- Eligibility: `common.LONGITUDINAL_ID_LIST`, column `patient_id`.
- BTRIS 11D: `/data/salazarda/data/eda_sjd/data_analytic/BTRIS/11D/medication_records.parquet`.
- BTRIS 15D: `/data/salazarda/data/eda_sjd/data_analytic/BTRIS/15D/medication_records.parquet`.

Writes default exclusively to `outputs/audit/08_treatments_v2/`. Preliminary
reruns overwrite only known artifact names and retain unrelated files and the
existing `restricted/.pseudonym_key`. They never delete the directory or v1
outputs. Source paths can be explicitly overridden for protected executions or
synthetic tests; no alternate source or cohort is searched automatically.

```bash
python src/block_A/audit/08_medications/legacy/08_treatment_audit_v2.py \
  --input /protected/visits.parquet \
  --eligibility /protected/list_ids_longitudinal.csv \
  --btris-11d /protected/BTRIS/11D/medication_records.parquet \
  --btris-15d /protected/BTRIS/15D/medication_records.parquet
```

Missing/unreadable sources produce `BLOCKED_SOURCE_UNAVAILABLE` and null counts;
readable empty sources produce zero observed rows. Missing required columns
produce `BLOCKED_SCHEMA`. Partial optional CTDB schema remains usable and is
reported separately. Blocked source/eligibility runs write diagnostic artifacts
and return CLI exit code 2. A linkage failure alone allows source inventory.
Input paths, SHA256, size, mtime and schema remain in the restricted manifest.

## Identity, provenance and counting units

The default `--linkage-mode serology_normalized_id` reuses the existing Step 01
normalization precedent: strip approved separators/leading zeros, safely accept
integral numeric Parquet MRNs, and reject unsafe floats, invalid IDs and CTDB
normalization collisions. It does not rewrite the linkage algorithm or match on
name, DOB, sex, age or protocol. `validated` is a technical gate; the assumption
that the two systems use the same identifier still requires external semantic
validation. New row-level fields expose `linkage_method`, `id_precision_qc` and
`linkage_semantic_validation=external_validation_pending`.

`--linkage-mode crosswalk_only` instead requires an existing authorized
crosswalk. An explicit `--crosswalk-manifest` overrides normalized matching:

```json
{
  "path": "existing_approved_crosswalk.csv",
  "mrn_column": "MRN",
  "patient_column": "patient_id",
  "authorization_evidence": "Existing approval reference"
}
```

Relative paths resolve against the manifest. Crosswalk keys require exact text,
trim whitespace only, and reject numeric MRNs. Repeated identical pairs are
harmless; every edge touching an identity conflict is excluded. This mode
requires a complete unambiguous eligible crosswalk for combined coverage.
No institutional authorization is inferred from the supplied string.

BTRIS counts use **order rows**, CTDB uses **current RX slots/visit rows**, and
coverage uses **sets of unique eligible patient pseudonyms**. All BTRIS rows,
including unmatched/outside-eligibility records, unknown products, possible
reexports and revisions, remain in inventory. No drug-name deduplication occurs.
Order-ID revision matching requires source folder, source MRN token and source
system; cross-folder candidates also require technical identity linkage and the
same source system. Numeric MRNs retain pseudonymous order identity for QC.
Counts of raw distinct order IDs are descriptive, not a deduplicated order total.

Folder 11D/15D is extraction provenance, never exclusive enrollment or POP.
Actual `Protocol Number`, source system, category, drug type, order set and type
remain separate. Without an approved POP column, the stratum is
`population_stratum_not_available`; no downstream data is imported.

CTDB keeps literal v1 field names, including frequency underscores in slots
3/5/6 and `anticogaulant_1_dose`. Existing nonempty/orphan/negative slot records
remain traceable. Column availability is distinct from an empty cell in schema
QC and `start_column_available` / `end_column_available`. PMH RX names remain
quarantined possible allergies, excluded from all current-medication counts.
Unassigned doses stay separate.

## Classification axes and uncertainty

Existing v2 normalization columns and files are preserved. New axes are additive
and versioned as `08-classification-2.1.0`:

- Ingredient scalar plus stable JSON ingredient list, product, strength tokens,
  route, formulation and pharmacologic family. Product identity retains route,
  formulation and observed strength; no dose equivalence is asserted.
- Potential SjD systemic, ocular or oral relevance; general/concomitant,
  procedure/diagnostic, dental/local, device or indeterminate candidates.
- Original order type/context and a separate procedure-context candidate based
  only on explicit `procedure`, `visit`, `test`, `biopsy` or `sialometry` order-set
  tokens. The rule ID and original restricted text remain inspectable. Secondary
  context labels do not multiply order counts.
- Evidence source (`ctdb_reported_current_medication` or
  `btris_documented_order`), rule IDs, identity confidence and review reasons.

Rules accept exact aliases with an anchored strength/formulation suffix, never
fuzzy similarity. Complete recognized combinations retain every ingredient.
Unknown extra components cannot yield a single-agent exact mapping. Generic
artificial tears/Systane identify a lubricant product, with composition pending
and no asserted chemical ingredient. Conflicting
product/ingredient identities keep both original fields, clear the asserted
canonical ingredient and flag review. Route conflicts are unresolved.
`high_exact` describes identity matching only; it does not confirm clinical use.
`candidate_review` and `unmapped` are pending review. Clinical indication is
always false; administration and dispensing are always `not_established`.

The observed priority vocabulary has explicit conservative rules:

| Product | Candidate treatment of the record |
| --- | --- |
| Citric Acid Monohydrate oral solution | Procedure/diagnostic candidate; use pending; never systemic SjD therapy |
| Sodium fluoride topical gel | Dental/local support |
| Ibuprofen / acetaminophen | General/concomitant symptomatic product |
| Carboxymethylcellulose ophthalmic | Ocular support, distinct from oral/generic formulations |
| Cevimeline / oral pilocarpine | Oral symptomatic candidate; no individual indication assigned |
| Mouth Kote Dry Mouth Treatment | Oral support candidate; composition unresolved |
| Ophthalmic cyclosporine / Restasis / Cequa | Ocular candidate, distinct from oral/systemic cyclosporine |
| Myfortic vs CellCept | Separate molecules; no dose collapse |
| Antifungals | Specific family; SjD relevance indeterminate |
| Contrast / AeroChamber | Procedure or device, never systemic SjD by default |

Local, ophthalmic and systemic glucocorticoids are separate; an unresolved route
requires review. Salt/brand expansions are limited to declared aliases. The
restricted dictionary enumerates aliases, versions, route/formulation policy,
rule provenance and confidence policy. Investigator-provided nomenclature is a
candidate rule source, not an independently validated institutional dictionary.

## Order semantics and temporal QC

All observed status codes, including AUC1, CANP, OPCMP, DISC, OPODC, OPOA, AUA1,
COMPT and unknown future codes, remain `unvalidated`, with evidence `none`.
OnSite, Take Home and Home Med never establish administration or dispensing.
The generated semantic-reference template contains the requested effective-date,
source-system, clinical-support and reviewer/documentation fields, all pending.
No external calls or protected dictionary loads occur. Institutional semantic
validation is a separate future input; this iteration supplies no approved map.

MDY parsing explicitly supports `MM/DD/YYYY HH:MM` and `MM/DD/YYYY HH:MM:SS`.
Intraday reversals use the original timestamps. ISO/native timestamps retain
precision and timezone. Month/year inputs retain calendar uncertainty bounds
and an empty observed date; no day is imputed. Dates are never corrected.

`order_interval_qc` definitions:

| Label | Administrative interval definition |
| --- | --- |
| reverse | Exact end precedes exact start |
| zero | Exact endpoints equal |
| short | Positive interval below 24 hours |
| plausible | Interval at least 24 hours; label is not clinical validation |
| missing | At least one endpoint missing |
| uncertain | Partial/unparsed endpoint or incompatible timezone awareness |

`order_start_semantics=order_or_start_unresolved` and
`order_end_semantics=order_validity_end_not_confirmed_discontinuation` apply to
BTRIS. Preconsent records, inversions, repeated intervals and overlaps remain
visible. No exposure, received dose, treatment duration, adherence or episodes
are derived.

## Outputs, privacy and reconciliation

Directories are 0700 and generated files 0600. All detailed vocabularies,
patient-level pseudonyms and row-level records remain under `restricted/`.
Raw MRN, subject name, DOB and free-text order instructions/non-formulary notes
are never copied to the canonical rows. HMAC namespaces distinguish patients,
source MRNs, orders and signatures. The pseudonym key is reused across reruns,
or an existing protected >=32-byte key can be supplied explicitly.
No outputs or keys are committed to Git.

All legacy v2 products continue to be generated, including source/schema/status,
drug/date/dose inventories, order/slot Parquets, linkage, overlap, concordance,
coverage, allergy quarantine, feasibility and the input manifest. New products:

| File | Scope |
| --- | --- |
| `restricted/08_treatment_classification_dictionary.csv` | Exact aliases and reproducible rule metadata |
| `restricted/08_treatment_record_classification_audit.parquet` | One row per existing BTRIS order or CTDB current/orphan slot |
| `restricted/08_treatment_unmapped_priority.csv` | Pending vocabulary, sorted by BTRIS row weight; first 20 are review priorities |
| `restricted/08_btris_status_type_matrix.csv` | Five labeled matrices; each separately reconciles to loaded order rows |
| `restricted/08_btris_order_set_context_qc.csv` | Order set/type/category/product context and counts |
| `restricted/08_treatment_coverage_by_evidence.csv` | Unique eligible patients by source, evidence level, route and union |
| `restricted/08_treatment_classification_conflicts_qc.csv` | Identity/route/manual/status/duplicate/temporal review rows |
| `restricted/08_btris_status_semantic_reference.csv` | Observed code/type/source combinations, pending institutional review |
| `08_treatment_classification_v21_summary.json` | Aggregate counts, coverage, reconciliation, fingerprints and NO_GO |
| `08_treatment_classification_v21_report.md` | Scientific review narrative and decision table |

Matrices cover folder/protocol/status/type; status/type/take-home/source;
status/type/category/drug-type; status/date precision/temporal flags; and
order-type/relevance/family. Each group has `n_order_rows`, raw distinct
pseudonymous order IDs, technically linked eligible patients (null if linkage
blocked), and valid-start counts. Do not sum different matrices together.
Ingredients are never exploded to compute an order-row metric.

Coverage includes any documentation, identifiable ingredient, systemic/ocular/
oral candidates, procedure candidates, unmapped orders, full order datetimes,
identified route + valid start, and route strata. Each scope uses only the
approved full eligible denominator; no cohort size is hard-coded. Cross-source
counts are null if linkage or required sources are blocked. Patients appearing
in both folders count once in the union. Clinical subcategories overlap and
must not be summed. Missing documentation never establishes untreated status.

Summary checks reconcile source row sums, classified + pending order rows,
status matrix totals and record identity uniqueness. `n_orders_classified`
means `high_exact`; pending includes `candidate_review` and `unmapped`.
Ingredient mentions are a separate count and may exceed order rows. The v2
summary additionally retains the four overlap categories and their denominator.
Prior investigator numbers are reconciliation references, never constants.

On rerun the manifest is read before overwriting. Source SHA changes and changes
to the versioned dictionary fingerprint are reported in `fingerprint_drift`;
manual edits to the previous generated dictionary are also detected;
a first run has `no_previous_manifest`. The manifest stores the dictionary
fingerprint; original source hashes/mtimes are rechecked at the end. A change
during execution blocks the run. Previously generated summaries are not reused
as current scientific results.

## Validation and remaining NIH execution

```bash
python -m pytest -q \
  tests/test_treatment_audit.py tests/test_treatment_audit_v2.py \
  tests/test_treatment_classification_v21.py
```

Synthetic tests cover priority vocabulary, all observed status codes and three
order types, combinations, conflicts, local/systemic routes, Myfortic/CellCept,
1,208 synthetic retained rows (not a new clinical result), intraday inversions,
identity safety/collisions, overlapping patient coverage, privacy, source and
protected-code hashes, repeatable outputs/key and input/dictionary drift.

The implementation environment has no actual NIH CTDB, eligibility, BTRIS or
prior audit files. Synthetic tests do not update the investigator's reference
counts. Run this entrypoint in the protected NIH environment to obtain real raw
and unique counts, the top 20 unresolved products, status/type distributions and
coverage/QC reconciliation. Until then those clinical results remain pending.
Stop at audit outputs and scientific review: no downstream integration is
approved by this implementation.
