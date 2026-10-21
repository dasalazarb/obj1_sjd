# Step 08: CTDB + BTRIS audit 2.0

`src/block_A/08_treatment_audit_v2.py` adds a separate, read-only audit of CTDB
visit medications and both BTRIS medication-order exports. The v1 entry point,
rules, interfaces and outputs remain unchanged. This implementation does not
create exposures or modify Step 10, Step 14, Step 15, studies or master datasets.

## Run

Use the project's Python environment with pandas and pyarrow installed:

```bash
python src/block_A/08_treatment_audit_v2.py
```

The defaults are:

- CTDB: `common.DEFAULT_ANALYTIC_DATASET`, currently
  `/data/salazarda/data/obj1_sjd/data/raw/visits_long_collapsed_by_interval_codebook_corrected.parquet`.
- Eligibility: `common.LONGITUDINAL_ID_LIST`, currently
  `<repository>/data/raw/list_ids_longitudinal.csv`, with exact key `patient_id`.
- BTRIS 11D: `/data/salazarda/data/eda_sjd/data_analytic/BTRIS/11D/medication_records.parquet`.
- BTRIS 15D: `/data/salazarda/data/eda_sjd/data_analytic/BTRIS/15D/medication_records.parquet`.
- New output directory: `outputs/audit/08_treatments_v2/`.

These follow the current repository's CTDB/eligibility conventions, rather than
silently searching the previous audit's `/vf/users/...` location. The selected
paths, SHA256, size, mtime, Parquet metadata and missing/unexpected columns are
recorded in the restricted input manifest. No fallback source or cohort is used.
Verify that the selected eligibility file is the approved list before running.

Override paths explicitly, and use a fresh output directory on subsequent runs:

```bash
python src/block_A/08_treatment_audit_v2.py \
  --input /protected/visits.parquet \
  --eligibility /protected/list_ids_longitudinal.csv \
  --btris-11d /protected/BTRIS/11D/medication_records.parquet \
  --btris-15d /protected/BTRIS/15D/medication_records.parquet \
  --crosswalk-manifest /protected/approved_crosswalk.json \
  --previous-summary /protected/audit_v1/08_treatment_audit_summary.json \
  --output-dir /protected/audits/08_treatments_v2/run_02
```

The output directory must be new or empty and outside master-data directories
and v1 outputs. Unreadable/corrupt sources generate `BLOCKED_SOURCE_UNAVAILABLE`
and null counts, while readable empty files generate zero observed records.
Both are distinguishable from missing exact schema columns (`BLOCKED_SCHEMA`).
The remaining source audits continue. A blocked source, schema or eligibility
returns CLI exit code 2 **after** writing the diagnostic artifacts. Crosswalk
failure alone permits source inventories to finish with explicit review gates.

## Authorized identity linkage

No name/DOB matching or implicit equivalence of MRN and CTDB patient number is
implemented. Supply an existing authorized crosswalk via a JSON manifest:

```json
{
  "path": "existing_approved_crosswalk.csv",
  "mrn_column": "MRN",
  "patient_column": "patient_id",
  "authorization_evidence": "Reference to the approved linkage source and review"
}
```

`authorization_evidence` records the operator's assertion; the program cannot
independently establish institutional authorization. It is not a prompt for the
software to invent an identity link. Relative paths resolve against the manifest.
CSV keys are read as strings; numeric Parquet MRNs are invalid because leading
zeroes may already have been lost. Whitespace is trimmed, with no zero stripping,
case folding, fuzzy matching or numeric key conversion.

Repeated identical pairs are harmless. Any edge touching a one-to-many or
many-to-one conflict is excluded. Individual safe links may be inspected, but
**combined coverage requires a complete, unambiguous mapping of every eligible
patient**, readable sources and complete expected source schemas. This deliberately
conservative gate prevents an unmatched patient from being assigned to “neither.”
Without it, combined counts are null with `blocked_linkage` or `blocked_source`.
Full-universe MRN-token counts are distinguished from validated patient counts.

The overlap scopes `11D`, `15D` and `union` describe extraction sources, not
protocol-specific eligible enrollment cohorts. There is no authorized
per-protocol cohort denominator in the supplied schema. CTDB protocol remains
`protocol_provenance_unavailable` unless an existing documented column is passed
through `--ctdb-protocol-column`. BTRIS folder provenance and observed protocol
values are retained separately; mismatches are flagged, not filtered.

## Rules and units

- All 28 BTRIS headers are literal. Missing near-matches are reported without
  substitution. All 66 CTDB names are reused from v1's literal schema, including
  the frequency underscores in slots 3/5/6 and `anticogaulant_1_dose`.
- BTRIS counts use order rows; CTDB extraction uses visit rows and current RX
  slots. Cross-source coverage uses unique eligible patients. Global BTRIS
  inventories retain records outside eligibility and records with invalid IDs.
- A BTRIS order, its end date, take-home flag and authorized refills do not
  establish administration, discontinuation, adherence or SjD indication.
  Every observed order status/type/source label remains unvalidated until source
  owners review it. Cancelled/completed/active labels and revisions are preserved.
- Exact signatures include all source columns and distinguish nulls from blank
  strings. Same-ID discordance, exact reexports and possible cross-protocol
  reexports remain separate flags; records are never deduplicated or fused.
  Cross-protocol candidate matching requires an unambiguous crosswalk identity.
- PMH RX names are quarantined as `unverified_possible_allergy`; comparison with
  the allergy field is QC, not proof of medication history. Unassigned CTDB doses
  remain separate and are never attached to RX slots.
- Candidate normalization rules are versioned in code. Complete recognizable
  ingredients take precedence; product conflicts and unverified ingredients
  require review. Combinations retain components, Myfortic remains distinct from
  MMF, ocular cyclosporine remains distinct from systemic cyclosporine and local
  steroids are not counted as systemic glucocorticoids. Unmapped names remain in
  the complete restricted inventory, ordered for review by frequency/priority.
- MDY is an explicit parser rule, including `06/08/2015` and `03/27/2015`.
  Calendar errors are flagged. Month/year dates have calendar bounds and an
  empty observed `parsed` date; no observed day is fabricated. Native/ISO
  datetimes retain time and timezone. Bounds are compared as calendar dates;
  timezone differences do not establish intraday chronological order.
- Preconsent, pre-index and out-of-window orders remain visible. An optional
  existing `--ctdb-index-column` enables pre-index QC without downstream reads.
  Concatenated visit dates stay intact. Overlapping documented intervals are
  flagged without creating treatment courses or calculating durations.
- Prescription completeness is a structural count: name, full date, route,
  dose/strength, units and frequency. PRN/titration/loading/infusion/refill
  ambiguities have separate QC flags. Completeness does not validate dose
  intensity, daily dose or actual exposure. Original free text remains in the
  immutable source; the canonical audit copies presence flags rather than
  duplicating instructions or non-formulary narrative.
- The overlap definitions are an observed current CTDB name vs at least one
  BTRIS order record. “Neither documented” does not mean untreated. Incremental
  full dates, route, frequency, dose and recognized drugs are documentation gains.
  Union drug/class counts deduplicate validated patient IDs while retaining
  distinct evidence types and all source observations.

## Outputs and protection

All outputs are audit-only and generated under a fresh directory with mode 0700.
Files have mode 0600. This supplements the operator's existing protected storage;
the tool does not turn an arbitrary external location into approved storage.

The run root contains an aggregate summary and human-readable report:

- `08_treatment_audit_v2_summary.json`
- `08_treatment_audit_v2_report.md`

`restricted/` contains the 13 required CSV/Parquet products named in the
specification: BTRIS schema, source profile, status dictionary, full drug
inventory, date QC, dose/route QC and canonical orders; linkage QC, overlap,
normalization review, CTDB corrections, combined coverage and feasibility.
Additional restricted products provide the CTDB schema, current-slot Parquet,
unassigned-dose table, pairwise concordance, date availability and input manifest.
Raw vocabularies and dates remain restricted even in frequency tables. The
shareable report/summary contain fixed labels and aggregate numbers, with paths
and full input metadata available through the restricted manifest pointer.

Patient identifiers use keyed HMAC pseudonyms with separate namespaces for MRN,
eligible patient and order identity. A random per-run key is stored as
`restricted/.pseudonym_key`. For repeatable pseudonyms, pass an approved protected
key of at least 32 bytes with `--pseudonym-key-file`; it is never printed or
committed. Source row positions are zero-based before eligibility filtering.
Source hashes and mtimes are checked again at the end. A source change blocks
the run and requires a new run; input files are never modified.

The report ends with the decision table and a NO-GO for downstream integration
until a separate scientific review. Clinical pre/post ESSDAI is explicitly
`not_evaluated_in_this_audit`, never a count of zero eligible patients.

## Validation and outstanding clinical execution

```bash
python -m pytest -q tests/test_treatment_audit.py tests/test_treatment_audit_v2.py
```

Tests use synthetic CSV/Parquet fixtures, including both protocols, missing and
corrupt sources, all four overlap categories, empty tables, absent/ambiguous
crosswalks, PHI-bearing source text, schema exceptions, partial dates, order
revisions and immutable inputs/legacy scripts. No synthetic results are shipped.
The combined v1/v2 verification passes 97 tests. An invocation with the default
source locations was also checked: it writes the blocked-source diagnostics and
returns exit code 2, with unknown clinical counts left null.
The implementation environment has no actual CTDB/eligibility/BTRIS inputs or
authorized crosswalk; real coverage, observed vocabulary and clinical source
semantics remain unmeasured. Run the command in the protected BTRIS environment
to obtain the real inventories and pending-decision report.
