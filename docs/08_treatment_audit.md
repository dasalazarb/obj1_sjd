# Step 08: isolated treatment audit

`src/block_A/08_treatment_audit.py` inventories medication documentation and
creates reviewable audit artifacts. No scientific Step 08 script existed at the
implementation base, so this is a new standalone entry point. No existing script,
input, analytic contract or output is changed. It reads CSV/Parquet sources and
the existing longitudinal eligibility list; it never reads the Excel codebook.

## Run

From the repository root, with the established raw inputs available:

```bash
python src/block_A/08_treatment_audit.py
```

Defaults are `common.DEFAULT_ANALYTIC_DATASET` (legacy raw visits),
`common.LONGITUDINAL_ID_LIST` (CSV with `patient_id`), and
`outputs/qc/blockA/08_treatment_audit/`. Override the paths as needed:

```bash
python src/block_A/08_treatment_audit.py \
  --input /path/to/visits.parquet \
  --eligibility /path/to/list_ids_longitudinal.csv \
  --output-dir outputs/qc/blockA/08_treatment_audit/run_01
```

The default visits keys reuse `config.PATIENT_ID` and `config.VISIT_DATE`.
Missing inputs or configured keys block execution; no data is fabricated. Each
run requires a new or empty output directory, preventing overwrites and stale
products. Source positional rows are zero-based physical DataFrame rows, before
eligibility filtering; CSV row zero is the first data row below the header.
Source paths, SHA-256 hashes, original index, patient, visit and slot provide
record provenance. Eligibility and source hashes are recorded in the summary.

For multiple sources, canonical episode keys, protocol metadata or an index
already present in the source, pass `--sources-json /path/to/sources.json`.
This JSON list replaces `--input`. Required entries are `name`, `path`,
`patient_column`, and `visit_column`. Names must be unique simple identifiers.
Relative source paths resolve against the manifest directory. For example,
using the established upstream raw episode spine and its existing keys:

```json
[
  {
    "name": "episode_spine",
    "path": "/path/to/clinical_episode_spine_sjd.parquet",
    "patient_column": "patient_id",
    "visit_column": "clinical_episode_id",
    "index_column": "clinical_baseline_date"
  }
]
```

The example documents configuration, not availability of medication fields in
that spine. Verify each source schema. Optionally provide `protocol_column`
with a verified existing field, or `protocol` with a verified constant for a
single-protocol source. Both together are rejected. Otherwise protocol remains
`unknown`; interval names, form presence and missing records never determine
protocol or untreated status. This phase uses no downstream data sources.

## Products and rules

The field inventory is emitted and summarized before slot transformation. It
includes all 66 literal codebook fields for every observed protocol and separate
rows for `11-D-0172` and `15-D-0051`, even when no rows exist for either. A field
present in a table without rows for a protocol does not establish coverage for
that protocol. Counts use eligible source rows; eligible patients without any
source rows are reported separately in the summary.

All products use the `08_treatment_` prefix in the isolated audit directory:

| Suffix | Purpose |
| --- | --- |
| `field_inventory.csv` | Schema availability, dtype, null/negative counts, cardinality, patient/visit coverage and parse errors by source/protocol |
| `raw_slots_long.csv` | Every nonempty current slot, including orphan dates/routes and explicit negatives, with raw values and provenance |
| `historical_inventory.csv` | Separate historical names, never promoted to current exposure |
| `unassigned_doses.csv` | Raw dose fields and numeric/unit diagnostics; no RX-slot mapping |
| `normalization_map.csv` | Original current/historical vocabulary, source/protocol, exact rule, class and confidence |
| `episode_candidates.csv` | One audit-only candidate per observed current name; repeated visits remain separate |
| `patient_coverage.csv` | Distinct eligible patients by source/protocol/drug and class, with start/stop/bounded-course coverage |
| `temporal_qc.csv` | Record-level uncertainty, contradictions, duplicate names, repeats and potential concomitance |
| `feasibility.csv` | Coverage and explicit name/episode/date/indication gates; no automatic analysis approval |
| `audit_summary.json` | Reconciled totals, source exclusions, hashes, rule version, protocol gaps and blockers |
| `audit_report.md` | Findings, limitations and a final pending-decision matrix |

Rules version `08-treatment-audit-1.0.0` is declared in source. All field names,
including `frequency_` slots 3/5/6 and `anticogaulant_1_dose`, are literal. No
alternative frequency names are silently substituted. Route/frequency accept
exact codes, labels or consistent code-plus-label values; conflicting or unknown
values remain unmapped and appear in QC. Observed raw values are retained.

Name rules use exact generic/brand/abbreviation aliases and conservative
strength/form suffixes. Combination names and ambiguous matches remain unmapped.
Classes distinguish biologics, non-biologic systemic drugs, glucocorticoids,
symptomatic drugs, explicitly mapped other drugs and unmapped names. Matches
require review and provide no indication evidence: indication stays
`indeterminate`. Allergy fields are inventoried but never extracted as exposure.
The token `none` in a name field is an explicit negative documentation value;
it is never a patient-level conclusion of no treatment. Null/unknown cells and
unavailable columns have separate inventory states.

Monthly dates stay `YYYY-MM`, with original raw values and `month` precision.
No day is imputed. Full ISO dates/datetime objects retain `day` precision and
receive source-review flags; calculations conservatively compare their months.
Year-only values remain partial and other formats are flagged as ambiguous.
Equal start/stop months cannot establish ordering. Start before an explicitly
configured source index is a valid prevalent-treatment candidate. Missing stop
means open **or undocumented**, not confirmed ongoing treatment. Potential
bounded courses require start month strictly before stop month. Their coverage
is only a feasibility count, not an observed exact duration or a confirmed
clinical episode. Overlapping bounded month intervals and multiple names at a
visit trigger review; neither proves concomitant exposure.

No clinical outcomes were previously available inside Step 08. ESSDAI/ESSPRI,
Pop labels and post-treatment 3–12 month windows therefore remain pending
integration; no new clinical dependencies, pre/post studies, response models,
comparative effects or therapeutic sequencing are implemented.

## Implementation validation and outstanding data run

The implementation environment has no clinical visits, episode spine or
longitudinal eligibility dataset, either in the repository defaults or the
standard external locations. Consequently **there are no real treatment
coverage, vocabulary or date-quality findings yet**. The raw file
`graph_s4_lab_group_map.csv` is a laboratory mapping, not medication observations.
Run the command against verified clinical sources to generate the actual
feasibility report and pending-decision matrix. No synthetic results are shipped.

Verification:

```bash
python -m pytest -q tests/test_treatment_audit.py
```

The tests exercise irregular literal names, unavailable columns vs missing
cells vs explicit negatives, historical/allergy separation, conservative name
mapping, code/label conflicts, monthly/year/invalid dates, pre-index treatment,
repeated and concomitant candidate records, multiple protocols/sources,
eligibility reconciliation, provenance after filtering, inventory-first order,
CSV/Parquet manifests, missing-input blocking, and unchanged input/old-output
hashes. All new outputs are tested in temporary directories.

An upstream regression check of `test_input_data.py`,
`test_build_visit_spine.py`, `test_visit_dates.py` and `test_comorbidities.py`
found two existing failures: `test_date_fragments` expects `no_valid_date` for
`unknown` but gets `missing_date`; `test_family_burdens_are_separate_and_no_combined_total_exists`
expects progression conditions without `raynaud`/`cryoglobulinemia`. Both were
reproduced from unchanged base commit `08d542923870472d601581fda9ce65ab770467ec`.
No fixes to unrelated steps are included in this audit delivery.
