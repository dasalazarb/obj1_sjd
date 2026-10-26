> Historical documentation. Superseded for production by [the Step08 episode-name contract](08_medications_analytic_contract.md). Old count fields, month windows and audit-root CLI commands below describe archived products only.

# Step 08 — Medication documentation release (independent of exposure reconstruction)

## Objective

Make a dated, auditable medication **documentation** dataset available now, without blocking on the remaining ~81 CTDB unmapped records, BTRIS status validation, 20 reversed intervals, or uncertain product composition. **Do not equate documentation with administration, filled prescription, treatment start, adherence or effect.**

This is a separately invoked protected export, not a replacement for the Step 08 audit. It DOES NOT update Step 10, Step 14/15, pharma or graphs. Nothing outside the protected Step 08 directory is written.

## Research standards and interpretation

- HL7 FHIR distinguishes MedicationRequest (ordering), MedicationDispense (provision), MedicationAdministration (actual administration), and MedicationStatement (reported use); see https://hl7.org/fhir/R4/medicationrequest.html.
- OMOP CDM keeps order and exposure provenance, allows unmapped source vocabulary rather than forcing erroneous concepts, and only constructs medication eras after specifying exposure-end and gap assumptions; see https://ohdsi.github.io/CommonDataModel/cdm54.html.
- NLM RxNorm gives ingredient, form, strength, and brand normalization but does not cover all supplements, devices, or ambiguous descriptions: https://www.nlm.nih.gov/research/umls/rxnorm/overview.html.

## Run (Biowulf)

```bash
cd /data/salazarda/data/obj1_sjd
git pull origin main
/usr/local/Anaconda/envs/py3.10/bin/python -m pytest -q \
    tests/test_treatment_classification_v21.py \
    tests/test_treatment_audit_v2.py \
    tests/test_08_build_medication_dataset.py

# Refresh audit; source-only classification and existing QC remain unchanged.
/usr/local/Anaconda/envs/py3.10/bin/python src/block_A/audit/08_medications/legacy/08_treatment_audit_v2.py

# Creates medication documentation tables within protected audit directory.
/usr/local/Anaconda/envs/py3.10/bin/python src/block_A/audit/08_medications/legacy/08_build_medication_dataset.py \
    --audit-root outputs/audit/08_treatments_v2
```

The `outputs/...` path above assumes the repository's configured `OUTPUTS_DIR`. If your `08_treatment_audit_v2.py` output directory is elsewhere, use its actual full path for `--audit-root`.

## Restricted outputs

`restricted/medication_documentation_release/`:

| File | Grain | Meaning |
| --- | --- | --- |
| `08_medications_documented_catalog.parquet` | Named, linked patient-source record | ALL named linked documented medications, including unresolved/undated records |
| `08_medications_dated_observations.parquet` | Same, with an exact-day evidence timestamp | Dated orders/reported start/medication observed on visit, **not confirmed use** |
| `08_medications_normalized_dated.parquet` | Subset of dated observations | Recognized product/ingredient or approved study category, non-indeterminate category, no major QC contradiction |
| `08_medications_release_source_qc.csv` | Source | Aggregated totals only, no patient identifiers |
| `08_medications_release_summary.json` | Entire release | Counts and explicit downstream gate decisions |

**Date columns:**
- `date_of_documentation`, `date_evidence_kind`: dated evidence event and its type, not an exposure start.
- `source_start_date_exact`: from explicit day/datetime source start/order field (blank for month/year-only source dates).
- `source_start_lower_bound`, `source_start_upper_bound`: original uncertainty bounds for partial dates.
- `source_end_date_exact_unvalidated`: source end date only if noncontradictory; still NOT discontinuation.
- `source_time_semantics`, `source_end_semantics`: source-specific date cautions.
- `qc_flags`, `classification_review_reasons`: retained uncertainties.

No time period or medication era is inferred. No overlapping records are dropped automatically. `Citric Acid` remains `procedure_related`, `Mouth Kote` remains `symptomatic_sjd`; their classification confidence remains distinct from the category signoff.

## Authorised patient-level joining (optional)

Default release retains **pseudonymous linked patient IDs** only. Downstream joins to clinical episodes MUST NOT rely on presumed equality of pseudonyms.

ONLY in a protected, authorized NIH analysis environment and after confirming that the `patient_id` identifiers match the Step 08 source-eligibility contract:

```bash
/usr/local/Anaconda/envs/py3.10/bin/python src/block_A/audit/08_medications/legacy/08_build_medication_dataset.py \
  --audit-root outputs/audit/08_treatments_v2 \
  --eligibility PATH_TO_SAME_STEP08_ELIGIBILITY_FILE \
  --release-patient-ids
```

This option uses the pre-existing protected HMAC key (not regenerated), requires unique nonmissing eligible patient IDs, and rejects any row that does not map back. Resulting `patient_id` is restricted, still not an approval to join automatically in Step 10. Never upload protected row-level data to public channels.

## Acceptance criteria

1. Tests pass (unit tests + audit tests + release tests).
2. `08_medications_release_summary.json` reports a nonzero catalog and dated rows.
3. `n_documented_named = n_dated_observations + n_undated_retained`.
4. `n_normalized_dated <= n_dated_observations`.
5. All source inputs, version and schema reconcile; unchanged source fingerprints.
6. Check counts by `date_evidence_kind` and source; no false assignment of visit dates as treatment start.
7. External semantic validation of BTRIS status/identifier equivalence and actual exposure remains separately **NO_GO**.

## Next scientific stage

Use the dated documentation in **descriptive** timeline analyses (observation processes, cumulative dated-order counts, associations explicitly labeled as documentation associations). To estimate actual treatment effects or treatment eras, first validate BTRIS status and administration/dispense evidence, distinguish observed CTDB use from prescription, and specify interval construction and immortal-time/look-ahead safeguards.

Unknown products or partial dates remain in the catalog so analyses can include an unknown/partial-date sensitivity scenario rather than suppressing all other documented treatments.
