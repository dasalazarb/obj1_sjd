# Step 10 — Medication-name summary and protected medication–episode detail

## Purpose

Step 10 includes the 11 previously approved medication-documentation descriptors **plus** a twelfth descriptive column, `meddoc__medication_names`, containing **distinct normalized ingredient names**, separated by semicolons. It is **not** a list of active medications; missing/undetermined ingredients remain absent from the name list but their documentation is retained in counts.

The master and scientific context keep a single row per authoritative Step 00 clinical episode. The existing 1,516 Step 08 many-to-many episode edges correspond to 1,466 original documents with at least one primary link; a BTRIS order can link to more than one clinical episode without generating a second original order. Original BTRIS identifier-QC blocks and one-calendar-month tolerance matches remain excluded from the main join.

## Outputs

1. `common.INTEGRATED_LONGITUDINAL_PARQUET` and matching CSV (default `10_integrated_longitudinal_clinical_episode.parquet/.csv`) and `10_integrated_longitudinal_context.parquet`: 11 existing `meddoc__*` descriptor columns plus `meddoc__medication_names`, if and only if Step 10 is run with `--medication-audit-root`.
2. **Restricted**: `outputs/audit/08_treatments_v2/restricted/step10_medication_episode_details/10_medication_documentation_episode_detail.parquet`. One row per primary medication document–episode edge, with patient, episode, original record ID, source, date, normalized ingredient, raw name, route, candidate categories, QC and BTRIS dose/UOM/frequency when documented. The protected detailed table must NOT be committed to GitHub or copied to public report outputs.
3. Step 10 QC: `outputs/qc/blockA/10_build_integrated_longitudinal_dataset/10_medication_documentation_qc.json`, including counts for mapped-name availability and dose/frequency presence. Actual QC directory is defined by `common.BLOCKA_QC_DIR`.

## Dose/route/CTDB boundaries

- **BTRIS dose:** raw `dose_raw`, `uom_raw`, `frequency_raw` from the restricted Step 08 treatment-record classification audit, validated with `source_audit_record_id`, protocol source and Step 08 release hashes. `strength_canonical` and `formulation_canonical` are carried from the restricted BTRIS release, not assumed equivalent to administered dose.
- **CTDB current at original visit:** name, ingredient and route; **no per-medication dose/frequency derived**. Earlier CTDB unassigned dose fields cannot reliably be linked to individual medication slots; they remain absent in this detail table.
- BTRIS `Order Date/Start Date` does not prove administration/start or treatment duration. CTDB report-current does not prove ongoing therapy outside source visit. A record may be **after** the clinical anchor while still **within** the authoritative episode interval. Protected table includes `documentation_date_relative_to_anchor`; names and counts in master are **descriptive, not pre-index exposure**.
- `meddoc__medication_names` uses only `ingredient_canonical` (not unreviewed raw names) so records with unknown/unmapped names will be counted but may not appear in the name list. It does not preserve multiplicity; consult the protected detail and count columns.
- `meddoc__n_high_exact_identity_documents` is a legacy name for **high_exact drug-classification confidence**, not independently confirmed patient identity; its interpretation must not be overstated.

## Run in Biowulf

```bash
cd /vf/users/salazarda/data/obj1_sjd
git pull origin main
PY=/usr/local/Anaconda/envs/py3.10/bin/python
$PY -m pytest -q tests/test_10_medication_documentation.py tests/test_10_medication_master_publication.py
$PY src/block_A/10_build_integrated_longitudinal_dataset.py --medication-audit-root outputs/audit/08_treatments_v2
```

Expected on the unchanged frozen upstream inputs: **497 clinical episodes, 159 patients, 497 main columns** (485 clinical and 12 descriptive medication), a 1,000-column context (previously 999), and **1,516 protected detailed links**. Actual execution is needed before declaring those new outputs validated.

## Data hygiene

No overwrite of frozen Step 00 or protected Step 08 catalog and link files. Protected detail write is atomic with directory mode 0700/file mode 0600. If any upstream catalog/release/audit hashes do not reconcile, Step 10 fails closed rather than silently reclassifying or rematching. Do not use medicine names/dose as predictors for pre-index outcomes or Mapper embeddings without a separate time-indexed methodology review.
