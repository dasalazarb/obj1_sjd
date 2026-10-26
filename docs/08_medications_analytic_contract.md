# Step 08 medication documentation contract

`src/block_A/08_build_medications.py` is the sole productive Step 08 entrypoint. It reads the recovered cohort-filtered BTRIS 11D/15D CSVs, the longitudinal cohort, original CTDB visits and row map, and both authoritative Step 00 spines. It does not read Step 10 or import archived scripts. Productive functions under `_internal/medications/` reuse the audited identifier normalization, HMAC namespaces, literal CTDB slot definitions, classification and duplicate/date QC rules.

The existing NIH HMAC key is required. A missing/short key or a changed key scope fails; no key is generated. Supply `--pseudonym-key` pointing to the established protected key when moving from an older audit root. Reconcile its scope with that root before the first NIH run.

## Published schema

`common.MEDICATION_EPISODE_FEATURES_PARQUET` points to `data/intermediate/08_medication_episode_features.parquet`. Its manifest replaces `.parquet` with `.manifest.json`. There is exactly one row per `(patient_id, clinical_episode_id)`, including episodes without assignable documentation. Patient/episode counts come from the configured Step 00, never fixed constants.

The specification's table contains **seven name columns**, plus the episode-key group (rather than eight distinct name columns). All seven are nullable strings, with sorted, unique, lowercase names separated by `; `. Null means no documentation assignable to that category in the reviewed sources; it does not mean no medication.

| Column after `medication__` | Meaning |
|---|---|
| `ctdb_reported_names` | Current medication reported at the original CTDB visit assigned by Step 00 |
| `btris_order_within_episode_names` | Selected exact order/start or end date within the episode |
| `btris_order_before_episode_1y_names` | Selected order date before the episode, within one calendar year |
| `btris_order_after_episode_1y_names` | Selected order date after the episode, within one calendar year |
| `btris_order_nearby_1y_names` | Union of the three order-date categories |
| `btris_consent_only_review_names` | Consent is the only matching reference; chart review only |
| `chart_review_candidate_names` | Union of original CTDB reports, order matches and consent review candidates |

High-confidence canonical ingredients are preferred. Otherwise, the source drug name (or original ingredient when the name is absent) is retained as a sanitized review candidate. Contact-instruction patterns are suppressed from the episode names; untouched evidence stays protected. `name_requires_review`, confidence, classification/QC flags and source multiplicity remain in the long catalogue. Sanitization is a conservative heuristic, not permission to make a patient-level contract public.

## Time and identity

Only episodes of the same validated patient can match. Exact original order/start, order end and initial consent are compared against `[episode_start - DateOffset(years=1), episode_end + DateOffset(years=1)]`, inclusively. This handles leap years as calendar years. Parser rules retain partial-date precision/bounds; month/year values never acquire a fabricated exact day. ISO and explicit MDY timestamps, including fractional seconds and numeric zones, retain the source calendar day.

One document has at most one relation to a given episode. The selected reference priority is order/start, order end, consent, restricted to dates that actually match the window. All matching reference kinds remain on that relation; all parsed original references and bounds remain in the protected catalogue. A document with several reference dates occupies the category of its selected reference, never several duplicated edges. An order may have relations to multiple episodes. CTDB reports keep the original episode even if their recorded visit date raises QC; no guessed episode or dose-slot assignment is made.

Consent never becomes a medication date. A malformed order date can coexist with a consent-only review candidate, with the parsing failure explicitly counted in QC and retained per document. Undated and out-of-window documents remain patient-anytime/unassigned. End dates do not establish discontinuation; no output establishes active therapy, administration, dispensing, adherence, SjD indication, continuity or causal effects. Selected dates after the anchor are flagged in the protected relation table. All medication columns are descriptive and rejected by the shared study predictor validator, including `from_` versions; longitudinal graph embeddings also exclude them.

## Validation and persistence

Manifest version `08-medication-episode-features-1.0` records the producer version, UTC run time, code commit, SHA-256 of the final parquet and source/code inputs, structural Step 00 signature and file hash, columns/dtypes, exact keys, counts, calendar-year policy and aggregate QC. No source paths, rows, identifiers or key material are exported in that manifest.

The producer stages files on the output filesystem, validates the parquet round trip, and uses a publication lock during replacement. Readers hold the same lock through manifest/hash/schema/spine validation. Rename failures restore the old public pair; an abrupt process interruption that leaves mismatched files fails closed on the next read. Protected long products live in immutable `restricted/generations/<run>/` directories, with `restricted/current` and the documented filenames pointing to the current generation. Its tag and detail-file hashes are recorded in the manifest. Directories are 0700 and data/manifest/QC files are 0600. A failed publish may leave an unreferenced protected generation for investigation.

The protected products are `08_medication_documentation_long.parquet`, `08_medication_candidate_episode_links.parquet`, `08_medication_order_date_details.parquet`, `08_medication_patient_anytime.parquet`, `08_medication_unassigned.parquet`, `08_medication_review_queue.parquet`, and aggregate reconciliation QC. They preserve source documents, raw names/ingredients, source/order pseudonyms, route/dose/frequency/status, date provenance and possible reexports. CTDB unassigned visit doses stay explicitly unassigned. No restricted product is an input to Step 10.

`--dry-run` builds/validates in memory without publishing. `--reuse-validated-inputs` reuses only a complete contract whose source/code hashes and protected key scope still match; it never trusts old truncated BTRIS parquets. `--force-rebuild` allows a replacement publication, preserving previous protected generations; it cannot authorize a key change. Fixed inputs produce identical analytic parquet bytes.

## NIH staged validation

```bash
python src/block_A/08_build_medications.py --help
python src/block_A/08_build_medications.py \
  --pseudonym-key /protected/existing/audit/restricted/.pseudonym_key \
  --eda-root /protected/eda_sjd \
  --output outputs/staging/08_medication_episode_features.parquet
python src/block_A/10_build_integrated_longitudinal_dataset.py \
  --medication-input outputs/staging/08_medication_episode_features.parquet \
  --output outputs/staging/10_integrated_longitudinal_clinical_episode.parquet \
  --qc-dir outputs/staging/10_qc
```

Step 10 requires this input by default; `--no-medications` explicitly produces a clinical-only build and records the omission in medication QC. The old `--medication-audit-root` option is removed. Both analytic and context products receive identical name fields and matching registry metadata, with no medication counts or raw dose/order detail. Never promote the staged Step 10 until the NIH comparison and scientific review are approved. Protect these generated files; none belongs in GitHub.
