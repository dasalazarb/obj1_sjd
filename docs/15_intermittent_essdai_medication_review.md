# Optional Step 15: intermittent ESSDAI and medication documentation

`src/block_A/15_intermittent_essdai_medication_review.py` is an independent,
read-only command. It does not change the existing Step 15 command, any upstream
producer, the official clinical baseline, or the four immutable Step 13 v2
classes. Selection is retrospective and cannot be used as a baseline predictor.

## What it reconstructs

For each threshold independently (`k=3`, `k=5`), select the authenticated primary
`intermittent_observed` assignment, requiring three or more selected dates. Use
every consecutive pair in the original Step 13 sequence, checking dates, episode
identity and ESSDAI against the exact Step 10/11 inputs hashed by Step 13. No
classification, interpolation, imputation or model fitting is performed.

Each interval reports its actual endpoints, elapsed days, change from the index
and previous visit, entries/exits of `index ± k`, sign reversal, and an independent
`abs_change_ge_k_in_interval` descriptor. A small segment can still explain a
threshold reversion. The gap warning defaults to more than 365 days and is an
observation-density warning, not a significance criterion.

All 12 domain ordinal values retain missingness. No ordinal differences are
summed into ESSDAI points. Weighted contributions are marked unavailable because
the current integration has no separately certified weighted-domain contract.
Step 10 values remain inspectable in the visit table. ESSPRI and lab endpoint
context retains actual measurement dates; a lab without a valid date is marked
unavailable. Optional Step 10 context must have the identical episode-key set.

## Authentication and identity

The command reuses `verified_manifest` and `validate_inputs` from the existing
secondary trajectory analysis: definition, origin, PASS, **all** Step 13 artifact
hashes, exact Step 10/11 bytes, primary date policy and baseline identities.
The selected sequence must already be chronological and match the protected
support and index score.

For Step 08 it reuses the production `load_contract`, under a shared publication
lock, and additionally checks:

- the Step 00 spine hash/signature, exact Step 00/10 keys and structural dates;
- the seven nullable, canonical name fields (also against Step 10 when present);
- `confirmed_medication_exposure=false`, consent policy and calendar-year policy;
- the current restricted generation and every declared restricted artifact hash;
- reconciliation QC against the manifest;
- the established key against `08_key_compatibility.json`, then each catalog HMAC
  against the in-memory HMAC of its Step 00 patient ID;
- document/episode links against patient, source, original CTDB episode,
  ingredient, reference type, original date, evidence level and cutoff.

`--pseudonym-key` selects the existing institutional key; the default is
`<medication-audit-root>/restricted/.pseudonym_key`. The command never generates a
key, exports it, logs its value, or writes a raw/HMAC crosswalk. A missing or
incompatible key blocks processing. Raw MRNs are never guessed to equal HMACs.
Unassigned and unresolved-name queues must belong to the authenticated catalog.

## Documentary semantics

The original seven Step 08 name columns are preserved for every selected
patient's clinical episode, including visits without a selected ESSDAI score.
Null means undocumented/indeterminate; it does not mean untreated. Endpoint
contrasts say `reported_only_at_from`, `reported_only_at_to`,
`reported_at_both`, or `not_documented_either`, with list availability separately
recorded. Appearance and disappearance never mean initiation or discontinuation.

The full patient catalog, including unassigned, partial and unresolved documents,
supplies the history. Every original `order_start`, `order_end` and consent
reference is preserved, rather than using nearby-name unions as events. All
episode links and alternative selected references remain in separate audit
tables. Multiple references/links can belong to one document: document counts
deduplicate `(patient_id, documentation_record_id)`.

For `(from,to]`, each reference is labeled before/from, inside the interval,
after/to (default maximum 90 days), unknown/partial, or consent-only. Partial
dates keep their precision and bounds without an invented day. Future dates,
reversed start/end dates and unparsed references stay in review but are excluded
from strict temporal positioning. Consent is never a medication event. CTDB
uses its original source-visit date, which need not equal the clinical anchor.
An undated CTDB report stays undated.

All Step 08 events have `confirmed_exposure=unknown` and an empty
`confirmed_exposure_event`. Orders establish neither administration, dispensing
nor adherence; `order_end` is not a validated stop. Indication is retained only
when upstream documented it and remains unverified. Systemic SjD-specific
exposure flags therefore remain `unknown`. No new drug classification is inferred;
existing upstream categories and confidence are retained. CTDB visit-level doses
that have no slot mapping are never assigned to a drug.

The interval flags prioritize chart review and identify possible reverse
causation for orders after observed worsening. They do not support odds ratios,
efficacy claims, treatment-response labels or causal attribution.

## Protected outputs

Provide an institutionally authorized, external root named `restricted` or
under a directory with that name. The command rejects repository destinations,
symlinks in the destination path, and existing roots not owned by the running
user with mode `0700`. It creates directories with `0700` and files with `0600`
from the outset using a private umask. Local POSIX permissions do not replace
institutional access approval or the storage administrator's ACL policy.

Canonical layout:

```text
<authorized>/restricted/step15_intermittent_medication_review/
  clinical_baseline/delta3/                 # separate delta5 sibling
    intermittent_patient_roster.parquet
    intermittent_essdai_visits.parquet
    intermittent_essdai_jumps.parquet
    intermittent_essdai_domains.parquet
    secondary_endpoint_context.parquet
    patient_medication_history.parquet
    patient_medication_episode_inventory.parquet
    patient_medication_episode_links.parquet
    jump_medication_evidence.parquet
    visit_medication_report_contrasts.parquet
    threshold_comparison.parquet
    figures/patient_<hashed_pseudonym>_timeline.png
    figures/jump_<hashed_pseudonym>_interval_<ordinal>.png
    qc_checks.csv
    run_manifest.json
  controlled_aggregates/delta3/              # separate delta5 sibling
    cohort_documentation_summary.csv
    cohort_documentation.png
```

Figures align actual dates in three bands: ESSDAI, original ordinal domains and
documentary medication markers. No medication-duration bars are inferred.
Consent/uncertain references are counted in a separate non-temporal annotation
and remain fully inspectable in the evidence table. Each jump figure shades its
original interval. Patient filenames use a sanitized hash rather than an MRN or
unprocessed research identifier. None of these outputs may be committed.

Cohort tables give unique patient, interval and document counts and proportions,
including unknown/partial, consent and no-source-documentation categories, separately by threshold,
direction and source. Threshold-exit intervals also appear as a distinct,
overlapping review stratum. These strata and sources must not be summed as
independent patients. The controlled manifest records N at each threshold,
intersection and differences; the comparison table records changing assignments.
Sparse aggregate rows are suppressed in full, including denominators and
proportions, with a configurable local minimum of at least five. Zero is distinct
from suppressed. Aggregates remain controlled-access and require disclosure
review before external use; the script grants no public-release approval.

Publication stages a complete run privately on the destination filesystem and
renames it atomically. Existing runs are always protected by no-overwrite; there
is no overwrite flag. Failed staging is removed. Manifests record input hashes,
generation, commit, source hashes, date/disclosure policies and output hashes.
Console messages contain fixed statuses only, without identifiers or drug names.

## CLI

```bash
# Validate every input and compute tables in memory; zero output directories.
python src/block_A/15_intermittent_essdai_medication_review.py \
  --threshold both \
  --medication-features data/intermediate/08_medication_episode_features.parquet \
  --medication-audit-root /authorized/audit/08_medications \
  --pseudonym-key /authorized/keys/existing_step08_key \
  --restricted-output-root /authorized/research/restricted \
  --data-cutoff 2026-10-27 --dry-run

# Once inputs and protected storage are authorized, omit --dry-run to write.
# --no-jump-figures omits interval zoom PNGs; patient timelines remain.
```

Defaults use the existing canonical Step 00/08/10/11/13 paths in `common.py`.
Explicit input paths and `--context` support institutional layouts. Set
`--post-window-days`, `--measurement-gap-days`, and `--minimum-cell` before review
according to the protocol and local disclosure policy. `--format png` is the
only supported image format.

Missing canonical inputs return `BLOCKED_UPSTREAM` with exit code 2 and no
clinical products. Other validation errors return `BLOCKED_VALIDATION` with no
source values in console diagnostics. `--dry-run` creates no output directories,
Parquets or figures. `--input-origin synthetic_validation` must match the Step 13
manifest and inserts `synthetic_validation/` below the protected root; synthetic
inputs cannot be represented as canonical.

## Validation and limits

`tests/test_intermittent_essdai_medication_review.py` builds exclusively invented
Step 00/08/10/11 data and genuine Step 13 v2 synthetic producer outputs. It covers
independent threshold sets, original consecutive intervals, two-date rejection,
leading zeros, date/reference uncertainty, conservative report contrasts,
document deduplication, missing domains, cryptographic and semantic identity
failures, private publication, valid PNGs, input byte preservation, empty cohorts,
dry-run and blocked-upstream behavior. Run it alongside the existing Step
08/10/13/15 regressions.

No canonical cohort roster or medication results were generated during software
implementation: required protected canonical products are absent in this
checkout. True exposure intervals, adjudicated indication and causal longitudinal
analyses require separate future clinical validation and are outside this command.

### Implementation validation record

On 2026-10-27, the new synthetic suite passed **21 tests**. The required existing
Step 08/10/13/15 suites plus integrated-longitudinal tests produced **176 passed,
2 failed**. Both failures were reproduced in an independent, clean worktree at
the original commit `6678e255454a0edeb83d1a8d106a91e577d7f4e6`:

- `test_observed_patterns_v2.py::test_v1_endpoints_outputs_and_step14_source_are_unchanged`
  compares Step 10 source against an older historical commit, preceding the
  already-landed Step 08 integration changes.
- `test_08_medication_temporal_matching.py::test_leap_year_boundaries_priority_and_patient_isolation`
  supplies a legacy fixture without `canonical_medication_name` and
  `name_requires_review`, now required by the existing Step 08 matcher.

Those upstream sources and tests were left unchanged. No existing tracked file
was modified by this implementation. A canonical `--dry-run` returned
`BLOCKED_UPSTREAM` without creating its requested output root. Synthetic PNGs
were inspected and labeled explicitly as synthetic validation.

```bash
python -m pytest -q tests/test_intermittent_essdai_medication_review.py
python -m pytest -q \
  tests/test_secondary_trajectories_by_essdai_pattern.py \
  tests/test_observed_patterns_v2.py \
  tests/test_08_medications.py tests/test_08_medication_episode_contract.py \
  tests/test_08_medication_temporal_matching.py \
  tests/test_10_medication_input_contract.py \
  tests/test_10_medication_master_publication.py \
  tests/test_integrated_longitudinal_dataset.py
```
