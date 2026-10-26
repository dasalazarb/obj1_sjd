# Step 15: secondary measurements within fixed observed ESSDAI groups

**TESTED_SYNTHETIC / NOT_RUN_CANONICAL — BLOCKED_UPSTREAM**

Implementation audited against live `main` commit
`9dc66a8b55037fc738e032c83f26f27163b26359` on 2026-10-26. The initial clean
local checkout was `e52a4a1`; it was fast-forwarded to current main before
implementation. This is a software validation report, not a cohort analysis.

## Input readiness

Neither canonical Step 10 nor Step 11 Parquet is available in this checkout.
The protected Step 13 v2 patient-pattern, eligibility and selected-date files,
and its manifest, are also unavailable. The supplementary XLSX codebook is
absent. Clinical group sizes, clinical variable availability, real input hashes
and clinical PNG counts are **not available**, not zero. The default canonical
dry run exits with `BLOCKED_UPSTREAM` and writes no final output.

The source audit covered Steps 01 (extended phenotype and serology), 06, 09,
10, 11, 13 v2, the exact `plot_pattern_trajectories` reference, the Step 14
predictor metadata, shared overlap definitions, PRO metadata, common paths and
config. No upstream script, model, classification rule or baseline is changed.

## Consumer contract

The entry point is `src/block_A/15_secondary_trajectories_by_essdai_pattern.py`.
The three `secondary_trajectory_*` modules implement the catalog, observed
measurement analysis and PNG presentation. The consumer never calls a pattern
classifier, questionnaire scorer, phenotype derivation or statistical model.

Before analysis it verifies the manifest definition `observed-essdai-pattern-v2`,
QC PASS, input origin, every listed artifact SHA256, the exact protected pattern
path, and the producer's Step 10/11 input-byte hashes. The manifest is resolved
from its canonical repository-style `outputs/qc/blockA/13_.../` location.
Relocated upstream bundles must preserve that layout. It validates primary v2
date policy, both thresholds, all episode/patient identities, complete Step 11
baseline coverage, baseline equality, selected-date/index identities and support,
and exact pattern/eligibility-table agreement. It does not reconstruct groups
from ESSDAI values. Each join is explicitly one-to-one or many-to-one.

The group map is fixed per threshold. Four panels always retain Step 13's
`LABELS` order and `COLORS`. Patients without a secondary measurement remain in
`N_group`. Exactly two ESSDAI dates are provisional and use dashed paths;
intermittence requires three or more. Missing measurements are never negative,
zero, carried forward or otherwise imputed.

Step 10's context sidecar supplies laboratory units, actual measurement dates
and protocol metadata. It is joined only on the canonical episode keys, with
complete coverage and null-safe equality checks for shared fields. It never
promotes a clinical measurement absent from the analytic master. The source
variable registry adds candidates, not new measurements. Both sidecars are
optional if absent; explicit missing sidecar paths block execution, and missing
required metadata excludes affected variables. Sidecar bytes are hashed in the
run manifest; the Step 13 manifest does not authenticate these later sidecars.

## Measurement policies

The main X clock is the actual measurement date minus the official Step 11
clinical baseline, divided by 365.25. Both this clock and the ESSDAI-index clock
are retained in memory. ESSDAI-index sensitivity retains negative times for
measurements on/after clinical baseline but before the later ESSDAI index.
Malformed, partial, contradictory or post-cutoff calendar dates block execution;
undated and pre-baseline secondary measurements are counted and excluded.
Episode-aligned ESSPRI, PRO, overlap and extended measures follow their
producer's existing episode assignment. Laboratories require their independent
`__measurement_date`; no nearest-episode matching is performed.

Within a variable/date, an official baseline-episode observation on the official
baseline date has priority. Concordant same-day duplicates use the last sorted
canonical episode ID. Discordant dates without a justified official tie are
excluded and audited; different tests are never averaged. The conservative
discordant-date exclusion is the implemented policy, not a separate fitted
sensitivity analysis. No left/right ocular values are combined downstream.

The reference is the first retained valid measurement on/after clinical
baseline, with official-episode priority on the baseline date. A laboratory
measured on a later independent date remains a delayed reference even if its
episode is called baseline. The reference stays at its actual X coordinate;
change is `value - own_reference`. A singleton contributes a profile/absolute
observation and never a delta trajectory. The reference point has delta zero
only for a person with repeated measurements.

One last actual observation contributes per patient per half-open annual
window `[start,end)`. Individual points remain at their real dates. CSVs report
median, IQR, mean, SD, extrema, raw visit count, unique-patient count and delta
follow-up support. Numeric figures show observed paths without a population
curve. Category proportions use evaluable patients and Wilson 95% intervals
for binaries; intervals are suppressed below `--minimum-support` (default 5).
Sparse estimates are flagged, never joined into a smooth curve. Comparisons
across groups require adequate support in all four groups.

Protocol availability counts each patient once per group/window/protocol.
A patient observed under two protocols may occur in both protocol rows; these
rows must not be summed into an overall cohort denominator.

Cumulative `ever_positive_through_episode` fields are displayed exclusively as
first-available known-history profiles, not repeated tests or seroconversion.
First-profile category denominators pool one observation per patient across
their actual first-measurement dates and imply no shared measurement time.
Undated external baseline serology summaries (including low C4 and leukopenia)
are cataloged and explicitly excluded until a verified individual test date is
available; baseline membership does not manufacture a test date.

The catalog covers every analytic column and every source-registry candidate,
plus the prespecified ESSPRI, ocular, salivary, key laboratory, serology, PRO,
pathology and SGUS candidates. Known clinical metadata determines type, not
pandas dtype. Unverified fields, identifiers, technical metadata, unavailable
columns, missing dates, unknown/incompatible laboratory units, assay/scoring
changes and invalid scales receive explicit decisions. Constant unit aliases
are consumed as published; this consumer performs no new unit conversions.
ESSDAI-domain and ESSDAI-dependent overlap measures carry a circularity note.

## Execution

Use the repository Python environment with pandas, pyarrow, matplotlib and
the existing dependencies. Tests also require pytest and Pillow.

```bash
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py --help
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py --threshold 3 --dry-run
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py --threshold 3 --format png --dpi 250 --no-overwrite
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py --threshold 5 --no-overwrite
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py --threshold both --time-axis essdai_index --no-overwrite
```

Use `--data-cutoff YYYY-MM-DD` for the study's verified freeze date; the
default is today's UTC calendar date. `--variables` and `--domains` select
explicit subsets while retaining unselected catalog records. `--max-variables`
is development-only. Use `--context`, `--source-registry`, `--integrated`,
`--baseline`, `--patterns` and `--patterns-manifest` for authorized alternatives.

For a deliberate rerun, `--overwrite` requires `--backup-root` below this
Step 15 run's effective output root:

```bash
python src/block_A/15_secondary_trajectories_by_essdai_pattern.py \
  --threshold 3 --overwrite \
  --backup-root outputs/backups/15_secondary_trajectories_by_essdai_pattern
```

All prior files of the selected Step 15 bundle are backed up with a UTC stamp.
The complete replacement is staged first; installation errors roll back final
directories. QC/manifests install last. This is exception rollback, not a
filesystem-wide crash-atomic transaction. Other thresholds and clocks remain
unchanged. `--dry-run` validates all selected analyses with no final writes,
even if their destination already exists.

`--input-origin synthetic_validation` must match the producer manifest and
forces output below `output-root/synthetic_validation/`; PNGs visibly state
`SYNTHETIC SOFTWARE VALIDATION`. An authenticated synthetic bundle cannot be
run with the canonical label.

## Outputs and privacy

The two clocks and thresholds have separate namespaces to permit independent
execution and backups. For each of `figures`, `tables` and `qc`, paths are:

```text
outputs/<area>/blockA/15_secondary_trajectories_by_essdai_pattern/
  clinical_baseline/delta3/
  clinical_baseline/delta5/
  essdai_index/delta3/
  essdai_index/delta5/
```

Figures add domain subdirectories and deterministic names
`<variable>_absolute.png`, `_delta.png`, `_prevalence.png`,
`_category_distribution.png` or `_first_available_profile.png`.

Each threshold's tables contain `variable_registry.csv`,
`group_support_by_variable.csv`, `trajectory_summary_by_window.csv`,
`binary_category_summary.csv`, `availability_by_protocol.csv`,
`excluded_variables.csv` and `plot_manifest.csv`. Baseline-only profiles are
distinguished from exclusions. Every QC threshold has `qc_checks.csv` and its
own `run_manifest.json`, so a later ±5 run cannot invalidate ±3 provenance.
The run manifest records code commit and source hashes, flags, inputs and all
Step 13 artifact hashes, definitions, temporal/duplicate rules, exact aggregate
group sizes, warnings and output SHA256s.

Patient frames, reference dates and mappings remain in memory. No patient-level
CSV, Parquet, IDs or individual dates are written to publishable `outputs/`.
The PNGs do contain observed individual paths without patient labels and remain
controlled research artifacts. Small aggregates are not automatically cleared
for external disclosure. Generated files and datasets are not committed.

## Validation and limits

Synthetic tests exercise the actual Step 13 v2 producer's manifest and files,
four patterns plus a two-date provisional patient, both thresholds, immutable
membership despite identical secondary curves, delayed C3 references, both
clocks, singleton/missing groups, same-day conflicts, evaluable denominators,
patient-window and protocol counts, origin/hash/identity/date failures, registry
coverage, valid PNGs, exact panel/colors/legend semantics, privacy, no-overwrite,
sequential thresholds, explicit backups, rollback and a no-classifier/model AST
check. Full synthetic execution creates **19 PNGs per threshold (38 total)**;
these are software fixtures, not cohort findings.

```bash
python -m pytest -q tests/test_secondary_trajectories_by_essdai_pattern.py
python -m pytest -q tests/test_observed_patterns_v2.py tests/test_disease_activity_progression.py tests/test_risk_factors_progression.py
```

Step 15: **29 passed**. Combined upstream regression: **220 passed, 1 existing
failure**. The failure is
`test_v1_endpoints_outputs_and_step14_source_are_unchanged`: it compares today's
Step 10 bytes with historical commit `e96f189`, before the already merged
medication-documentation changes. It was independently reproduced in a clean,
detached worktree at `9dc66a8`, without any Step 15 files. No upstream tests or
code were edited to hide it. SHA256 checks against the execution-base files
confirm that Steps 10/11/13/14, the v1/v2 classifiers and figure module are
unchanged. Canonical output hash comparisons remain unavailable because those
products are absent; synthetic upstream products are verified byte-for-byte
unchanged. PNG examples of numeric absolute/change, binary, categorical/ordinal
and known-history profile modalities were visually inspected.

The manifest's scientific limitations include retrospective complete-history
selection, unequal follow-up, regression to the mean, floor/ceiling effects,
informative missingness, assay/scoring changes, concomitant treatment,
unobserved inter-visit fluctuations and clinical-indication bias. Results do
not establish prospective risk, continuous biological trajectories or causal
effects. Two observations establish neither persistence, remission nor
seroconversion. Canonical execution and clinical interpretation remain pending
the authenticated upstream products.
