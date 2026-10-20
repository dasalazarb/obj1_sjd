# Steps 13/14: observed patterns and explain-why feasibility

This implements E0 (audit/freeze) and E1 (descriptive patterns) of the supplied
2026-10-19 specification. The audited live `main` was
`06b492c2904f59d8ce557e61e186593a90cafd38`, verified through GitHub; the local
`origin/main` cache was older and was **not** used as the source of truth.
Implementation branch: `codex/step13-step14-explain-why`.

**PENDING / NOT RUN — canonical inputs unavailable.** No real-cohort estimate,
class size, event count, association, or clinical closure is claimed. The
researcher's example counts are not constants or QC targets.

## Scientific question and estimands

Step 13 asks which threshold excursions and subsequent reversions were observed
relative to the official baseline. Its unit is a patient, separately for ±3 and
±5 points of recorded `essdai__total`. Level 1 remains the existing baseline
plus at least one strictly later valid assessment. All existing ever/last flags,
first documented dates, denominators, category transitions and +5 downstream
sensitivity remain unchanged.

Level 2 retains that same patient universe but assigns `insufficient_observations`
unless there are at least two **distinct follow-up dates**, in addition to the
official baseline. Missing baseline is never replaced. Same-day baseline
companions never become follow-up. Missing scores are skipped for patterns,
without changing the adjacency rules of previous analyses.

For patterns only, sort valid follow-up by actual date and canonical episode ID
and select the last episode on each date. The official baseline remains fixed.
Audit dates with discordant scores and repeat classification after excluding
those follow-up dates entirely. Exclusion retains the Level-1 patient universe;
even patients left with baseline alone remain in the sensitivity as insufficient,
with missing final delta and `directionality_at_last=not_evaluable`.

With d(t)=ESSDAI(t)−baseline and k=3 or 5, labels have this precedence:

1. `insufficient_observations`: fewer than three distinct valid dates total.
2. `within_threshold_observed`: neither d≥k nor d≤−k was observed.
3. `intermittent_observed`: both directions were observed, or a threshold was
   crossed and a **later distinct date** no longer met that threshold.
4. `worsening_threshold_observed_no_reversion`: worsening only, with no later
   documented loss of its threshold.
5. `improvement_threshold_observed_no_reversion`: improvement only, with no
   later documented loss of its threshold.

The version is `observed-essdai-pattern-v1`. The patient product is long format:
one row per patient × threshold. `observed_pattern` and `threshold_abs` together
encode the requested k-specific label. Nonexclusive excursion/reversion flags,
first selected-date excursions, final direction, delta extrema and distance,
follow-up years, retained canonical point counts and distinct-date counts are
also exported. `n_points` includes baseline and all valid retained canonical
follow-up episodes; `n_dates` counts the actual points used for pattern shape.
Boundary crossing counts include entries and exits, starting from baseline
inside the band. A direct +k to −k transition changes two boundary indicators.
Direction changes count changes in the sign of successive score differences,
ignoring zero differences. These are lower bounds under intermittent observation.

The new selected-date `first_up_date` is a pattern descriptor. The original
`first_observed_worsening_ge3_date` considers **all** canonical episodes and
remains the source for a future Step 14 event analysis. Date-tie discrepancies
in ever flags are explicitly counted in aggregate QC. Last flags must agree.
Events at ±5 imply events at ±3; pattern labels need not nest.

No biomarker or treatment participates in classification. Complete-history
labels must never become FROM predictors. No reversion documented does not
establish persistence, irreversible damage or continuous progression. Within
threshold does not establish absence of biological change.

## Descriptive outputs and QC

The standard Step 13 run adds the following products beneath its existing stem
`13_disease_activity_progression`:

| Area | New products |
| --- | --- |
| `data/analytic/blockA/` | `13_essdai_patient_observed_patterns.parquet` (controlled) |
| `outputs/tables/blockA/` | `13_essdai_observed_pattern_summary.csv`, `13_essdai_pattern_by_baseline_state.csv`, `13_essdai_pattern_by_pop.csv` when verified, `13_essdai_pattern_followup_sensitivity.csv` |
| `outputs/qc/blockA/` | `13_essdai_observed_pattern_qc.csv`, `13_essdai_observed_pattern_manifest.json` |
| `outputs/figures/blockA/` | `13_essdai_observed_pattern_aggregate.pdf` |

All tables retain empty/insufficient classes. Summary percentages use the full
Level-1 denominator and additionally report `n_pattern_eligible`. Sensitivities
report 3+ and 4+ dates, duration <2 / 2–<5 / ≥5 years, and discordant-date
exclusion, separately overall and by baseline Low/Moderate/High. These are
post-baseline observation strata, not standardized fixed-horizon risks.
`can_improve`/`can_worsen` use baseline floor and the existing validation ceiling;
the previous mathematical-opportunity tables remain unchanged. High alone does
not imply a mathematical ceiling.

Pop is consumed only from official Step 11 rows. Classified Pop1/2/3 must agree
with recorded baseline ESSDAI/ESSPRI under existing cutoffs. Unknown or
unclassifiable is retained; it never becomes Pop3. An unavailable Pop product
is recorded in the manifest and no Pop table is published. A stale Pop table
from a previous overwritten run is removed.

QC checks unique patient/threshold keys, labels, date eligibility, reconstruction,
floor/ceiling, nested excursions, ever discrepancies explained by discordant
ties, last agreement and summary partition. QC is saved before publication;
failed checks prevent new patient/summary/figure publication. The existing
`--dry-run` and `--no-overwrite` guards protect new products. Step 13 preserves
its established overwrite behavior; no new Step 13 backup policy is introduced.
The new manifest hashes every new product, input DataFrames, definition code,
Step 13/config source and records input origin and review status.

## Isolated Step 14 feasibility entrypoint

```bash
python src/block_A/14_risk_factors_progression.py --explainwhy-audit-only
```

This audit never builds or fits model tasks and does not overwrite primary
Step 14 outputs. Its own repeated outputs require `--overwrite` (the existing
CLI default) and are copied first to a timestamped `outputs/backups/` bundle.
`--no-overwrite` refuses an existing audit bundle before any writes. The
pre-run snapshot fingerprints source, packages, input availability/hashes and
pre-existing Step 13/14 products. Missing input counts are unknown, never zero.
Missing required canonical inputs or rejected contracts return exit code 2
**after** publishing structured audit outputs. Available valid inputs return a
descriptive `PARTIAL`, which does not approve additional models.

New outputs under the Step 14 tables/QC stem have prefix `14_explainwhy_`:
`prechange_snapshot.md`, `input_availability.csv`, `modality_feasibility.csv`,
`outcome_support.csv`, `feasibility_decisions.csv`, `treatment_readiness.csv`,
`manifest.json`; with valid inputs, also `pattern_support.csv` and
`domain_composition_support.csv`. Failed subsequent audits remove stale support
products after backup. No row-level composition audit is exported by this mode.

The modality matrix includes labs (CRP and hsCRP separate), serology, all ESSDAI
domains/total, glandular/ocular phenotypes, ESSPRI/components, SF-36,
PROFAD/MDAFS and treatment. It separates column presence, schema values and
evaluable values, baseline patients, patients with repeated dates, valid episodes,
patient-date pairs, original adjacent lagged intervals, baseline-predictor
coverage among Δ3/Δ5 events, missingness, unknown timing/units, incompatible
units and paired ESSDAI episodes. `n_distinct_dates` means distinct
**patient × date** pairs, not global calendar dates. Baseline activity and
verified Pop strata are separate, with unclassifiable strata retained.

Existing predictor evaluability is consumed. For this audit, labs additionally
require explicit measurement dates at/before the episode, known consistent
units, and unique registry producer/scope. No unit conversion is invented;
multiple units block that continuous series. Original adjacent intervals are
formed before availability filtering, require valid FROM/TO ESSDAI, and never
bridge a missing middle episode. PRO instrument versions/validity are required
for non-ESSPRI instruments; absent registry-only version evidence remains
unavailable rather than inferred. The audit-only context join preserves
available scoring versions/validity and lab selection/conflict metadata without
altering the primary Step 14 join. The existing complete-panel composition
audit is reused and only category counts are exported; arithmetic agreement
does not certify same-version provenance or permit subtraction by itself.

Registry/measurement counts establish descriptive availability only. They are
not complete-case model support, PH checks or sufficient repeated-pair support.
No statistical significance or data-driven class merging is used for decisions.

## Gates and remaining upstream work

| Gate | State in this environment | Required next evidence |
| --- | --- | --- |
| G0 | BLOCKED_UPSTREAM | Authorized Step 10 integrated/context/registry and Step 11 baseline; matching hashes/contracts |
| G1 | TESTED_SYNTHETIC; real review pending | Run Step 13 on canonical inputs, review class support and QC |
| G2 | BLOCKED; NOT_IMPLEMENTED | Real Δ3 events and predictor-specific complete cases under existing support hierarchy, PH and separate FDR plan |
| G3 | BLOCKED; treatment adjustment unchanged | Registered dated upstream exposure, indication, start/stop, dose, external treatment and unknown-state coverage |
| G4 | BLOCKED | Real glandular/ocular/PRO pairs, assay/units/instrument/version/date comparability |
| Molecular work | NOT_IMPLEMENTED; out of scope | Biobank inventory, consent, sample timing, funding and scientific protocol |

E2 is intentionally not implemented because its prerequisite failed. No new
Δ3 Cox/logit/model family, pattern inference, treatment covariate or parallel
lagged model was added. Existing tasks, model formulas, support thresholds,
adjustment order, FDR families and treatment `not_implemented` semantics are
unchanged. NOT_ESTIMABLE is reserved for supported-input model attempts; there
were no new clinical model attempts here. Missing inputs are NOT_RUN instead.

A future treatment producer must publish reviewed `medication_exposure_intervals`
with start/stop, indication, source evidence, external-use coverage and conflict
policy. Upstream Step 10 must expose a registered `systemic_treatment_exposure`
concept as baseline/as-of binary or NA with producer and measurement date. The
existing Step 14 consumer decides whether a sensitivity is permissible. Unknown,
documented absence, documented exposure and not evaluable must remain distinct.
No EHR extraction, patient list or retrospective medication parsing is added.

## Reproduction on authorized canonical inputs

Use the repository's configured Python environment. First produce Step 10/11
under their existing contracts; this ticket does not regenerate upstream data.
Then run these commands from the repository root:

```bash
python -m pytest -q tests/test_explainwhy_extension.py
python -m pytest -q tests/test_disease_activity_progression.py tests/test_risk_factors_progression.py tests/test_integrated_longitudinal_dataset.py tests/test_integrated_baseline_characterization.py
python src/block_A/13_disease_activity_progression.py --help
python src/block_A/14_risk_factors_progression.py --help
python src/block_A/13_disease_activity_progression.py --dry-run
python src/block_A/14_risk_factors_progression.py --dry-run
python src/block_A/14_risk_factors_progression.py --explainwhy-audit-only --no-overwrite
python src/block_A/13_disease_activity_progression.py --no-overwrite --input-origin canonical
```

On a reviewed rerun, use `--overwrite`; Step 14 audit backs up its prior bundle,
while Step 13 retains its documented default regeneration policy. For external
authorized storage use `--integrated`, `--baseline`, `--context`,
`--variable-registry`, `--progression` and `--output-root`; context/registry
default beside the integrated file. The audit checks the Step 13 patient-delta
product under `--output-root` if available, comparing it to the imported Step 13
definition. Execute the audit again after Step 13 to fingerprint its new products.
Test fixtures and validation runs must use `--input-origin synthetic_validation`.
No real PHI is needed to run the tests.

Review sampling, irregular follow-up, informative observation, measurement
variation, regression to the mean, low-baseline floor and treatment confounding.
Review of canonical results, hashes, estimability and disclosure remains
**PENDING_CLINICAL_REVIEW**. Neither synthetic PASS nor an audit's PARTIAL is
clinical closure. Individual Parquets belong only in authorized ignored storage;
aggregate sparse labels are not permission for external disclosure.
