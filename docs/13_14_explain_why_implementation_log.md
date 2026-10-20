# Explain-why implementation log

## Audited state and scope

- Source `main`: `06b492c2904f59d8ce557e61e186593a90cafd38`, independently
  verified using GitHub on 2026-10-19 (America/New_York). Local branch was clean
  `work`; implementation started on `codex/step13-step14-explain-why`.
- Shell Git fetch could not connect to its configured proxy. The GitHub
  integration established the current main SHA instead of trusting cached
  `origin/main`. Publication uses the same authorized integration if needed.
- Reviewed README/config, Steps 10/11/13/14 contracts, Step 13 absolute-change
  guide, Step 14 analysis and closure guides, targeted tests, and PRO/extended
  phenotype producers. No AGENTS.md applies to the scientific files; dashboard
  instructions are scoped to dashboard and no dashboard file is changed.
- Canonical Step 10 integrated/context/registry, Step 11 baseline and Step 13
  products are absent. No prior productive estimates/QC exist for comparison.
  The attempted pre-change dry runs created only missing-input logs/QC, not
  clinical results. The original Step 14 closure is therefore still conditional.
- Before edits, scientific source/contract hashes, packages, clean status and
  targeted tests were frozen in ignored `work/explainwhy/`. Input availability,
  code/package/output hashes are also reproducible through the new audit mode.
  The compact versioned validation record contains no patient information.

## Changes

| File | Purpose |
| --- | --- |
| `src/block_A/observed_patterns.py` | Isolated versioned shape classifier, tie sensitivity, Pop verification, summaries, aggregate QC/figure and output hashes |
| `src/block_A/explainwhy_feasibility.py` | Isolated read-only feasibility/availability audit with backups, unknown counts and G0–G4 decisions |
| `src/block_A/13_disease_activity_progression.py` | Add writer call and input-origin flag; preserve previous functions |
| `src/block_A/14_risk_factors_progression.py` | Explicit audit-only branch before primary run; preserve prior workflow |
| `tests/test_explainwhy_extension.py` | Synthetic edge/temporal/privacy/output/manifest/audit and AST regression tests |
| `tests/test_disease_activity_progression.py` | Extend output inventory assertions to the new aggregate PDF and patient-pattern product |
| `README.md` | Link entrypoint and protocol |
| `docs/13_14_explain_why_extension.md` | Scientific protocol, definitions, outputs, gates, limitations and exact reproduction |
| `docs/13_14_explain_why_validation.json` | Sanitized source/dependency/availability/validation evidence |

Steps 10/11, original endpoint functions, all Step 14 model/task/FDR/adjustment
functions, config thresholds, pharma, graph and longitudinal_graph are unchanged.
The AST regression compares every pre-existing scientific function/class except
CLI parsing and main with the audited original main. No original assertion is
weakened; the Step 13 integration output inventory is extended for added outputs.

## Validation and decisions

The original focused Step 10/11/13/14 suite produced **200 passed, 2 failed**
before any implementation edits. Both failures are existing Step 14 assertions:

- `test_weighted_other_domain_adjustments_aliases_missing_and_inconsistent_panels`
- `test_to_negative_subtraction_and_discordance_fail_closed_without_clipping`

They expect raw Articular contribution 8; the configured ordinal score/weight
produces 4. Both failures are recorded in pre-change output and prior Step 13
documentation. No clinical weights, expectations or unrelated model logic were
changed to make them pass. Dependency deprecation warnings also remain.

New tests cover 0→4→0, 0→4→6, 5→1→5, 5→9→1, within band, ±3/±5 boundary
differences, reentry, plateaus, same-date discordance, missing/official baseline,
strictly later dates, insufficient and 4+ dates, floor, Low/Moderate/High/Pop
denominators, unknown Pop, no biomarker input to shape, future/missing lab dates,
mixed/missing units, registry/instrument requirements, original adjacency,
unknown treatment, manifests/privacy, dry-run/no-overwrite, backup and stale
support removal. No clinical counts or sample-level PHI enter documentation.

Final test counts and verification status are recorded in
`13_14_explain_why_validation.json`. The Step 13 integration exercises the old
and new outputs together. The new aggregate PDF is checked for valid rendering.
Full-suite collection limitations are recorded separately from focused checks.

G0 is BLOCKED_UPSTREAM. E0/E1 are IMPLEMENTED and TESTED_SYNTHETIC; nothing is
EXECUTED_CANONICAL. E2 models are NOT_IMPLEMENTED because their real feasibility
gate failed; no clinical model was attempted or labeled NOT_ESTIMABLE. Treatment,
ocular/salivary and PRO inference remain blocked pending upstream evidence;
no absent variable becomes a zero. Molecular implementation remains out of scope.
Real class support, temporal/composition QC, scientific and institutional
disclosure reviews are PENDING_CLINICAL_REVIEW.

The implementation commit is the Git commit containing this log and the listed
files (`git log -1 -- docs/13_14_explain_why_implementation_log.md`). A run's
manifests record its exact code commit and source hashes without a circular
self-reference in this document.
