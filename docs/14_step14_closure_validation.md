# Step 14 closure implementation and validation

The closure implementation adds future ESSDAI excluding the source domain,
matched comparisons, an explicit treatment decision, a unique-episode
composition audit, and a scientific code fingerprint. Primary estimands,
predictors, temporal risk sets, adjustment hierarchy and support thresholds
remain unchanged. The real-cohort analysis is **not closed** in this environment:
canonical Step 10/11 inputs and the previous productive outputs are absent.

## Files and reasons

- `src/block_A/14_risk_factors_progression.py`: one parameterized FROM/TO
  composition function; fail-closed complete-panel concordance; raw contribution
  audit; the Gaussian GEE leave-source outcome; matched primary and prior-state
  comparisons with fixed complete cases and demographic level; source-backed
  treatment decisions; metadata, SHA-256 manifests and conditional closure.
- `tests/test_risk_factors_progression.py`: valid and rejected TO subtraction,
  partial/discordant panels, exact matched keys/hash/adjustment, FDR exclusion,
  wrong-episode leakage, unique discordant episodes, treatment source/timing,
  provenance preservation, environment exclusions and conditional closure.
- `.gitignore`: generated analytic products, outputs, scratch work and runtime
  environments cannot make an otherwise committed source tree dirty.
- `docs/14_risk_factors_progression.md`: current formulas, output contracts,
  provenance/treatment policies and reproducibility instructions.
- `docs/14_step14_refactor_validation.md`: retains the earlier validation while
  identifying the subsequent closure implementation.

## Composition and upstream decision

The producer code identifies recorded `essdai__total` as the Pop producer's
`essdai_total` (`01_pop_distribution.py`), derived from the canonical clinical
episode's recorded total. The 12 ordinal scores are recoded by
`06_overlap_glandular.py` and `src/derivations/overlap_flags.py`, then mapped by
Step 10. These independent outputs do not certify same-version composition.
No available canonical row-level evidence establishes an upstream bug.
**No upstream fix is applied, no producer is rerun, and no cohort is changed.**

The conservative decision is to preserve recorded totals and domains and treat
discordance as a source/provenance inconsistency. Both sides of a composition
sensitivity require all 12 evaluable ordinal scores, total agreement to absolute
tolerance 1e-8, and a nonnegative target subtraction. Partial panels remain
missing. Negative values are not clipped and totals are not replaced by sums.
The runtime producer registry and available context provenance fields are
retained; missing versions are never invented. A generic ESSDAI version field
does not certify a domain version. There is no partial-panel composition
certification contract in the current upstream producers.

The discordant-episode CSV covers unique canonical episodes, including TO-only
endpoints. Metadata separately gives the unique FROM snapshot comparable with
the request, and the all-episode snapshot. Summary strata include composition,
exact/absolute deltas, visit, available protocol, year, activity state, active
domains, Articular, negative subtraction and available source/version evidence.

## Beyond-self-domain sensitivity and matched comparisons

For each ESSDAI domain D:

```text
TO_total - canonical_weight[D] * TO_ordinal_D
    ~ FROM_ordinal_D
    + FROM_total - canonical_weight[D] * FROM_ordinal_D
    + interval_years + prespecified demographic adjustments
```

The predictor remains the unweighted ordinal score observed at FROM. TO only
supplies the outcome and its composition eligibility. The new sensitivity is
named `leave_source_domain_outcome`, has `claim_status = sensitivity` and
`fdr_eligibility = false`, and receives no family or primary outcome-wide BH.
Existing primary family/outcome-wide corrections remain separate and unchanged.

The comparison CSV has primary full sample, primary matched sample, prior
other-domains-adjusted matched sample, and leave-source outcome matched sample.
The last three use exactly the same patient/interval keys and sample hash.
The adjustment level is selected once before fitting and fixed for all three;
numerical failure does not reselect covariates. Empty/unsupported comparisons
retain their counts and support reasons. Interpretation concerns subsequent
activity outside the source domain, not causal effects or incident organs.

## Treatment and duration

The current Step 10/11 producer contracts do not define a registered, temporally
verified systemic treatment exposure. Step 14 now audits the provided integrated
and official-baseline schemas and variable registry at runtime. A name or an
ever-treated flag cannot become a covariate. If no valid exposure is available,
the decision CSV and structured metadata record `not_implemented`, sources,
candidate columns, temporal scope, coverage, reason and protocol consistency
under "as appropriate". No ad hoc treatment derivation is performed.

A future upstream product can be used only when it registers the canonical
`systemic_treatment_exposure` concept, producer, baseline/as-of temporal scope,
binary 0/1 exposure and a `__measurement_date` at/before the canonical episode.
It enters treatment-adjusted sensitivities only; primary models stay unchanged.
Unknown/future dates remain missing. Constant or sparse treatment exposure
cannot silently disappear from a supposedly treatment-adjusted model.
This is an adjusted observational association, not confounding elimination.

Disease duration is not derived or repaired here. Its availability/exclusion
decision is recorded from the official baseline, preserving age/sex and the
existing deterministic hierarchy when duration is absent.

## Validation

- Step 14: **68 tests**, including new closure contracts.
- Relevant suite: **205 passed** across `test_pop_transitions_episode_contract`,
  `test_overlap_tristate`, `test_integrated_longitudinal_dataset`,
  `test_integrated_baseline_characterization`, `test_disease_activity_progression`
  and `test_risk_factors_progression`.
- Four targeted closure/provenance tests also passed independently.
- A full synthetic CLI run (40 patients / 160 episodes, explicitly labeled
  `synthetic_validation`) completed all 1,396 planned associations, including
  12 leave-source models with four supported valid fits. Structural and temporal
  QC passed with zero leakage violations. All 40 artifact hashes verified;
  matched comparison keys/counts/hashes agreed and all new sensitivity BH
  values were missing as specified. Its real-cohort closure remained conditional.
- `git diff --check` passed.
- Differential validation on explicitly synthetic inputs (40 patients, 160
  episodes) compared **all 1,031 planned primary task datasets** to the original
  commit. Predictor/outcome/state, interval/event keys, baseline adjustments and
  sample data were identical. Selected Biological/Low-C4 trajectory, lagged and
  event model support/specifications and fitted estimates/CIs/p-values were
  exactly unchanged. Synthetic values are not scientific results.
- Full `pytest -q` could not collect two pre-existing tests: they import absent
  `src/20_btris_visit_date_match_report.py` and
  `src/studies/pharma/01_run_pharma.py`.
- With those two collection errors excluded, the wider suite had **411 passed,
  1 skipped, 10 failed**. All ten failing tests were rerun in an isolated checkout
  of original commit `0c2130bd5523dab919211a5161008b1c7ef2cade` and failed there too.
  They concern comorbidities, cross-domain correlations, missingness, Pharma,
  PRO labels and visit-date fragments, not changed Step 14 files. No unrelated
  test repair is included in this closure change.

## Productive before/after snapshot

The "before" values below are the **user-supplied reference**, not independently
verified local results. The canonical inputs are missing, so no real after
snapshot or Biological leave-source estimate can be reported or forced.

| Metric | Supplied reference | Real after-run |
| --- | ---: | --- |
| Baseline patients | 159 | Pending canonical inputs |
| Episodes | 497 | Pending canonical inputs |
| Complete / concordant / discordant unique FROM panels | 304 / 241 / 63 | Pending canonical inputs |
| Articular negative subtractions | 48 | Pending canonical inputs |
| Valid primary models | 105 | Pending canonical inputs |
| Failed primary Cox | 6 | Pending canonical inputs |
| Primary family q < .05 | 4 | Pending canonical inputs |
| Primary outcome-wide q < .05 | 2 | Pending canonical inputs |
| Supported primary trajectories / fallbacks | 17 / 17 | Pending canonical inputs |
| Biological future leave-source estimate | Not previously implemented | Pending canonical inputs |

The audit and model outputs regenerate these values from inputs. Reference
counts and effects are never hardcoded into analytical decisions.

## Reproducibility and remaining closure step

The fingerprint keeps committed source/config, tests and dashboard code. It
prunes virtual environments, site-packages, vendors, caches, data/outputs and
work/build directories. It records commit/branch, tracked diff, untracked code,
per-file/code-tree hashes, input hashes and package versions. The output manifest
hashes artifacts except itself and the open log.

After tests and a code commit, rerun the full script using the real Step 10/11
products, optional context/registry and verified Step 13 product. Review temporal
QC, real composition discordances, treatment decision, model failures, both
primary corrections and Biological matched results. Inputs, clean code commit
and generated output hashes must agree. No code edits may follow that final run
without another rerun. Synthetic validation explicitly uses
`--input-origin synthetic_validation` and cannot produce real-cohort PASS.

The generated closure summary is:
`outputs/qc/blockA/14_risk_factors_progression/14_closure_summary.md`.
Missing required inputs write a conditional summary/metadata and fail loudly.
Until the canonical run is reviewed, **do not claim "Step 14 is closed"**.

STEP 14 CLOSURE STATUS: CONDITIONAL
