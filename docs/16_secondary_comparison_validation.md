# Step 16 implementation validation — 2026-10-26

This is software validation using fictional patients. It is not a report of NIH-cohort findings or clinical review.

## Checks performed

- Combined Step 16 + Step 15 measurement/figure/performance regression: **108 passed, 0 failed, 0 skipped** (199.03 seconds).
- After adding a final protection against mutation between manifest verification and hash capture, all Step 16 tests were rerun: **37 passed, 0 failed, 0 skipped** (81.00 seconds). This includes the new artifact-race test and both complete synthetic threshold bundles. Across both runs, 109 distinct tests were exercised.
- Dependency warnings remain (NumPy/pandas timedelta deprecations and empty-series summary warnings); no failed model is promoted because of them. Model-fitting warnings are independently captured and tested by the model eligibility logic.
- CLI help, Python compilation and `git diff --check` passed.
- Canonical `--analysis feasibility --dry-run` correctly stopped with `BLOCKED_UPSTREAM: missing canonical Step 10/11/13 files`. No canonical result bundle was created.
- Source changes add Step 16, its documentation/configuration and tests. No Step 10/11/13/15 source, scientific output or cohort definition is changed relative to the updated `origin/main` base.

## Synthetic validation evidence

The small authenticated v2 fixture checks independent threshold denominators, late laboratory references, missing binaries, cumulative serology, nonestimable cohorts, no-write feasibility, hash tampering and output transaction failure/rollback.

A separate fictional 60-patient cohort has 15 patients in each of the four producer-assigned groups, with three dated observations per patient and known secondary-outcome slopes. The real v2 producer is used **only in tests** to create authenticated synthetic inputs. Step 16 reads them without upstream writes. Both ±3 and ±5 full bundles recover slope signs and magnitudes within the stated test tolerance, produce valid multiplicity denominators, render marginal-trajectory/forest/availability/effect figures and verify all artifact hashes. Original fixture input hashes remain unchanged. Rendered trajectory and forest figures were visually inspected; synthetic validation is explicitly labeled in the figures.

Other tests check unequal numbers of visits without replacing patient clusters by independent rows, zero interaction, Gaussian GEE fallback and its attempt audit, logistic probabilities and separation, constant/sparse/unsupported outcomes, reference-zero exclusion, singleton exclusion, preindex support recalculation, common-horizon AUC eligibility, valid-only separated FDR families, canonical source keys through symlink aliases, identifier rejection, collision prevention, backup retention, export failure, installation rollback and independent threshold preservation.

The generated `00_variable_feasibility.csv` and its gate are inspected by assertions before supported model validation. Sparse fixtures remain `NOT_ESTIMABLE`; the fictional supported cohort is eligible. No synthetic outcome or inferential table is committed as a real clinical finding.

## Limits requiring real-cohort work

The protected canonical cohort was unavailable. Real group support, measurement-opportunity denominators, clinically approved outcomes/covariates, assay comparability, model stability, informative visits, clinical relevance and institutional disclosure controls could not be validated. No real biomarker differences, p-values or clinical conclusions are asserted.

Ordinal/nominal longitudinal inference, approved covariate adjustment, flexible time effects, transformations and visit-intensity weighting are explicitly unsupported until prespecified/validated. They are not silently replaced with an inappropriate model. Aggregate bundles are restricted internal preliminary products; small cells require institutional review before circulation.

See [methodology and CLI](16_compare_secondary_trajectories.md) and the [versioned configuration](16_secondary_comparison_config.json) for estimands, exact gates, output namespaces and operational instructions.
