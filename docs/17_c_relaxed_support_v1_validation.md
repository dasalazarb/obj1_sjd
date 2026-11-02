# Step 17 C_RELAXED_SUPPORT_V1 — software validation

Date: 2026-10-29. Specification: `temporal-spline-v2`.
Analysis role: `exploratory_posthoc_support_revision`.

## Automated regression

```bash
python -m pytest tests/test_temporal_spline_*.py tests/test_secondary_comparison_*.py -q
```

Executed in the implementation environment: **88 passed**, 64 warnings.
The new revision tests cover metadata collision errors and semantic aliases,
89/19/27/15 group counts with total 150, suppression and complementary totals,
canonical homonyms in classification and sensitivity matching, identical A/B
coefficients/covariances/SEs/tests on identical rows and configuration, C gates,
rank/condition failures, invalid covariance/nonconvergence, failed B preventing C,
fixed policy parameters, version isolation and backup guards. Existing end-to-end
checks authenticate synthetic inputs, publish bundles/figures, verify hashes,
forbid identity exports, preserve upstream artifacts and exercise no-flag defaults.

## Aggregate synthetic support demonstration

One explicitly synthetic numeric outcome was evaluated independently under both
threshold labels, using the same fixed support policy. There are 26 synthetic
patients per group, each with exactly two dates: **zero** patients per group meet
the three-date diagnostic. These labels are software fixtures and do not represent
clinical ESSDAI threshold memberships or clinical results. Estimation used the
normal ML/GEE selection path, without forcing an estimator.

| Threshold label | Model | Eligible | Estimated | Failed | Selected estimator |
| --- | --- | ---: | ---: | ---: | --- |
| ±3 | A | 1 | 1 | 0 | MixedLM ML, patient random intercept |
| ±3 | B | 1 | 1 | 0 | MixedLM ML, patient random intercept |
| ±3 | C | 1 | 1 | 0 | MixedLM ML, patient random intercept |
| ±5 | A | 1 | 1 | 0 | MixedLM ML, patient random intercept |
| ±5 | B | 1 | 1 | 0 | MixedLM ML, patient random intercept |
| ±5 | C | 1 | 1 | 0 | MixedLM ML, patient random intercept |

Each fixture has 13 simultaneously supported grid times, span 3.0 years, three
regions and four annual bins (the inclusive endpoint at year 3 has its own bin).
Both C tests pass software estimation QC per fixture. No p/q are interpreted as
clinical findings. Under V1 these fixtures would fail the mandatory five
three-date-patient gate; V2 retains the unchanged A/B and matrix/fit QC and permits
C. Separate tests demonstrate valid C with only two regions/two annual windows,
and reject fewer than four supported times or supported spans below one year.

## Institutional execution status

The canonical prerequisite check was executed with `--dry-run --no-progress` and
stopped before analysis with:

```text
BLOCKED_UPSTREAM: missing canonical Step 10/11/13 files
```

This environment does not contain the authorized institutional inputs. No clinical
models, updated institutional CSVs, old-versus-new cohort comparison or real-data
QC PASS are claimed. No historical outputs or upstream producers were modified.
The institutional rerun remains pending, using the frozen cutoff **2026-05-31**
and a new output root. The normal no-flag command remains unchanged; its outputs
now include the `temporal-spline-v2` namespace. Each successful institutional
bundle will record actual A/B/C eligibility/fit/failure counts, reasons,
estimators, valid C tests and shared temporal coverage in `run_manifest.json`.
