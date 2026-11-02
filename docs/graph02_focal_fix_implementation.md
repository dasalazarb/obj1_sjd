# Graph02 focal correction: imputation and observed-pair graphs

This change adds fixed-panel imputation and a separate sparse-graph sensitivity.
R0/R2 definitions, Mapper settings, M mask weights (0.10/0.25/0.50), historical
controls, and Graph03's defining release reader remain unchanged. All new results
are non-defining sensitivities. R2 remains exploratory provisional.

## Baseline and execution limits

The starting code is `bff0c22a2d6a783388f8262e01c9e389a7b0d116`.
Before editing, the existing missingness variants, missingness v2, and pipeline
suite passed: **56 tests**, with 270 existing NumPy timedelta warnings.

The audited clinical run `20261029T224759222952Z_48edb990` and the clinical inputs
are absent from this workspace. Its manifest/artifact hashes and reported
159-patient, 98/67-feature panels cannot be independently reproduced here.
No historical run was modified. `graph02_focal_fix_baseline_contract.json`
records identical baseline/current hashes for eleven primary configuration,
representation, Mapper, release-contract and Graph03 files. Counts quoted in the supplied specification are
prior evidence, not new execution results.

The requested entry point was attempted without flags:

```sh
python src/studies/graph/02_graph_representation_sensitivity.py
```

It stopped at source validation with
`FileNotFoundError: Missing integrated baseline source; run the authorized upstream pipeline first`.
No clinical estimate or completed clinical release was created. In the authorized
clinical environment, rerun this command with the existing upstream inputs; it
creates a fresh immutable run. Do not change a prior run or fabricate its manifest.

## Implementation by module

- `s02_imputation.py`: continuous/ordinal/binary low-rank route, robust training
  scale fallback when IQR is zero, and diagnostics retained on unsuccessful
  fits. Rank selection uses development patients; evaluation patients cannot
  select rank. The 200-iteration budget, 1e-5 tolerance, 0.10 shrinkage, and
  candidate ranks 2/3/4 are preserved. Exhausting the budget is not convergence.
- Nominal panels route to `mixed_iterative_v1_ridge_softmax`. Each source's
  one-hot group is one softmax classification target. Continuous targets use
  standardized ridge regression; binary targets use logistic classification;
  ordinal predictions use reviewed training levels. All models, bounds,
  standardization and initialization are trained within the refit. Mask columns
  are not predictors. Missing categories must have at least three measured
  training examples per level; unsupported/constant sources remain explicit
  `NOT_ESTIMABLE`. The estimator is not missMDA::imputeMFA.
- `s02_missingness_variants.py`: dispatches I to the compatible estimator while
  preserving observed entries and the full encoded panel. D still calls
  `require_complete()`. M's implementation and mask weights are unchanged.
- `s02_pairwise_distance.py`: adds support counts/quantiles without altering
  distances, source deduplication, or the NA rejection policy.
- `s02_sparse_observed_graph.py`: implements **R0-DG/R2-DG**, not Mapper. Only
  finite, support-eligible observed pairs can become edges. The primary k is 10;
  5 and 15 are prespecified sensitivities. Union symmetrization, at least two
  eligible neighbors, minimum connected fraction 0.50, minimum largest-component
  fraction 0.50, and at least ten connected patients are fixed before outcomes.
  Louvain uses resolution 1.0 and the configured seed. Similarity `1/(1+d)`
  strictly preserves distance ranking. Unsupported/isolated patients receive no
  artificial assignment. Eligibility components and final graph components are
  reported separately. No shortest-path distance completion is used.
- DG comparisons use common supported patient assignments only and
  carry `cross_algorithm_not_mapper`. Subsampling recomputes scales, observed
  distances, eligibility, neighbors and groups from original sampled values.
  One-to-one Jaccard matching has explicit successful/evaluable denominators and
  Wilson intervals; failed refits remain in the requested denominator. DG does
  not reuse caches or enter Mapper patient-membership output.
- `s02_missingness_study.py`: retains failed imputation validation rows, adds
  attempt/rank/type/audit tables, DG registry/support/stability/comparison/refit
  tables, focal QC and a method summary. Panel lineage/order fingerprints are
  checked within refits. Refit provenance includes original-mask SHA-256, panel hash, seed and estimator
  version. Diagnostic RNG seeds distinguish the sequential split stream from the
  per-repeat model seed. Counts of successful refits include every requested
  imputation subsampling/bootstrap budget.
- `s02_reporting.py`: adds three aggregate PNG/PDF figures with source CSVs;
  estimates below publication support are suppressed. All-NA reconstruction
  errors remain NA rather than becoming zero through aggregation.
- `s02_pipeline.py`: adds estimator and sensitivity provenance to the existing
  SHA-256/privacy inventories. Graph03 and its reader are unchanged; DG cannot
  enter defining membership or eligible representations.

`graph_missingness_methods_v1` is retained. The graph configuration, mixed-model
settings and `cache_version` are additive optional keys, and historical v1
configuration without them is tested. The estimator version is
`graph02_focal_imputation_graph_v2`; cache version is `graph02_focal_v2`.
Existing signatures also hash code, source, raw values, baseline, panel, masks,
configuration and seeds, invalidating old algorithm caches.

## Validation design and interpretation

Artificial masks operate on source or instrument blocks in patients separate
from training. A 0.05 training-block mask is prespecified in YAML. Median/mode
baselines use the identical training fold and evaluation cells. Numeric panels
select ranks on inner development splits by hidden-cell-weighted MAE normalized
by training scale. Outer evaluation is reported separately. Mixed panels have no
rank selection or pooled numeric/categorical optimization criterion.

Continuous errors, reviewed ordinal-level errors, and categorical accuracy,
balanced accuracy and class support are separate fields. Empty masks yield
`n_hidden=0`, NA metrics and `NOT_ESTIMABLE`. Attempts expose iterations, relative
change, loss, penalty, scale diagnostics and source/repeat/rank reasons even when
no rank survives QC. A fit failure never silently substitutes median completion.

The synthetic matrix in `graph02_focal_fix_synthetic/acceptance.csv` covers the
six original sensitivities plus DG. Its 40-by-12 public numeric fixture has complete
pair support; therefore D is estimable there. Independent tests cover incomplete
D matrices, DG isolates, nominal panels, rare categories, nonconvergence, zero
hidden cells, fixed-panel lineage, holdout leakage, and refit-local graph support.

| Method | Audited clinical state supplied by user | Current public synthetic fixture | New clinical state |
| --- | --- | --- | --- |
| R0-I | NOT_ESTIMABLE, nominal unsupported | COMPLETE; separate nominal fixture also passes | Not run: missing source |
| R2-I | NOT_ESTIMABLE, no validated rank | COMPLETE | Not run: missing source |
| R0-D | NOT_ESTIMABLE, incomplete distances | COMPLETE on fully supported fixture; partial-matrix rejection passes | Not run: missing source |
| R2-D | NOT_ESTIMABLE, incomplete distances | COMPLETE on fully supported fixture; partial-matrix rejection passes | Not run: missing source |
| R0-DG | New method | COMPLETE, non-Mapper | Not run: missing source |
| R2-DG | New method | COMPLETE, non-Mapper | Not run: missing source |

R0-M/R2-M also complete on the fixture; original R0/R2 distances and hard
assignments are asserted unchanged. Contract tests verify Graph03 consumes only a
verified frozen release, tampering cannot replace its pointer, DG remains
non-defining, and optional v1 keys do not invalidate historical configuration.

All clinical artifacts remain `INTERNAL_RESTRICTED`, with automatic sharing
disabled. No patient tables, pair matrices, embeddings or IDs are committed.
The checked-in evidence contains **public synthetic aggregate data only**.
Topology robustness does not establish biological phenotypes, causality,
absence of MNAR bias, or temporal progression.

## Test results

The expanded regression run passed **118 tests** (272 pre-existing NumPy timedelta
warnings). The complete core missingness/release contract rerun passed **67 tests**
(270 existing warnings). After final seed/mask-provenance instrumentation, the focused
missingness and acceptance suites passed **36 tests**. Synthetic manifest and
code/fixture SHA-256 verification and `git diff --check` also passed.

## Reproduction

```sh
python -m pytest -q tests/test_graph_missingness_variants.py tests/test_graph_missingness_v2.py tests/test_graph_missingness_focal_fix.py tests/test_graph_pipeline.py
python -m pytest -q tests/test_graph_representation_sensitivity.py tests/test_graph_bootstrap_qc.py tests/test_graph_clinical_domain_mapping_v2.py tests/test_graph_coverage_diagnostics_v2.py
PYTHONPATH=. python tests/graph_focal_fix_evidence.py
```

The evidence generator uses the public fixture seed 927, the pipeline's fixed
seed, two refits per budget and two masked validation repeats. These reduced
synthetic budgets do not change production YAML budgets. It writes the complete
internal synthetic study under `work/focal_synthetic` and copies only aggregate
acceptance/figure sources into `docs/graph02_focal_fix_synthetic`. Its manifest
records code/config/fixture SHA-256, estimator/cache versions, seeds and output
hashes. No clinical source is used by this generator.

The 159-by-98 numeric low-rank acceptance benchmark passed in 5.69 seconds total
process wall time, with peak child RSS 213856 KiB (includes Python, imports and
pytest). This is a synthetic smoke benchmark, not a clinical-run performance
estimate. Production cost also includes nested validation and per-resample refits.
