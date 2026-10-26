# Step 15 performance refactor

Step 15 still consumes authenticated Step 10/11 and Step 13 v2 products. It never
reclassifies ESSDAI, regenerates canonical upstream products, imputes a value,
or changes a measurement/reference/window policy. The reference implementation is
commit `c39d8c4cd6bfb49d8fa16d4b03db6a463f9617e5`.

## Execution and memory

Previously each variable/threshold merged the entire wide integrated table with
groups, selected one Python Series per patient/date, retained all observation
frames until rendering, and recalculated summaries inside every figure. Numeric
plots created a line and scatter artist per patient.

The implementation now:

- Validates selected index dates and episode membership with grouped statistics
  and a MultiIndex, independently for every protected patient/threshold.
- Aligns context by episode keys once, compares common columns null-safely without
  repeated merges, and appends only consumed date/unit/conflict/version/protocol
  metadata. The inventory still uses exclusively the original Step 10 columns.
- Projects each variable's structural fields, measurement, date and applicable
  assay/unit/conflict metadata before the many-to-one group join. Units, scales,
  conflicts and all exclusion QC remain conditional on the current threshold's
  eligible patients. Each threshold gets its own official group map.
- Resolves same-day measurements with grouped sizes/distinct values and stable
  sorting. The first official baseline row wins even on discordant dates;
  otherwise only concordant dates retain their greatest episode ID. QC keeps
  the historical one-row-per-duplicate-date granularity and ordering.
- Precomputes raw window/protocol counts instead of rescanning all observations
  for every summary group. Pandas summary statistics and Wilson intervals retain
  their existing definitions.
- Renders each variable immediately inside an isolated temporary directory under
  the effective output root. Only aggregate tables and small plot-manifest rows
  survive the iteration. Final hashes, verified provenance, backup/rollback and
  QC-last installation still guard publication. Temporary directories are
  removed even when rendering or final verification fails.
- Uses one scatter per populated numeric panel and at most two LineCollections
  for consecutive within-patient segments (solid / two-ESSDAI-date provisional
  dashed). Figures close on save failures and partial construction failures.
  The four panels, colors, titles, points, support annotations, 13×10 inches,
  requested DPI (250 default), and tight bounding box remain intact.

Absolute/delta figures share their already-computed summaries. There is one
intentional compatibility exception: the historical **numeric first-available**
figure annotates real measurement windows, while its CSV pools first measurements
in a profile window. Those are distinct scientific summaries, so this uncommon
case prepares a separate figure summary once. Category profiles reuse the pooled
summary. Direct callers may omit the optional `summaries` argument and retain the
previous `build_figure` / `plot_variable` API.

Extraction is intentionally repeated on **narrow** tables per threshold. Sharing
cleaned clinical values or QC across thresholds would need separate proof of
eligibility equivalence. No new mandatory dependencies, persistent patient cache,
resume mode, reduced image quality, or multiprocessing were introduced.

## Progress and diagnostic metrics

All previous CLI arguments retain their meanings. Progress is enabled by default;
`--no-progress` suppresses it. Messages go to flushed stderr and identify input
validation, catalog, variable/threshold analysis, rendering, final hashes and
installation. Failure messages name the last phase/variable without patient data.

The final JSON report adds input dimensions, patient/candidate/plotted counts and
`runtime`: wall seconds, process peak RSS in MiB, phase times, Python/pandas
versions, and per-variable extraction/analysis/figure times, observation and PNG
counts. Extraction is a subset of analysis time; figure time includes PNG writing
and its hash. `resource` is optional; unavailable RSS is `not_available`.
The process RSS high-water mark includes imports and any earlier work in the same
interpreter. Persisted run manifests contain the runtime snapshot immediately
before manifest publication; the console report includes installation (temporary-directory cleanup follows report preparation).
Phase times charge work to the previously announced phase and are not a sampling
profiler. No patient IDs, individual values or dates enter diagnostics.

`--dry-run` still performs the full analysis without staging, PNGs, logs on disk,
or output directory creation. Its historical report `n_png` counts planned
figures; per-variable runtime `n_png` counts actual writes (zero in dry runs).

## Regression validation

`tests/step15_reference.py` loads the three original modules from the pinned Git
commit and wires them together without modifying production inputs. These tests
require that commit in local Git history (fetch history first in shallow CI).

The tests compare original and optimized measurements, selected episodes, dates,
references, deltas, statuses, QC, support and all three summary tables across
thresholds 3/5 and both clocks. Cases include discordant/concordant duplicate
dates, baseline priority, delayed reference, singleton, missing/invalid scale,
ordinal values, assay changes, producer conflicts, out-of-group bad units,
prebaseline and missing dates, binary observations, and cumulative known history.
Context duplicate/missing/extra keys, null-safe values and index inconsistencies
remain blocking. Step 10 and Step 11 tampering and protected Step 13 artifact
changes remain blocking.

Full synthetic bundles compare every CSV, including registry, support, numeric,
category, availability, exclusions, logical plot manifest and QC; only PNG hashes
are excluded. Existing manifest scientific/provenance fields are compared too.
Floating comparisons use `rtol=1e-10`, `atol=1e-12`, with compatible dtypes.
Figure tests compare numeric point coordinates, segment endpoints and dash styles
(including empty panels), plus category heights, confidence-interval segments,
point coordinates, titles and support text for both thresholds. They cover
absolute/delta, binary, ordinal, categorical and first-available plots. PNG binary
identity is deliberately not required.

Failure injection verifies partial rendering and final-hash failures leave prior
bundles byte-identical and remove staging. Existing tests also verify no-overwrite,
backup/rollback, independent threshold installation, output privacy and upstream
read-only behavior. Run with an existing project environment:

```bash
python -m pytest tests/test_secondary_trajectories_by_essdai_pattern.py \
  tests/test_secondary_trajectory_performance.py tests/test_observed_patterns_v2.py -q
```

The extended Step 13 test run has one pre-existing failure:
`test_v1_endpoints_outputs_and_step14_source_are_unchanged` compares Step 10 bytes
against `e96f18955cc5d41208fdb3aa66122b20f25a249a`. Step 10 had already changed
before the reference commit used here. Its bytes match the pre-refactor commit
exactly; this refactor does not edit that producer or relax the legacy test.
The other 66 upstream tests passed in that run. Upstream snapshot maintenance is
outside this Step 15 change.

Validation result: **70 Step 15 tests passed** (69 in the complete Step 15
run, plus the added null-protocol regression; its seven related QC edge cases
also passed again). The extended upstream run had 66 passes and the single
pre-existing snapshot failure described above. Python 3.10 syntax was checked;
execution measurements and tests used Python 3.12.

## Reproducible synthetic benchmark

```bash
python tests/benchmark_secondary_trajectories.py --root work/step15-benchmark-new
```

Choose a new root for each run. The harness builds nonidentifying synthetic
upstream products under that root, authenticates them through the original Step
13 producer, then measures old/new Step 15 in fresh subprocesses. Upstream fixture
construction, imports and historical-module loading are outside the wall timer;
peak RSS covers the whole worker. Every scenario uses the same input bytes and
hardware; PNG scenarios use 250 DPI for both versions. Results are written to
`results.csv`, with per-worker JSON. Published PNG scenarios compare **all** CSVs
except image hashes; dry runs compare report metadata and rely on the full
regression comparisons for scientific tables.

The fixture has 250 synthetic patients, 3,000 episodes (12 visits per patient),
228 integrated columns, 17 eligible clinical variables, and 19 PNGs per threshold.
It intentionally has many irrelevant columns. The broader regression fixture
covers all four ESSDAI groups; the benchmark itself occupies stable/intermittent
groups after repeating observations.

Environment: Linux x86_64, Python 3.12.14, pandas 2.3.3, NumPy 2.5.3,
Matplotlib 3.11.2, PyArrow 25.0.1. These are single-run cloud measurements;
other validation processes ran on the host during parts of this measurement.
Host-load/cache noise and synthetic shape limit extrapolation. Re-run with a quiet
host and repetitions for a production capacity decision.

| Scenario | Old wall (s) | New wall (s) | Speedup | Old peak RSS (MiB) | New peak RSS (MiB) | Scientific comparison |
|---|---:|---:|---:|---:|---:|---|
| `delta3_three_dry` | 7.35 | 0.93 | 7.93× | 280.3 | 261.3 | report metadata; full tables checked by regression tests |
| `delta3_full_dry` | 83.34 | 1.82 | 45.77× | 371.0 | 261.3 | report metadata; full tables checked by regression tests |
| `both_full_dry` | 197.99 | 2.60 | 76.19× | 467.3 | 261.3 | report metadata; full tables checked by regression tests |
| `delta3_numeric_png` | 15.44 | 2.17 | 7.10× | 339.2 | 315.3 | yes (8 CSVs) |
| `both_full_png` | 349.09 | 53.12 | 6.57× | 653.2 | 451.3 | yes (16 CSVs) |

The full two-threshold PNG run wrote 38 figures: 349.09 → 53.12 seconds (6.57×); peak RSS 653.2 → 451.3 MiB. All 16 CSVs (including QC, with PNG hashes excluded) matched at the stated tolerances. The residual runtime is mostly figure layout and PNG encoding.

Canonical/Biowulf performance is **not measured**. Rendering and 250-DPI PNG
encoding remain a material cost once analysis becomes inexpensive. No synthetic
speedup is a promise for the real cohort. No production Step 10/11/13 inputs were
regenerated, and no canonical Step 15 bundle was published as part of this change.

For Biowulf, use the existing authenticated products and unchanged commands:

```bash
PY=/usr/local/Anaconda/envs/py3.10/bin/python
/usr/bin/time -v "$PY" -u src/block_A/15_secondary_trajectories_by_essdai_pattern.py \
  --threshold both --dry-run
# Full PNG production, subject to normal no-overwrite protections:
"$PY" -u src/block_A/15_secondary_trajectories_by_essdai_pattern.py --threshold both
```

An existing final bundle still requires deliberate `--overwrite --backup-root`
inside its dedicated Step 15 backup namespace. Benchmark render runs should use
an isolated output root; synthetic products retain `synthetic_validation` origin
and cannot be presented as canonical evidence.
