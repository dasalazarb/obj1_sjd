# Serology OHE verification record — 2026-10-13

Base code: `527c678` (the implementation is recorded by the commit containing
this file). Workspace inputs were unavailable; no real study numbers are reported.
The Step-10 dataset and frozen historical reference were absent. Production
mapping `serology_ohe_mapping_v1` remains empty and unapproved. The configuration
seed is 20260831. Synthetic integration seed is 10; test-only bootstrap budget
is two; production S4 bootstrap enforces at least 200 per scenario.

Verified commands:

```text
python -m pytest tests/test_serology_ohe.py tests/test_longitudinal_graph.py tests/test_longitudinal_mapper_cli.py tests/test_longitudinal_methodology_review.py -q
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py --dry-run
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py --dry-run
```

Final targeted verification: **84 passed**, 26 existing/dependency deprecation
warnings, 89.74 seconds. `git diff --check` passed.

Dry-run 01: `not_run_missing_data`, missing
`data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet`,
`outputs_written=false`.

Dry-run 02: `blocked`, `G0_frozen_reference_unavailable`,
`outputs_written=false`, `temporal_flow_executed=false`.

Synthetic tests generate temporary fixed-schema raw/state views, manifests,
mapping/QC/provenance, full/missing-free/control Mapper fits, membership/topology,
loadings, contributions, patient bootstrap/matching, circularity, administrative
QC, sensitivity comparison and figures under pytest temporary output roots.
Historical source bytes are checked unchanged. Test products are not committed
as clinical data. Real output/config/dictionary hashes are emitted by Script 01
and checked before Script 02 accepts the view; scenario manifests record exact
features, weights, seeds, code and input hashes. No real input SHA exists here.

Dependencies used:

| Dependency | Version |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.5.3 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| NetworkX | 3.7 |
| PyArrow | 25.0.1 |
| pytest | 9.1.1 |
| KeplerMapper | 2.1.0 |
| PyYAML | 6.0.3 |

Pending validated study execution: exact private vocabulary inspection and clinical
approval, source date provenance, frozen upstream hashes and dirty historical
snapshot reconciliation, real S3 invariance, 159/497 cohort verification, real
S3/S4 comparison and >=200 patient refits, independent external outcome assessment,
missingness/calendar/protocol review, formal branch definition and documented
scientific approval. No final temporal flow or transition characterization ran.
