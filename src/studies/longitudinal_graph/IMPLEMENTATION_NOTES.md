# Additive episode serology representation

S3 retains its feature selection, median imputation, RobustScaler, family weights,
PCA and public output names. Step 10 and Scripts 03/04 are unchanged. Script 01
writes a separate 12-column binary view. It never inserts those columns into the
historical visit state. All joins validate episode keys one-to-one and preserve
row order and patient membership.

S4-serology-OHE appends the complete SSA, SSB and ANA blocks to the retained S3
list. S4-without-missing omits only the three missing indicators from its fit;
its source view retains all 12. S4-no-serology fits the historical state with
historical weights. OHE indicators bypass RobustScaler, receive fixed weights
of family_weight/sqrt(12), and are subsequently centered by PCA. Each biomarker
has equal squared weight mass. We report empirical weighted variance and squared
loadings by feature, family and biomarker, including missing and hsCRP. Constant
columns remain in the design. Spearman pruning never deletes or competes with
OHE blocks.

The default candidate-qc CLI runs the serology family in the existing create-only
review bundle, after verifying the frozen S3 reference. Its S3 feature list is read
from archived PCA loadings, not newly selected with serology. The original frozen
outputs remain untouched. Programmatic legacy runner calls also write separate
S4 products. A missing OHE bundle generates an explicit S4 blocking report while
S3 remains available. `--serology-state` optionally supplies the separate view;
its raw and provenance files must be siblings. The existing CLI freeze/reference
protections remain applicable. The three serology names are accepted by
`--scenario`; requesting a serology member evaluates the complete three-member
family so the controls always accompany the candidate.

## Clinical and temporal prerequisites

The shipped mapping YAML is deliberately **unapproved and empty**. No real
Step-10 Parquet, variable registry, historical output archive or clinically
reviewed vocabulary was available in this workspace. Real 159-patient/497-visit
coverage, 98/110-feature equivalence, topology and stability are therefore
**not verified**. No clinical approval, real comparison or improved phenotype is
claimed. Synthetic test approvals are fixtures only and never populate the
production dictionary.

Before a real run, privately inspect exact source value frequencies. Complete
`markers.<ssa|ssb|ana>` with exact normalized-label entries containing `level`,
`approved_by`, and `rationale`; supply mapping `version`, `approved_by`,
`approved_on`, and `rationale`. Nulls alone are automatically missing. Empty
strings, comments, numeric titers and unrecognized strings block encoding. Only
strip/whitespace normalization and casefold are applied; no clinical substring
inference occurs. Exportable vocabulary and mapping audits contain hashes rather
than arbitrary source text.

Each source requires a unique episode_resolved registry record. This does not
prove contemporaneous acquisition: Step 10 curates lab measurement dates and
anchor offsets as context, rather than establishing a universal no-future rule.
Where available, configure reviewed `temporal_review.date_columns` per marker
and documented temporal approval. Those dates must be present, parseable and at
or before the anchor for observed results. No new post-visit tolerance is assumed.
Without those dates, provenance remains `temporal_provenance_needs_review` and
confirmatory use remains blocked. No forward/backfill or later-episode lookup is
performed. Discordance is reported, never corrected automatically.

Bootstrap independently refits each S4 candidate from raw numeric observations
joined with raw OHE. It draws complete patient series, preserves original IDs for
matching, estimates medians/scaling/PCA only within the draw, freezes the S3 feature
list and keeps indicator columns and weights. An all-missing numerical feature
fails the replicate rather than borrowing a full-cohort median. Failed/no-node
replicates are retained. Production budget is at least 200 per scenario. Unit
integration tests use two replicates explicitly as a test-only speed override.

Outputs include separate topology, membership, PCA, variance/weight manifest,
bootstrap, comparison, circularity, administrative coverage, temporal concordance,
figures and provenance. ARI denominators and matched Jaccard universes are explicit.
Serology defining S4 is descriptive and circular, never independent validation.
ESSDAI/ESSPRI/Pop remain external; independent clinical validation and proxy review,
formal branch definition, temporal provenance, review of missingness and dominant
loadings, and additional hsCRP/PRO controls remain pending. Scientific gates stay
blocked and neither Script 03 nor 04 is executed.

## Verification in this workspace

Run from the repository root with the configured Python environment:

```bash
python -m pytest tests/test_serology_ohe.py tests/test_longitudinal_graph.py tests/test_longitudinal_mapper_cli.py tests/test_longitudinal_methodology_review.py -q
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py --dry-run
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py --dry-run
```

The dry runs report missing integrated data and missing frozen historical reference
respectively, write no outputs and leave temporal analysis blocked. The synthetic
integration test exercises coexistence in an isolated output root, real Mapper
fits, patient refits, full/missing-free schemas, circularity and historical state
bytes. No synthetic products should be interpreted as study results. Intermediate
logs are in ignored `work/`; dependency versions and verified test results are
recorded in `docs/serology_ohe_verification.md`.
