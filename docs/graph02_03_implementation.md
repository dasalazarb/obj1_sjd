# Graph02/03: implementation and validation record

Date: 2026-10-28. Protocol: `graph02_03_20261028_v1`.

## Inventory before extraction

The original numbered executables contained 1,254 (01), 1,376 (02), and 692 (03)
lines. Graph02 owned coverage/redundancy, RobustScaler/MFA/PCA, Mapper, community
membership, resampling QC, historical CSVs, and figures. Its historical procedure
called bootstrap was 80% sampling without replacement and matched each reference
to its best candidate independently. Graph03 dynamically imported 01 and 02,
rebuilt the feature catalogue at coverage zero, and restricted characterization to
S2/S3. Its reusable statistical primitives included Cliff's delta, odds ratios,
Fisher/Mann–Whitney tests, and BH correction.

## Final boundaries

`stage02/s02_legacy.py` preserves the historical algorithms and callable API.
Numbered 02 exports those functions for compatibility; its normal CLI enters the
full pipeline, while `--legacy` invokes the original CLI. Monkeypatch tests now
target the extracted implementation module. Graph01 only gains an optional
`resolve_profile=False` argument so an already verified exact source cannot be
silently replaced. Shared candidate resolution refuses distinct ambiguous bundles.

New stage02 modules implement clinical-map joining, clinical MFA, source-level
mixed distances/FAMD, separate lenses, resampling, perturbations, structural
comparators, technical decision rules, reporting, and release contracts. The fixed
primary is S3/R0. No clinical outcome reaches release selection. Source, domain
map, and experiment configuration are hashed. Failed runs never update the pointer.

Stage03 has no import of a numbered discovery executable or discovery model.
Its inputs are frozen raw baseline values, membership, feature roles, alignment,
refit assignments, and validated source hashes. It recovers statistics from the
previous implementation while adding roles, estimability, conditional incremental
associations, actual-time mixed models, discovery-refit sensitivity and reporting.

## Validation coverage

`tests/test_graph_pipeline.py` exercises all three entrypoints with **no argv**
using only a temporary synthetic source and a temporary default pipeline config.
The integration test reduces replicate budgets and Mapper grid size explicitly;
the production default of 200 final subsamples and the full 3×3 grid are asserted
separately. This is not a claim of a completed production clinical run.

Coverage includes:

* Actual Pharma00 provenance generation, including missing-source, duplicate-source
  ambiguity, pinned manifests, and source/output hash verification.
* S0–S4 preserved functions; R2 pending and synthetically approved mappings; an
  incomplete S4 map that does not prevent valid R0/R2/Gower releases.
* Gower symmetry, bounds and no-shared-observation failures; source-level nominal
  handling; clinical MFA empty/constant domains, duplicates and temporal rejection.
* Reproducible distinct subsamples and bootstrap collapse to unique patients;
  Hungarian label permutation/splits; failure denominators and null masks/marginals.
* Release status gates, no-stable-structure negative results, artifact tampering,
  path validation, exact release provenance, atomic latest pointer preservation,
  new immutable run directories and verified resume cache.
* A Graph03 execution with discovery fitting replaced by a function that raises,
  proving interpretation does not refit discovery.
* Defining/proxy circularity and separate independent anchors; estimable effect
  intervals/FDR, incremental associations, refit sensitivity and irregular-time
  mixed models on synthetic data with known group effects.
* PNG/PDF output existence and master index SHA concordance.

## Clinical and methodological limits

The committed D01–D09 source map is a **candidate map pending clinical review**.
It does not assert approval of assays, units, temporal validity, diagnostic roles,
or all available real columns. Unknown exact sources remain unmapped. R2 is
explicitly not estimable until applicable mappings are reviewed. S4 LAB categories
remain distinct from clinical domains; the historical RENAL_URINARY alias is
recorded explicitly rather than treated as a new clinical modality.

Optional UMAP requires its optional dependency. Rare events, diagnostic-selection
contrasts and knowledge-date contrasts require exact approved source metadata;
otherwise the registry records not-estimable reasons. No source name heuristic
invents an organ-event risk set, HLA encoding, or rare-event date. The current
longitudinal inferential endpoints are continuous measurements; population label
transitions are descriptive. No causal graph transit or out-of-sample projection
is claimed.

Clinical effect/model confidence intervals condition on the frozen estimated
labels. The additional repeated-discovery effect distribution is an empirical
sensitivity analysis, not a confirmatory confidence interval. No external clinical
validation or clinical-release approval is inferred from technical resampling.

Real Biowulf source preflight, domain/assay approval, manuscript Results with real
numbers, and a final real-data run require that infrastructure and its reviewed
sources. They are not replaced by the synthetic CI fixture. No patient-level
artifacts or real-data logs are committed.

## Local verification

The Graph/Pharma regression selection passed 228 tests. The final focused pipeline
selection passed 23 tests, including the additional future-measurement-date and
clinical-unit/anchor-role gates. All three zero-argument entrypoints were exercised
against synthetic Pharma00 outputs. Synthetic Mapper PNGs were visually checked
for node support, actual nerve edges, reference-matched colors, and uncovered
patients. `git diff --check` passed. No real Biowulf run was performed.
