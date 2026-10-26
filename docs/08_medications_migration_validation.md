# Step 08 migration validation

Validation date: **2026-10-26**. Executor: **Codex managed cloud environment, Python 3.12**, using invented fixtures only. This is not a Biowulf validation or permission to promote clinical outputs.

The branch starts at `origin/main` commit `64ca32b0bc4d2b37b4bf2247e41c5a3bf83d627c`. The protected source datasets, established NIH HMAC key and prior clinical-run manifests were unavailable here. No real patient data were downloaded, committed or published. No canonical clinical Step 10 dataset was regenerated or overwritten.

## Executed checks

- Exactly one top-level Step 08 script: `src/block_A/08_build_medications.py`.
- `python -m compileall -q src/block_A`: passed.
- Both productive CLI `--help` commands: passed. Every archived CLI `--help` smoke check passed except the explicitly disabled historical orchestrator, which has a regression test proving it exits before side effects.
- New source/temporal/contract/consumer tests: **41 passed** before the final affected-module run. Coverage includes independent full-source Step08, malformed/partial/ISO/MDY timestamp dates, leap-year inclusive windows, multi-reference provenance, consent-only exclusion from order fields, patient isolation, duplicate protocol documents, unmapped names, CTDB original visits/unassigned doses, missing keys, key-scope drift, dry run, validated reuse, file modes, hash/schema/spine rejection and publication rollback.
- Existing audits and original assertions were retained under `tests/audit/08_medications/` and `tests/audit/10_medications/`. Active publication assertions now require actual drug names, matching master/context/CSV/registry and equality of every nonmedication column, including dtypes, against the explicit clinical-only output.
- The shared study predictor validator rejects all documentation-name fields and their `from_` forms. Longitudinal graph embeddings explicitly reject medication namespaces.
- `git diff --check`: passed. Generated data, keys, parquets, manifests, restricted directories and intermediate outputs are ignored; only source, invented fixture code, tests and documentation are eligible for the commit.

Final affected-module run: **339 passed**, 10 pandas deprecation warnings, in 67.45 seconds. It includes the complete archived medication tests, all new producer/contract/temporal/consumer tests, integrated longitudinal tests, shared-study tests and longitudinal graph tests. No tests in that command failed.

Command (using the isolated `work/venv/bin/python` environment):

```bash
python -m pytest -q tests/audit/08_medications tests/audit/10_medications \
  tests/test_08_medications.py tests/test_08_medication_episode_contract.py \
  tests/test_08_medication_temporal_matching.py tests/test_10_medication_input_contract.py \
  tests/test_10_medication_master_publication.py tests/test_integrated_longitudinal_dataset.py \
  tests/test_studies_shared.py tests/test_longitudinal_graph.py
```

Synthetic publication evidence from that run (3 episodes / 2 invented patients):

- Step08 parquet SHA-256: `fcfb2b80d2916dc4b2de05b2a9c3d1fc2780b0852d07e66a7d0fbfd30a88ad2e`.
- Step10 staged parquet SHA-256: `9b76a40701beab7f20e4b344b7fe99ee5ee1bdd2376fd7a93f77d1230f7ac17d`.
- Step10 names match the manifest-validated Step08 fields in both analytic and context outputs; all nonmedication columns match the clinical-only build exactly, including dtypes.

These hashes describe **invented local validation fixtures**, not the NIH inputs or real clinical outputs.

## Wider repository checks and baseline reproduction

Full test collection found two pre-existing references to absent source files: `tests/test_btris_unmapped_lab_audit.py` expects `src/20_btris_visit_date_match_report.py`, and `tests/test_pharma.py` expects `src/studies/pharma/01_run_pharma.py`. Both files are absent from the unchanged base commit too.

With those two collection errors excluded, the wider run completed with **1131 passed, 21 failed, 1 skipped**. All **21 exact failing test cases were rerun on a separate, unchanged checkout of `64ca32b`**, where all 21 failed again. These are outside this medication migration; some require optional `lifelines`/`kmapper` dependencies unavailable in the test environment, while others assert pre-existing source hashes, output schemas or date labels. The focused pharma extension produced 346 passes and the same existing `test_domain_ordinal_delta_preserves_score_levels` failure (`interval_days` missing in its fixture), also reproduced independently on the base. The entire repository is therefore **not reported as green**.

The reproduced failure groups are comorbidities, cross-domain correlations, inclusive explainwhy, graph representation sensitivity, missingness observation, fixed observed-pattern source hashes, pharma plotting, longitudinal PRO labels, progression risk factors, topological phenotyping and visit-date labels. Their code and scientific rules were not changed to conceal the failures.

## Synthetic evidence and intended differences

The source fixture has **2 invented patients, 3 episodes, 14 BTRIS source records and 1 CTDB report**. All 15 source documents remain in the protected catalogue. Six document–episode relations use order dates, four use consent alone, and four documents remain unassigned because no exact reference date is available. Protocol reexports remain separate documents while their episode names are deduplicated. A separate invented size-regression fixture covers 159 patients / 497 episodes with every medication category missing; that test is not evidence about the real cohort.

For the first invented episode:

| Category | Synthetic names |
|---|---|
| CTDB reported | hydroxychloroquine |
| Order within episode | hydroxychloroquine |
| Order before / after episode | rituximab, in separately labelled fields from distinct documents |
| Nearby order union | hydroxychloroquine; rituximab |
| Consent-only review | invalid synthetic date; unknown synthetic product |

The deliberately malformed order date remains visible in parsing QC; its consent-only match never becomes an order-date association. Partial and undated documents remain patient history. These are methodological assertions tested on invented examples, not new clinical findings.

Intended migration differences are removal of the bidirectional dependency and old medication counts, separate source/time evidence for names, preservation of previously unmapped names, explicit omission instead of silent absence, strict publication validation, and retention of protected source multiplicity. No baseline count is forced and no order or consent becomes verified exposure.

## Protected Biowulf comparison — still pending

| Check from the supplied historical baseline | Previous result | New NIH run |
|---|---:|---|
| Clinical patients / episodes | 159 / 497 | Not executed; validate against configured Step00 |
| Consolidated BTRIS + CTDB documents | 9605 + 327 = 9932 | Not executed |
| BTRIS rituximab mentions / patients | 34 / 6 | Not executed; history must retain all source documents |
| Prior primary document–episode relations | 2753 | Not directly equivalent to the new selected-reference contract |
| Exploratory order-date / consent-only relations | 5399 / 7642 | Not executed; reconcile differences explicitly |
| Rituximab order-date relations / episodes / patients | 2 / 1 / 1 | Not executed |
| Rituximab consent-only relations / episodes / patients | 28 / 10 / 6 | Not executed |

Before promotion, an authorized NIH operator must freeze prior manifests and input hashes, verify the established HMAC scope, run the two staged commands in `08_medications_analytic_contract.md`, inspect all source and parsing exclusions, reconcile rituximab and other count differences, and compare all nonmedication Step10 values/dtypes and episode identities. Record the actual Biowulf executor, date, code SHA, source hashes and scientific review outcome here or in a protected companion record. Do not copy rows, patient identifiers, pseudonyms, raw drug detail or keys into this repository.

The real-data acceptance gates and clinical promotion remain **open**. This commit supplies the implementation, synthetic tests and migration evidence; it does not claim an NIH run or authorize a merge/promotion.
