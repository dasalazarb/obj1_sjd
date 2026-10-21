# Step 14 composition provenance and figure review — 2026-10-21

Implementation base: `08d542923870472d601581fda9ce65ab770467ec` (`main`).
Branch: `codex/step14-composition-qc-figures-20261021`.
Scope: the inclusive consumer, its aggregate QC, figures and synthetic tests.
Steps 10/11/13 and the scientific functions/CLI of Step 14 are unchanged.

## Source findings and limits

The Step 10 classifier explicitly assigns `_essdai_version`,
`_essdai_total_version`, `_essdai_resolution_method` and `essdai_total_source`
to PROVENANCE. Its builder seeds the analytic product from structural columns,
then merges only ANALYTIC roles; its context product merges only CONTEXT roles.
Thus these provenance columns are excluded by the current standard producer
contract. This is a code-level finding. Physical coverage in the investigator's
clinical files cannot be certified in this environment: no canonical Parquet,
registry, manifest or prior clinical result bundle is present here.

The upstream producer has a further semantic limitation. In
[`eda_sjd/src/09d_finalize_clinical_episode_dataset.py`](https://github.com/dasalazarb/eda_sjd/blob/ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4/src/09d_finalize_clinical_episode_dataset.py#L588),
`apply_essdai` sorts variable suffixes and resolves each variable separately.
Line 640 assigns `_essdai_version = 'r' if r else 'legacy'` **inside that loop**.
Lines 641–642 assign `_essdai_total_version` only when resolving the total.
Therefore the first field is the form source of the last resolved variable;
it cannot certify all twelve domains. These are form-source labels, not proven
versions of the clinical scoring algorithm. Restoring the two columns and
testing equality would still be insufficient. The reviewed Git blob is
`eeb9467e602612c85d8e990e2e19c38728eaeee4`; the pinned repository commit is
`ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4`.

`obj1_sjd/src/00_build_visit_spine.py` copies the authoritative EDA episode spine
and records its SHA-256; it does not reconstruct scoring provenance. Module 01
copies the canonical observed ESSDAI total. No all-domain scoring certificate
was established by the reviewed sources. The existing Step 14 context join
already rejects duplicate keys and conflicting shared columns; the consumer
now also rejects duplicate/missing episode keys directly before interval
construction. It does not manufacture versions, infer them from totals, or
repair producer files.

The investigator's 313 intervals, 301/297 concordant endpoints and 36 unsupported
sensitivities are reference observations, **not newly reproduced results**.
Actual cohort/model counts are derived on each run. This patch cannot report
clinical coefficients, a before/after clinical table, or corrected clinical
PDFs without those physical inputs. Synthetic renderings are labeled accordingly.

## Decision contract

The default decision is `UNVERIFIED`. The two-sided subtraction remains masked,
and unsupported leave-source tasks include
`not_estimable_version_provenance_unverified` with detailed category names.
Arithmetic results remain diagnostic. No arithmetic-only inferential analysis
is enabled. Primary tasks, support minima, clustering, adjustment selection,
index routes, censoring and primary FDR membership are preserved.

An optional **upstream-authorized** `composition_provenance` object in the Step
13 v2 producer manifest can bind a future scoring certificate to physical
Step 10/context inputs. It requires all of the following:

- `author`, `protocol`, `source_role` (`step10` or `context`), `source_sha256`.
- `producer_path` within the repository, `producer_sha256` matching both the
  physical source code and the producer manifest's `source_hashes`, and
  `producer_commit` matching that manifest's `code_commit`.
- `domain_version_field` and `total_version_field` physically present in the
  bound input; explicit semantics `all_12_domain_scoring_algorithm` and
  `recorded_total_scoring_algorithm`.
- `temporal_policy = episode_as_of_no_future_information`, canonical weights,
  and a nonambiguous `compatible_versions` matrix. Each entry names
  `domain_version`, `total_version`, and `scoring_system`.

Different tokens can be compatible only through that explicit matrix. Both
endpoints must be certified and map to the same scoring system. A contract
cannot create missing physical columns. An invalid declared contract blocks
publication; no declared contract leaves sensitivity safely unverified. A
manifest declaration is an accountable upstream authorization, not automatic
proof extracted from a column name. The existing reviewed EDA fields do not
meet these semantics. No authorization is supplied or invented by this patch.
New external recovery or changed producers require a separate controlled
upstream regeneration and reconciliation, preserving prior snapshots.

When enabled by valid evidence, other-domain tasks use the domain predictor at
FROM, the other-domain score at FROM as state, and the score at TO as outcome.
An additional matched-total task uses the same interval sample and freezes the
other-domain model's selected adjustment. Sample hashes must agree. Official
comparisons freeze the inclusive adjustment and scaling. Comparisons are
descriptive; no formal test of coefficient differences is introduced.

## Aggregate products

All filenames below have prefix `14_explainwhy_inclusive_v2_` and reside in the
normal Step 14 QC directory:

| Product | Meaning |
| --- | --- |
| `composition_version_source_audit.csv` | Physical fields, dtypes, nonnull coverage, registry role/producer/clock, reviewed semantics and source hashes; includes missing optional name mapping and code inventory. No patient values. |
| `composition_interval_gate_qc.csv` | Full-spine and sequential interval counts, FROM/TO/both, each domain, all/official/alternative source; unique patients, any-predictor availability and fallback counts. Observed cohort events are separate from modelable event intervals. |
| `leave_source_support_qc.csv` | Actual estimator rows and source-specific candidate/verified/predictor/fitted counts; adjustment, formula and detailed failure counts, including the excluded-discordant-date sensitivity. Source-specific complete-case interval counts unavailable from the estimator are NA. |
| `composition_provenance_decision.json` | Authorization or UNVERIFIED decision, evidence hashes, compatibility matrix, author/protocol, pinned source-code finding and limitations. |
| `domain_composition_qc.csv` | Existing arithmetic columns plus post-version endpoint/pair counts and missing/incompatible/unverified-identity counts. The old hardcoded manifest flag is explicitly labeled as historical and is not a disk inventory. |
| `primary_result_regression_qc.csv` | Prior non-composition results compared when physical input hashes and dependency versions match (`rtol=1e-8`, `atol=1e-10`); mismatches block publication. Missing prior tables are MISSING; differing inputs/dependencies are NOT_RUN. |

The composition products include `qc_rule_version`, consumer/producer commit,
input origin and source hashes. Source inventories use MISSING for absent
fields; model support uses NOT_ESTIMABLE. Gate-count PASS means the audit ran,
not that every interval qualified. Failure categories are exclusive with
priority: missing target/total, incomplete panel, discordant total, missing
version evidence, incompatible versions, unverified scoring-producer identity,
compatible verified. Both endpoint categories and version-only states remain
available in the controlled analytic interval product for secondary causes.

Every new artifact is hashed in the transactional bundle manifest. Prechange
snapshots now capture both inclusive and historical output hashes. Input and
authorization evidence is rechecked before publishing. Backups, rollback,
manifest-last installation, dry-run and no-overwrite protection remain in place.
Only code/docs/synthetic tests are committed; generated clinical data and PDFs
remain in controlled storage with institutional disclosure review pending.

## Figures and validation

Trajectory charts have one axes title, a separated privacy footer, temporal
origin, observed patient/observation counts, lighter official lines and the
original alternative-index × artists. The forest uses a shared ordered catalog
in both panels, explicit unsupported rows, common log-HR limits covering full
finite confidence intervals, and per-model N/events/fitted-alternative counts.
Nominal p, within-family q and separate outcome-wide primary q are labeled;
laboratory sensitivity estimates use a separate color. Each plotted estimate
has a visible predictor label. No missing effect is drawn at HR 0 or 1.

Synthetic tests cover semantic certification, unequal-but-authorized tokens,
missing evidence, equal-but-unverified strings, scoring incompatibility despite
identical arithmetic, cross-endpoint scoring mismatch, incomplete/discordant
panels, duplicate/colliding keys, frozen matched comparisons, unchanged primary
interval/event contracts, both clocks and exactly ten alternative-index artists
including scatter. Existing event-censoring, no skipped visits, FDR, CLI,
hash/rollback/no-overwrite tests remain active. Real execution results and
dependency versions are recorded in `14_explainwhy_inclusive_v2_validation.json`.
