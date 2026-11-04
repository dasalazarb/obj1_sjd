# N3 audit contract: graph03_n3_audit_v1

Graph03 consumes verified frozen Graph02 inputs. The N3 audit does not change
feature roles, Mapper, hard memberships, R0/R2 labels, historical incremental
estimators, or longitudinal endpoint policies. N1 is internal characterization;
N2 is related characterization. Neither becomes independent validation.

Run only the standalone entrypoint, from the repository root:

```bash
python src/studies/graph/03_graph_community_characterization.py
```

Use the project's configured interpreter when available. This command requires
the completed Graph02 pointer, its hash-verified artifacts and unchanged source
inputs. Missing clinical files must be supplied through the existing release
contract; the audit does not invent a baseline or search arbitrary patient paths.
Do not run `run_graph_study.py` to obtain N3 outputs: that would rerun discovery.

## Audit and review

`s03_n3_candidate_audit.py` contains pure source-level functions. Inputs are the
frozen baseline, `feature_roles`, feature sets, clinical registry, manifest,
hard memberships and versioned validation configuration. All baseline columns,
including UNMAPPED and unavailable sources, retain one row per representation.
The known reference release enforces 505 sources and 159 patients; an optional
`expected_source_count` checks an explicitly amended contract for other releases.

The review states are PENDING_REVIEW, BLOCKED_DEPENDENT, SOURCE_UNAVAILABLE and
N3_REVIEWED. Statistical nonestimability is separate from clinical independence.
Screening prioritizes documentary clinical relevance and support, never p values.
Possible same-assay encodings, ESSDAI activity domains, ESSPRI versus discovery
PROs, biopsy selection and IgG subclass risks are review alerts. They are not
newly invented lineage edges. Existing documented typed dependencies block
inference in both directions without bridging unrelated components through Pop.

All independence attestations remain in
`characterization.independence_reviews`, keyed by the exact representation and
source. Every existing required review field must be complete. A new source may
be reviewed through this contract without silently changing historical anchors;
`tier_original` preserves its old role. Identifiers, administrative variables,
adjustments and downstream clinical labels remain excluded.

A source reference must resolve to an actual repository file under `src/`,
`docs/` or `tests/`; optional `:symbol` references must occur in that file.
The IO layer records the evidence file's SHA-256. File existence establishes
provenance only. Human clinical, proxy, origin, exclusion and baseline review
remain mandatory. Invented attestations exist exclusively in synthetic tests.

Additional representation/source measurement policies live under
`n3_validation.measurement_policies`. They require documented origin review and
verified units for continuous sources, or verified category conventions for
binary/categorical sources. Positive, negative, unknown and not-tested tokens
must be disjoint and explicitly documented before assay counts are exported.
Not tested is never negative. Without these conventions the audit records
observed coverage and leaves assay-specific denominators unavailable.

A separately reviewed `n3_validation.baseline_era_reviews` record is required
before using any date to construct adjustment eras. Naming a column
`clinical_anchor_date` does not verify its semantics. This is distinct from the
outcome's basal window and from all longitudinal date approvals.

## Conditional comparisons

`prespecified_outcomes` is empty in production configuration. Empty plans or
unapproved endpoints produce NOT_ESTIMABLE rows with null estimates and an empty
comparison table, not evidence of no association. Reviewed outcomes can have
restricted descriptive profiles without a prespecified inferential hypothesis.

Model A uses the existing adjustment variables and a reviewed five-year basal
era. Model B adds frozen community. Pop is used only after a separate, scoped
comparator review and is prohibited for ESSDAI/ESSPRI constructs and derivatives.
A covariates-only comparator is explicitly labeled.

C requires a prespecified reduced N1 basis in `source_comparators`, a documented
design review, exact N1 source membership and `common_complete_case` missingness
policy. D additionally requires `include_model_d`. No unregularized all-N1 model
or outcome-driven feature selection is enabled. C/D remain NOT_ESTIMABLE if their
design cannot be specified with integrity.

A/B and planned C/D use the union of required nonmissing sources on exactly the
same unique patients. No imputation occurs. Existing stability, minimum model
size (30), community support (8), variation, full-rank and residual-df gates
apply. Continuous models use HC3 intervals, adjusted R2 and paired patient
bootstrap intervals for its change. The seed derives from the frozen manifest
and the prespecified comparison; Mapper is never refitted in a bootstrap.

Binary outcomes additionally require a prespecified `binomial_logit`, documented
statistical review and a reviewer-specified positive
`minimum_events_per_parameter`. Insufficient events, separation or failed model
fits block the comparison. Odds ratios, conditional confidence intervals and
in-sample AUC/Brier/calibration diagnostics are endpoint-specific. Firth, mixed
and joint models are not enabled by package availability.

BH correction uses a family fixed by the representation's prespecified
comparisons; blocked planned tests remain in the family denominator. Missing
estimates never acquire a p value or q value. The models describe exploratory
in-sample association conditional on frozen labels. They do not provide external
prediction, prognosis, temporal progression or information beyond X.

## Outputs, recovery and privacy

The nine `03_graph_n3_*.csv` tables, QC JSON and Markdown summary are written in
the new immutable Graph03 run and included in its artifact SHA inventory and
results summary. Phase A is persisted before Phase B. A model-stage exception
produces schema-stable blocked tables and an exception type, retains the audit,
and never includes sensitive exception text. A later historical-analysis failure
also leaves the already written audit available. Previous runs are untouched.

All N3 tables are INTERNAL_RESTRICTED with public export disabled. Coverage is
computed once per unique patient before category expansion, and reconciled to a
single source/community denominator in historical characterization. Small-cell
community coverage remains restricted; no new figure/public-export route is
introduced. Existing `safe_characterization_panels()` remains the route for
reviewed, suppressed shareable panels.

Review notes, reviewer names and literal baseline windows are not copied into
artifacts: controlled statuses, presence indicators and evidence hashes preserve
traceability to restricted configuration. No patient IDs, episode IDs, exact
clinical dates or categorical free values enter the new outputs. No automatic
ZIP or public manuscript appendix is created from these internal tables.

Validate using invented data only:

```bash
python -m pytest -q tests/test_graph03_n3_candidate_audit.py \
  tests/test_graph03_n3_incremental.py tests/test_graph_characterization_tiers.py \
  tests/test_graph_pipeline.py
```

The full-pipeline tests build their own temporary synthetic releases. They do
not rerun or change a clinical Graph02 release.
