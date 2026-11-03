# Graph03 temporal contract

`graph03_episode_resolution_v1` resolves each endpoint once, before community
joins and observed-value filtering. Baseline and supported membership joins must
have one row per patient. Visits require actual elapsed time strictly after the
frozen baseline; two records on the same day never satisfy two follow-up dates.

The repository's Step10 public schema and producer expose a clinical spine
anchor. They do not, by themselves, establish the measurement time of every
attached instrument. The real integrated parquet and its frozen Graph02 release
are unavailable in the development workspace. Consequently the distributed
`temporal_resolution.endpoint_policies` is empty. This intentionally produces
`NOT_ESTIMABLE / UNVERIFIED_FOLLOWUP_MEASUREMENT_DATE` until a review against the
authorized Helix sources supplies endpoint-specific evidence. Graph03 does not
rerun Graph02 or change upstream records to satisfy this contract.

After inspecting the real schema, a reviewed endpoint policy records:

| Field | Required evidence |
| --- | --- |
| `date_field` | Existing source field containing this endpoint's measurement date |
| `date_semantics_review_status` | `VERIFIED` after source review |
| `date_semantics` | `endpoint_measurement_date` |
| `evidence` | Inspectable source/metadata citation and review record, without patient identifiers |
| `identity_review_status` | `VERIFIED` for the observation identity |
| `episode_key_fields` | Existing fields identifying the same endpoint observation/episode |
| `provenance_fields` | Existing instrument/source fields that must agree for a technical repeat |
| `source_grain` | Reviewed description of the actual source grain |

Column names are not inferred or approved from their spelling. Synthetic tests
supply explicit attestations about invented records; these policies never approve
NIH data. Multiple instruments are resolved independently. Technical repeats
collapse only when the reviewed identity, date, endpoint value and declared
source/instrument provenance agree. All original row references remain in the
in-memory reconciled record. Conflicting values/dates and distinct same-day
episodes are excluded as affected observational units, with aggregate reasons;
there is no first/last/mean consolidation policy. No patient-level diagnostic is
written automatically.

Pop uses this same reconciler for the upstream Pop observation. The principal
baseline-to-first-valid-followup estimand has at most one destination per patient.
An earlier or undated ambiguous observation prevents identifying that first
followup. A later conflict does not remove an earlier verified first visit. The
all-followups sensitivity counts patient-visit pairs; its cell patient counts are
nonadditive. Missing, Unclassified and invalid Pop labels remain separate.

Eligibility reports clinical date review, resolution losses, descriptive support
and model support independently. Descriptions can exist while stability,
community size, repeated measurements or prespecified adjustment requirements
block inference. Frozen stability labels do not establish clinical robustness.
R0/R2 common-universe comparisons preserve representation-specific community
labels; equal labels do not demonstrate alignment.

All audit tables and figure source CSVs are aggregate. Figures separate original
clinical units from patient-source availability, suppress small cells and their
complements, and retain internal/restricted status. An empty panel records a
limitation rather than inventing a trajectory. Run only the Graph03 entrypoint
against the verified release in Helix; inspect the date audit and the actual
resolved Graph02 run ID/SHA before interpreting any output.
