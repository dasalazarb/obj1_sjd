# Clinical missingness and HLA time contract

## Tri-state clinical phenotypes

An **observed inactive** state is not interchangeable with **not evaluated**.
Domain and manifestation flags therefore use pandas nullable booleans: `True`
means interpretable positive evidence, `False` means interpretable negative
evidence, and `NA` means missing or uninterpretable evidence. A grouped
glandular manifestation is positive when any declared source is positive and
negative only when every declared source is explicitly negative. Partial
negative evidence remains unknown.

Aggregate glandular and extraglandular flags use Kleene OR. They are negative
only when every component is decided and negative. Their evaluability flag is
true exactly when that aggregate is decided. Existing `n_*_active` names remain
totals: they are `NA`, rather than a misleading zero, unless every component is
decided. Likewise, active-domain names are missing until the complete domain
set is evaluated.

`overlap_active` is nullable Kleene AND. `overlap_status` is stricter: it is one
of `overlap`, `glandular_only`, `extraglandular_only`, or `neither` only when
both axes are known; otherwise it is `unclassifiable`. Thus one known-negative
axis can prove `overlap_active=False` while still being insufficient to assign
the complete category.

## Coverage

`n_integrated_blocks_available` retains its operational meaning. QC separately
reports `has_any_curated_clinical_evidence`, derived from observed ESSDAI,
observed ESSPRI/PRO, dated or counted episode laboratory measurements,
evaluability signals, and measured extended clinical features. Retrospective
HLA consensus and default/unclassifiable labels are not episode measurements.
All coverage sources are aligned on `(patient_id, clinical_episode_id)`.

## HLA retrospective and as-of values

`hla_*__patient_consensus_value` describes the whole observed trajectory and is
suitable only for explicitly retrospective description.
`hla_*__known_through_episode` indicates whether dated evidence exists at or
before the clinical anchor. `hla_*__asof_value` is recomputed at every anchor
from only dated evidence available by then, and is missing when that evidence
is absent or discordant. Undated or future results never enter it. A later
conflict therefore cannot erase an earlier unambiguous as-of result. Episodes
sharing a date use the same date cutoff; no unsupported intraday order is
invented.

Prospective Pharma, Graph, and Economic readers must use the as-of family (and
retain missing values), never the retrospective consensus as an episode-level
predictor. The shared study boundary already rejects columns containing
`patient_consensus`; no Graph or Economic episode reader currently exists in
this repository. Descriptive analyses may use consensus only with its
retrospective label and consensus-conflict QC.

## Regression metrics

Step 10 QC computes cohort, episode, baseline, coverage, zero-block, and
tri-state distributions from each run. No historical cohort size is an
acceptance threshold. A pre/post table can only be produced when both run
artifacts exist; numeric differences are descriptive, while key/grain,
tri-state, registry, and temporal-contract violations are failures.
