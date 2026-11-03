import numpy as np
import pandas as pd
from src.studies.graph.stage03.s03_profiles import pop_denominator_audit,profile_tables
from src.studies.graph.stage03.s03_incremental_value import pop_crosswalk
from tests.test_graph_pipeline import clinical_frames


def test_pop_universes_and_crosswalk_reconcile_without_reclassifying():
    baseline,membership,meta,cfg,roles=clinical_frames()
    baseline.loc[:5,'pop__status']=['Pop1','Pop2','Pop3','Unclassified',np.nan,'invalid_label']
    r2=membership.copy().assign(representation_id='R2')
    r2.loc[:9,'mapper_covered']=False;r2.loc[:9,'hard_assignment']=False
    r2.loc[10:19,'supported']=False
    membership=pd.concat([membership,r2],ignore_index=True)
    meta={**meta,'eligible_representation_ids':['R0','R2']}
    audit=pop_denominator_audit(baseline,membership,meta,cfg)
    counts=['n_pop1','n_pop2','n_pop3','n_unclassified_pop','n_missing_pop','n_invalid_pop']
    assert audit[counts].sum(axis=1).eq(audit.n_denominator).all()
    r0=audit.loc[audit.representation_id.eq('R0') & audit.population_universe.eq('hard_supported')].iloc[0]
    assert (r0.n_unclassified_pop,r0.n_missing_pop,r0.n_invalid_pop)==(1,1,1)
    assert audit.loc[audit.representation_id.eq('R2'),'n_denominator'].tolist()==[90,80,70]
    crosswalk,concordance=pop_crosswalk(baseline,membership,meta['eligible_representation_ids'],cfg)
    for identifier in meta['eligible_representation_ids']:
        hard=audit.loc[audit.representation_id.eq(identifier)&audit.population_universe.eq('hard_supported')].iloc[0]
        n=hard.n_pop1+hard.n_pop2+hard.n_pop3
        assert crosswalk.loc[crosswalk.representation_id.eq(identifier),'n_patients'].sum()==n
        assert concordance.loc[concordance.representation_id.eq(identifier),'n_common'].eq(n).all()
    flow=profile_tables(baseline,membership,roles,meta,cfg,2)['flow']
    row=flow.loc[flow.representation_id.eq('R0')].iloc[0]
    assert row.n_unclassified_pop==1 and row.n_missing_community_id==0
    assert pd.isna(row.n_unclassifiable) and 'DEPRECATED' in row.n_unclassifiable_definition


def test_prespecified_marker_can_describe_non_discovery_non_anchor_source():
    from src.studies.graph.stage03.s03_profiles import clinical_marker_summaries
    baseline,membership,meta,cfg,roles=clinical_frames()
    baseline['pro__profad_total']=np.arange(len(baseline),dtype=float)
    # Not a discovery feature or independent anchor: still an observed description.
    result=clinical_marker_summaries(baseline,membership,roles,meta,cfg)
    profad=result.loc[result.source_column.eq('pro__profad_total')]
    assert profad.n_observed.sum()==len(baseline) and profad['median'].notna().all()
    assert profad.validation_role.eq('unavailable').all()
    assert profad.inference_scope.eq('observed_baseline_description_not_independent_validation').all()
