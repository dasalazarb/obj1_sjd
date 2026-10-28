"""Sensitivity of anchor effects to refitted discovery, using frozen 02 refits."""
import numpy as np
import pandas as pd
from .s03_profiles import numeric
from .s03_statistics import cliffs_delta


def discovery_sensitivity(refits,baseline,roles,anchors,settings,validation):
    rows=[]
    if anchors.empty: return pd.DataFrame(columns=['representation_id','source_column','status','reason'])
    for anchor in anchors.itertuples():
        if not hasattr(anchor,'community_id') or pd.isna(anchor.community_id): continue
        identifier=anchor.representation_id; community=int(anchor.community_id); column=anchor.source_column
        row={'representation_id':identifier,'community_id':community,'source_column':column,
            'uncertainty_type':'empirical_discovery_refit_sensitivity_not_confirmatory_CI',
            'n_requested':int(settings['final_subsampling_replicates'] if identifier in settings['final_representation_ids'] else settings['secondary_subsampling_replicates']),'status':'NOT_ESTIMABLE'}
        if getattr(anchor,'status','')!='COMPLETE': row['reason']='PRIMARY_ANCHOR_TEST_NOT_ESTIMABLE'
        elif refits.empty or column not in baseline: row['reason']='FROZEN_REFIT_ASSIGNMENTS_OR_ANCHOR_UNAVAILABLE'
        else:
            part=refits.loc[refits.representation_id.eq(identifier)].merge(baseline[['patient_id',column]],on='patient_id',validate='many_to_one')
            effects=[]
            for replicate,frame in part.groupby('replicate_id'):
                inside=frame.reference_community_id.eq(community) & frame.matching_jaccard.ge(float(settings['match_jaccard']))
                x=numeric(frame.loc[inside,column]).dropna().to_numpy(float)
                y=numeric(frame.loc[~inside,column]).dropna().to_numpy(float)
                if min(len(x),len(y))<int(validation['minimum_group_patients']): continue
                effects.append(cliffs_delta(x,y))
            row['n_estimable_refits']=len(effects)
            if effects:
                lo,median,hi=np.quantile(effects,[.025,.5,.975]); original=float(anchor.effect)
                # For binary anchors the primary effect is OR; refit sensitivity
                # deliberately uses rank effect with null at 0, direction OR=1.
                direction=np.sign(original-1) if getattr(anchor,'effect_type','')=='odds_ratio' else np.sign(original)
                row.update(status='COMPLETE',reason='',effect_type='cliffs_delta',effect_median=float(median),
                    effect_p025=float(lo),effect_p975=float(hi),direction_consistency=float(np.mean(np.sign(effects)==direction)))
            else: row['reason']='INSUFFICIENT_MATCHED_REFIT_GROUP_SUPPORT'
        rows.append(row)
    return pd.DataFrame(rows,columns=None) if rows else pd.DataFrame(columns=['representation_id','source_column','status','reason'])
