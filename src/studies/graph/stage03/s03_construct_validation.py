"""Per-representation roles and explicit proxy/circularity gates."""
import pandas as pd


def feature_roles(feature_sets,baseline,validation,identifiers):
    rows=[]
    anchors={r['source_column']:r for r in validation['anchors']}
    for identifier in identifiers:
        selected=feature_sets.loc[feature_sets.representation_id.eq(identifier) & feature_sets.included]
        defining=set(selected.source_column)
        all_features=set(selected.feature)
        for c in sorted(set(baseline.columns)-{'patient_id'}):
            anchor=anchors.get(c,{})
            role='unavailable'; reason='not_prespecified_validation_variable'
            if c in defining or c in all_features:
                role='defining'; reason='frozen_embedding_or_lens_feature'
            elif c in anchors:
                dependencies=set(anchor.get('derived_from',[]))
                if c not in baseline or baseline[c].isna().all():
                    role='unavailable'; reason='no_observed_anchor_values'
                elif dependencies & defining:
                    role='derived_from_defining'; reason='prespecified_proxy_in_discovery'
                elif c.startswith('essdai__') and any(x.startswith(('essdai__','ovl__')) for x in defining):
                    role='derived_from_defining'; reason='ESSDAI_or_overlap_construct_in_discovery'
                elif c.startswith('esspri__') and any(x.startswith('esspri__') for x in defining):
                    role='derived_from_defining'; reason='ESSPRI_construct_in_discovery'
                else:
                    role='independent_baseline_anchor'; reason='held_out_with_no_declared_defining_proxy'
            elif c in validation['confounders'] or c==validation['date_column']:
                role='administrative_confounder'; reason='prespecified_adjustment'
            rows.append({'representation_id':identifier,'source_column':c,'validation_role':role,'reason':reason,
                'unit':anchor.get('unit','source_units'),'type':anchor.get('type','source_type')})
        for c,anchor in anchors.items():
            if c not in baseline:
                rows.append({'representation_id':identifier,'source_column':c,'validation_role':'unavailable',
                             'reason':'SOURCE_UNAVAILABLE','unit':anchor.get('unit','source_units'),'type':anchor.get('type','numeric')})
    return pd.DataFrame(rows)


def assert_independent(rows):
    if not rows.empty and not rows.validation_role.eq('independent_baseline_anchor').all():
        raise ValueError('Circular feature in independent anchor validation')
