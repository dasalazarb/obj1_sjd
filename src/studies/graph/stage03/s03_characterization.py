"""Aggregate characterization on frozen hard memberships; no discovery fits."""
import numpy as np
import pandas as pd
from .s03_release_reader import hard_members
from .s03_profiles import observed_values, numeric
from .s03_construct_validation import POLICY_VERSION, EVIDENCE


def statistical_role_gates(roles, baseline, membership, manifest, validation):
    roles = roles.copy()
    for index, role in roles.loc[roles.tier.eq('N3_REVIEWED')].iterrows():
        frame = hard_members(membership, role.representation_id).merge(baseline, on='patient_id', validate='one_to_one')
        reasons = list(filter(None, role.blocking_reasons.split('|')))
        if frame.empty: reasons.append('NO_SUPPORTED_HARD_ASSIGNMENTS')
        elif manifest['release_status'] == 'NO_STABLE_STRUCTURE' or not frame.structural_stability_status.eq('stable').all():
            reasons.append('TOPOLOGY_NOT_STABLE_FOR_ANCHOR_INFERENCE')
        elif role.source_column in frame:
            counts = frame.assign(observed=numeric(frame[role.source_column]).notna()).groupby('community_id').observed.sum()
            minimum = int(validation['minimum_group_patients'])
            if not any(n >= minimum and counts.sum()-n >= minimum for n in counts):
                reasons.append('INSUFFICIENT_OBSERVED_GROUP_OR_REST')
        roles.at[index, 'blocking_reasons'] = '|'.join(dict.fromkeys(reasons))
        roles.at[index, 'inferential_eligible'] = not reasons
    roles['inferential_eligibility_scope'] = 'independence_topology_and_observed_support_additional_model_gates_required'
    return roles


def characterization_tables(baseline, membership, roles, manifest, validation, anchors):
    """Source counts use unique columns; patient counts use unions, not sums."""
    counts = []; matrix = []
    panel = {m['source_column']: m for m in validation.get('clinical_marker_panels', [])}
    for identifier in manifest['eligible_representation_ids']:
        frame = hard_members(membership, identifier).merge(baseline, on='patient_id', validate='one_to_one')
        part = roles.loc[roles.representation_id.eq(identifier)].copy()
        part['clinical_domain'] = part.clinical_domain.fillna('')
        for (tier, domain), group_roles in part.groupby(['tier', 'clinical_domain'], dropna=False):
            sources = group_roles.source_column.unique().tolist()
            present = [s for s in sources if s in baseline]
            kinds=group_roles.set_index('source_column').type.to_dict()
            def valid(series, kind):
                clean=numeric(series) if kind in {'numeric','binary'} or pd.api.types.is_numeric_dtype(series) else observed_values(series)
                return clean.isin([0,1]) if kind=='binary' else clean.notna()
            observed = pd.DataFrame({s: valid(baseline[s],kinds[s]) for s in present})
            hard_observed = pd.DataFrame({s: valid(frame[s],kinds[s]) for s in present})
            encoded = {f for r in group_roles.matched_encoded_features for f in r.split('|') if f}
            counts.append(dict(representation_id=identifier, tier=tier, clinical_domain=domain,
                n_sources=len(sources), n_encoded_features=len(encoded), n_baseline=len(baseline), n_hard_supported=len(frame),
                n_patients_evaluable_baseline=int(observed.any(axis=1).sum()),
                n_patients_evaluable_hard=int(hard_observed.any(axis=1).sum()),
                denominator='unique_patients_with_any_observed_source_within_tier_domain',
                n_reviewed_sources=int(group_roles.tier.eq('N3_REVIEWED').sum()),
                n_inferentially_eligible_sources=int(group_roles.inferential_eligible.sum())))
        for role in part.to_dict('records'):
            source = role['source_column']
            if source==validation['pop_column'] or role['clinical_domain'] in {'D08','D09'}: continue
            if role['clinical_domain'] not in {f'D{i:02d}' for i in range(1, 8)} and source not in panel and not role['is_prespecified_anchor']:
                continue
            for community, group in frame.groupby('community_id'):
                raw = group[source] if source in group else pd.Series(np.nan, index=group.index)
                values = observed_values(raw); num = numeric(values)
                numeric_kind = role['type'] in {'numeric', 'binary'} or pd.api.types.is_numeric_dtype(raw) or (values.notna().any() and num.notna().sum() == values.notna().sum())
                valid = num.dropna() if numeric_kind else values.dropna()
                tokens = raw.astype('string').str.strip().str.lower()
                not_tested = tokens.isin(['not tested','not_tested','not_measured','not measured','not_requested','not requested'])
                unknown = tokens.isin(['unknown','unknown_detail','indeterminate','uninterpretable','not reported'])
                observed_tokens=values.dropna().astype('string').str.strip().str.lower()
                binary = role['type'] == 'binary' or (len(observed_tokens)>0 and set(observed_tokens) <= {
                    'positive','negative','pos','neg','reactive','nonreactive','detected','not detected'})
                if binary: valid=num.loc[num.isin([0,1])]
                n_pos = int(num.eq(1).sum()) if binary else np.nan
                n_neg = int(num.eq(0).sum()) if binary else np.nan
                row = {k: role[k] for k in ['representation_id','source_column','clinical_domain','subdomain','instrument','tier',
                    'validation_role','independence_status','relationship_type','evidence_level','unit','blocking_reasons','inferential_eligible','temporal_baseline_status']}
                row.update(community_id=community, n_group=len(group), n_observed=len(valid), n_missing=len(group)-len(valid),
                    n_tested=n_pos+n_neg if binary else np.nan, n_pos=n_pos, n_neg=n_neg,
                    n_unknown=int(unknown.sum()), n_not_tested=int(not_tested.sum()), n_sampled=np.nan,
                    sampling_origin='NOT_DOCUMENTED', observation_origin='frozen_baseline_raw',
                    n_stable_hard=int(group.structural_stability_status.eq('stable').sum()),
                    n_provisional_hard=int(group.structural_stability_status.ne('stable').sum()),
                    data_type='binary' if binary else 'numeric' if numeric_kind else 'categorical',
                    median=float(num.median()) if numeric_kind and len(valid) and not binary else np.nan,
                    q25=float(num.quantile(.25)) if numeric_kind and len(valid) and not binary else np.nan,
                    q75=float(num.quantile(.75)) if numeric_kind and len(valid) and not binary else np.nan,
                    category='', n_category=np.nan, proportion=np.nan,
                    status=('DESCRIPTIVE_PROVISIONAL' if identifier == 'R2' or not group.structural_stability_status.eq('stable').all() else 'DESCRIPTIVE') if len(valid) else 'NOT_ESTIMABLE',
                    inference_status='REVIEW_PENDING' if role['tier'] == 'N3_CANDIDATE' else 'NOT_ESTIMABLE',
                    reason=role['blocking_reasons'], clinical_mapping_status=manifest.get('clinical_mapping', {}).get('review_status','NOT_REVIEWED'),
                    analysis_population='covered_untied_supported_hard_assignment')
                tests = anchors.loc[anchors.representation_id.eq(identifier) & anchors.source_column.eq(source)] if not anchors.empty else pd.DataFrame()
                if not tests.empty and 'community_id' in tests:
                    tests = tests.loc[tests.community_id.astype(str).eq(str(community))]
                    if len(tests) and tests.status.eq('COMPLETE').all(): row['inference_status'] = 'COMPLETE'
                if binary:
                    row.update(proportion=n_pos/(n_pos+n_neg) if n_pos+n_neg else np.nan)
                    matrix.append(row)
                elif not numeric_kind and len(valid):
                    for category, n in valid.astype(str).value_counts().items():
                        matrix.append(dict(row, category=category, n_category=int(n), proportion=float(n/len(valid))))
                else: matrix.append(row)
    matrix = pd.DataFrame(matrix) if matrix else pd.DataFrame(columns=[
        'representation_id','community_id','source_column','clinical_domain','tier','n_group','n_observed','n_missing',
        'n_tested','n_unknown','n_not_tested','n_stable_hard','n_provisional_hard','status','inference_status'])
    audit = roles.loc[roles.is_prespecified_anchor | roles.source_column.eq(validation['pop_column'])].copy()
    audit['anchor_source'] = audit.source_column
    audit['status'] = 'NOT_ESTIMABLE'
    audit['reason'] = audit.blocking_reasons
    for index, role in audit.iterrows():
        tests = anchors.loc[anchors.representation_id.eq(role.representation_id) & anchors.source_column.eq(role.source_column)] if not anchors.empty else pd.DataFrame()
        if not tests.empty and tests.status.eq('COMPLETE').any(): audit.at[index,'status'] = 'COMPLETE'
        elif not tests.empty: audit.at[index,'reason'] = '|'.join(sorted(set(tests.reason.dropna())))
    summary = []
    if not matrix.empty:
        for (identifier, community, tier), group in matrix.groupby(['representation_id','community_id','tier']):
            selected = group.loc[group.source_column.isin(panel)]
            summary.append(dict(representation_id=identifier, community_id=community, tier=tier,
                evidence_level=EVIDENCE[tier], n_sources=group.source_column.nunique(),
                n_group=int(group.n_group.iloc[0]), n_stable_hard=int(group.n_stable_hard.iloc[0]),
                n_provisional_hard=int(group.n_provisional_hard.iloc[0]),
                prespecified_sources='|'.join(sorted(set(selected.source_column))),
                prespecified_results=selected[['source_column','unit','n_group','n_observed','n_missing','median','q25','q75','category','n_category','proportion','inference_status']].to_json(orient='records'),
                n_inference_complete=int(group.drop_duplicates('source_column').inference_status.eq('COMPLETE').sum()),
                interpretation='aggregate_source_description_no_external_clinical_validation',
                status='REVIEW_PENDING' if tier == 'N3_CANDIDATE' else 'DESCRIPTIVE'))
    return {'role_inventory': roles.copy(), 'tier_counts': pd.DataFrame(counts), 'clinical_matrix': matrix,
            'independence_audit': audit, 'evidence_summary': pd.DataFrame(summary) if summary else pd.DataFrame(columns=[
                'representation_id','community_id','tier','evidence_level','n_sources','n_group','prespecified_sources','prespecified_results','status'])}


def characterization_qc(tables, baseline, membership, provenance, frozen_unchanged):
    roles = tables['role_inventory']; matrix = tables['clinical_matrix']
    reconciled = matrix.empty or bool((matrix.n_observed + matrix.n_missing).eq(matrix.n_group).all())
    unique = not roles.duplicated(['representation_id','source_column']).any()
    blocked = bool((~roles.loc[roles.tier.ne('N3_REVIEWED'), 'inferential_eligible']).all())
    if not all([reconciled, unique, blocked, frozen_unchanged]): raise ValueError('CHARACTERIZATION_QC_FAILED')
    return dict(**provenance, classification_policy_version=POLICY_VERSION, status='PASS',
        graph02_unchanged=frozen_unchanged, no_refit=True, source_keys_unique=unique,
        observation_counts_reconciled=reconciled, independence_gates_reconciled=blocked,
        n_baseline=len(baseline), denominators=[dict(representation_id=rep,
            n_hard_supported=len(hard_members(membership,rep))) for rep in sorted(roles.representation_id.unique())],
        tier_counts=tables['tier_counts'].to_dict('records'),
        sources_requiring_review=roles.loc[roles.tier.isin(['N3_CANDIDATE','PENDING_REVIEW']),
            ['representation_id','source_column','tier','blocking_reasons']].to_dict('records'),
        limitations=['No external clinical validation', 'R2 clinical mapping remains provisional',
            'Independence requires documented human review', 'Baseline does not approve longitudinal date semantics',
            'Pop labels are preserved without recoding', 'Internal restricted aggregates require publication review'],
        privacy_class='INTERNAL_RESTRICTED', shareable_export_allowed=False)
