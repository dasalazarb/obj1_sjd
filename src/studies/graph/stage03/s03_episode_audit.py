"""Endpoint-specific temporal reconciliation, before membership or missingness filters.

Step10's public schema supplies a clinical spine anchor, not a measurement-date
attestation for every attached instrument. An endpoint policy must cite reviewed
source evidence; the presence of a date/episode column alone is never approval.
No source records are changed. Patient-level diagnostics stay in memory.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .s03_profiles import numeric, observed_values

VERSION = 'graph03_episode_resolution_v1'


@dataclass
class EpisodeResolution:
    observations: pd.DataFrame
    audits: dict
    unresolved: pd.DataFrame
    status: str
    reason: str
    date_review: str


def validate_unique_patients(frame, label):
    if 'patient_id' not in frame or frame.patient_id.isna().any():
        raise ValueError(label + '_INVALID_PATIENT_ID')
    if frame.patient_id.astype('string').duplicated().any():
        raise ValueError(label + '_NOT_ONE_TO_ONE')


def resolve_endpoint_episodes(master, baseline, endpoint, config):
    """Return post-index visits plus identifier-free audits and internal exclusions.

    Policies are endpoint-specific and opt-in, after review of the real schema.
    Exact repeats require verified identity keys AND identical endpoint/date/provenance records.
    Distinct same-day episodes are audited but excluded from independent visits.
    """
    validate_unique_patients(baseline, 'BASELINE')
    if 'patient_id' not in master or master.patient_id.isna().any():
        raise ValueError('SOURCE_INVALID_PATIENT_ID')
    temporal = config.get('temporal_resolution', {})
    policy = temporal.get('endpoint_policies', {}).get(endpoint, {})
    date = policy.get('date_field', config['date_column'])
    index_field = config['date_column']
    available = endpoint in master and date in master and index_field in baseline
    verified = (policy.get('date_semantics_review_status') == 'VERIFIED'
                and policy.get('date_semantics') == 'endpoint_measurement_date'
                and bool(policy.get('evidence')))
    review = 'VERIFIED' if verified else 'TEMPORAL_PROVENANCE_UNVERIFIED'
    reason = ('' if verified else 'UNVERIFIED_FOLLOWUP_MEASUREMENT_DATE') if available else 'ENDPOINT_OR_INDEX_DATE_UNAVAILABLE'
    index = baseline[['patient_id']].copy()
    index['patient_id'] = index.patient_id.astype('string')
    index['index_date'] = pd.to_datetime(baseline[index_field], errors='coerce', utc=True) if index_field in baseline else pd.Series(pd.NaT,index=index.index,dtype='datetime64[ns, UTC]')
    if available and index.index_date.isna().any():
        reason = 'UNVERIFIED_BASELINE_INDEX_DATE'
    raw = master.copy().reset_index(drop=True)
    raw['patient_id'] = raw.patient_id.astype('string')
    raw['_source_row'] = np.arange(len(raw))
    raw['visit_date'] = pd.to_datetime(raw[date], errors='coerce', utc=True) if date in raw else pd.Series(pd.NaT,index=raw.index,dtype='datetime64[ns, UTC]')
    raw['value'] = (observed_values(raw[endpoint]) if endpoint == config.get('pop_column') else numeric(raw[endpoint])) if endpoint in raw else np.nan
    keys = policy.get('episode_key_fields', [])
    identity = bool(keys) and policy.get('identity_review_status') == 'VERIFIED' and all(k in raw for k in keys)
    if verified and (not identity or any(k not in raw for k in policy.get('provenance_fields', []))):
        reason = 'TEMPORAL_PROVENANCE_UNVERIFIED'
    # A verified episode identity cannot move across measurement dates.
    inconsistent=set()
    if identity:
        for _,part in raw.groupby(list(dict.fromkeys(['patient_id',*keys])),dropna=False):
            if part.visit_date.nunique(dropna=False)>1: inconsistent.update(part.index)
    comparison_fields=list(dict.fromkeys([*keys,date,endpoint,*policy.get('provenance_fields',[])]))
    comparison_fields=[field for field in comparison_fields if field in raw]
    exclusions = []
    collisions = []
    keep = []
    collapsed = 0
    resolved_groups = 0
    unresolved_groups = 0
    for _, group in raw.groupby(['patient_id', 'visit_date'], dropna=False, sort=False):
        classification = 'VERIFIED_SINGLE_OBSERVATION'
        group_reason = ''
        if group.visit_date.isna().any():
            classification = group_reason = 'INVALID_OR_MISSING_DATE'
        elif not identity or group[keys].isna().any().any():
            classification = group_reason = 'UNRESOLVED_IDENTITY'
        elif any(i in inconsistent for i in group.index):
            classification = group_reason = 'CONFLICTING_DATES_SAME_EPISODE'
        elif len(group) > 1:
            n_episodes = len(group[keys].drop_duplicates())
            if n_episodes > 1:
                classification = 'DISTINCT_EPISODES_SAME_DAY'
                group_reason = 'AMBIGUOUS_FOLLOWUP_ORDER' if endpoint == config.get('pop_column') else 'SAME_DAY_DISTINCT_EPISODES_REQUIRE_POLICY'
            elif group.value.nunique(dropna=False) > 1:
                classification = 'CONFLICTING_VALUES_SAME_EPISODE'
                group_reason = 'UNRESOLVED_CONFLICT'
            elif len(group[comparison_fields].drop_duplicates()) == 1:
                classification = 'EXACT_DUPLICATE_SAME_EPISODE'
                collapsed += len(group)-1
                resolved_groups += 1
            else:
                classification = group_reason = 'UNRESOLVED_IDENTITY'
        n_episodes = len(group[keys].drop_duplicates()) if identity else np.nan
        if len(group)>1 or group_reason:
            collisions.append(dict(endpoint=endpoint, classification=classification, n_rows=len(group),
                n_patients=1, n_patient_dates=1, n_distinct_episodes=n_episodes,
                n_conflicting_groups=int(classification in {'CONFLICTING_VALUES_SAME_EPISODE','CONFLICTING_DATES_SAME_EPISODE'}),
                n_unresolved_groups=int(bool(group_reason)), n_observed_values=int(group.value.notna().sum()),
                status='NOT_ESTIMABLE' if group_reason or reason else 'DESCRIPTIVE', _patient=group.patient_id.iloc[0],
                _episodes=set(group[list(dict.fromkeys(['patient_id',*keys]))].dropna().itertuples(index=False,name=None)) if identity else None))
        if group_reason or reason:
            excluded = group.copy()
            excluded['reason'] = reason or group_reason
            excluded['collision_classification'] = classification
            exclusions.append(excluded)
            unresolved_groups += 1
        else:
            # Keep every original row reference internally for technical repeats.
            selected=group.iloc[:1].copy()
            selected['_source_rows']=[tuple(group._source_row)]
            keep.append(selected)
    visits = pd.concat(keep, ignore_index=True) if keep else raw.iloc[:0].copy()
    visits = visits.merge(index, on='patient_id', validate='many_to_one')
    visits['time_years'] = (visits.visit_date-visits.index_date).dt.total_seconds()/(365.25*86400)
    visits = visits.loc[visits.time_years.gt(0)].copy()
    flag = config.get('baseline_flag')
    if flag in visits:
        visits = visits.loc[~visits[flag].fillna(False).astype(bool)].copy()
    unresolved = pd.concat(exclusions, ignore_index=True) if exclusions else raw.iloc[:0].assign(reason=pd.Series(dtype=str))
    if not unresolved.empty:
        unresolved = unresolved.merge(index, on='patient_id', validate='many_to_one')
        unresolved['post_index'] = unresolved.visit_date.gt(unresolved.index_date)
    summary = []
    collision_frame = pd.DataFrame(collisions)
    if not collision_frame.empty:
        for classification, part in collision_frame.groupby('classification'):
            record = dict(endpoint=endpoint, classification=classification, n_patients=part._patient.nunique(),
                          status='NOT_ESTIMABLE' if part.status.eq('NOT_ESTIMABLE').any() else 'DESCRIPTIVE')
            for col in ['n_rows','n_patient_dates','n_distinct_episodes','n_conflicting_groups','n_unresolved_groups','n_observed_values']:
                record[col] = part[col].sum(min_count=1)
            record['n_distinct_episodes']=len(set().union(*part._episodes)) if identity else np.nan
            summary.append(record)
    if not summary:
        summary=[dict(endpoint=endpoint,classification='NO_COLLISIONS',n_rows=0,n_patients=0,n_patient_dates=0,
            n_distinct_episodes=0,n_conflicting_groups=0,n_unresolved_groups=0,n_observed_values=0,status='DESCRIPTIVE')]
    source = pd.DataFrame([dict(endpoint=endpoint,date_field_used=date,date_semantics_review_status=review,
        source_grain=policy.get('source_grain','verified_endpoint_observation_identity') if identity else 'UNVERIFIED',
        source_population='all_source_rows_before_frozen_patient_filter',
        temporal_evidence=policy.get('evidence','Step10 clinical spine anchor does not attest endpoint measurement time'),
        n_rows=len(raw),n_patients=raw.patient_id.nunique(),n_dates=raw.visit_date.nunique(),
        n_invalid_or_missing_dates=int(raw.visit_date.isna().sum()),n_observed_values=int(raw.value.notna().sum()),
        status='NOT_ESTIMABLE' if reason else 'DESCRIPTIVE',reason=reason)])
    audit = pd.DataFrame([dict(endpoint=endpoint,n_raw_records=len(raw),n_records_after_technical_collapse=len(raw)-collapsed,
        n_verified_visits=len(visits),n_verified_patients=visits.patient_id.nunique(),
        n_collapsed_technical_records=collapsed,n_resolved_groups=resolved_groups,n_unresolved_groups=unresolved_groups,
        n_excluded_records=len(unresolved),policy_version=VERSION,policy='verified_identity_exact_rows_only_no_same_day_pseudoreplication',
        status='NOT_ESTIMABLE' if reason else 'DESCRIPTIVE',reason=reason)])
    reasons=[]
    if not unresolved.empty:
        for why, part in unresolved.groupby('reason'):
            reasons.append(dict(endpoint=endpoint,reason=why,n_rows=len(part),n_patients=part.patient_id.nunique(),
                n_patient_dates=len(part[['patient_id','visit_date']].drop_duplicates()),status='NOT_ESTIMABLE'))
    return EpisodeResolution(visits, {'temporal_source':source,'collision_summary':pd.DataFrame(summary),
        'resolution':audit,'unresolved_reasons':pd.DataFrame(reasons,columns=['endpoint','reason','n_rows','n_patients','n_patient_dates','status'])},
        unresolved, 'NOT_ESTIMABLE' if reason else ('PARTIALLY_RECONCILED' if unresolved_groups else 'RECONCILED'), reason, review)


def resolve_all_endpoints(master, baseline, config):
    return {endpoint:resolve_endpoint_episodes(master,baseline,endpoint,config)
            for endpoint in dict.fromkeys([*config['longitudinal_endpoints'],config['pop_column']])}
