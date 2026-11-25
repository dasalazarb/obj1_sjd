"""Panel-specific observation lineage. Never infer availability from imputed values."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
import pandas as pd
from .s02_feature_registry import BASELINE, RESERVED, verify_observed_baseline_dates


class NotEstimable(ValueError):
    """Expected lack of methodological support, distinct from an implementation failure."""


@dataclass
class ObservationPanel:
    mask: pd.DataFrame
    states: pd.DataFrame
    metadata: pd.DataFrame
    lineage: pd.DataFrame
    panel_id: str
    fingerprint: str


def panel_metadata(raw, registry, reference):
    if not raw.columns.is_unique or not raw.index.is_unique:
        raise ValueError('AMBIGUOUS_PANEL')
    meta=registry.loc[registry.feature.isin(raw)].copy()
    if len(meta)!=raw.shape[1] or meta.feature.duplicated().any():
        raise ValueError('INCOMPLETE_PANEL_LINEAGE')
    meta=meta.set_index('feature').loc[list(raw)]
    temporal=meta.get('temporality',meta.get('temporal_status'))
    if temporal is None or not temporal.isin(BASELINE).all():
        raise NotEstimable('PANEL_TEMPORAL_INVALID')
    if reference=='R2' and (meta.clinical_domain.isin(['D08','D09','UNMAPPED']).any() or meta.role.isin(RESERVED).any()):
        raise ValueError('R2_PANEL_HAS_RESERVED_SOURCES')
    # The sensitivity estimator must never acquire outcomes, even under an inclusive legacy policy.
    from .._feature_catalog import OUTCOMES
    # Exact upstream outcome identities: defining symptom components retain
    # their existing clinical role and are not silently removed from the panel.
    prohibited=meta.source_column.str.lower().isin(OUTCOMES | {'pop','essdai','esspri','pop_status','essdai_total','esspri_total'})
    if prohibited.any() or meta.role.eq('post_baseline_outcome').any():
        raise NotEstimable('OUTCOME_PREDICTORS_FORBIDDEN')
    return meta


def build_panel(raw, registry, reference, baseline=None, date_column='clinical_anchor_date'):
    meta=panel_metadata(raw,registry,reference)
    if baseline is not None and date_column in baseline:
        m=meta.reset_index(); m['included']=True
        verify_observed_baseline_dates(baseline,m,date_column)
    # Connected components coalesce exact source/semantic/derivation aliases. An
    # instrument is provenance, not a reason to merge all its distinct subscales.
    parents={c:c for c in raw}
    def root(c):
        while parents[c]!=c:
            parents[c]=parents[parents[c]]; c=parents[c]
        return c
    for field in ('source_column','semantic_feature_id','derivation_group'):
        if field not in meta: continue
        for value,part in meta.groupby(field,sort=True):
            if pd.isna(value) or not str(value).strip(): continue
            columns=list(part.index)
            for c in columns[1:]: parents[root(c)]=root(columns[0])
    groups={}
    for c in raw: groups.setdefault(root(c),[]).append(c)
    masks={}; states={}; units=[]; lineage=[]
    for _,columns in sorted(groups.items()):
        part=meta.loc[columns]; row=part.iloc[0]
        sources=sorted(part.source_column.unique())
        key='unit_'+hashlib.sha256('|'.join(sorted(columns)).encode()).hexdigest()[:16]
        per_source=[]; supplied=[]
        for source in sources:
            cols=part.index[part.source_column.eq(source)]
            observed=raw[cols].notna().any(axis=1)
            state=pd.Series(np.where(observed,'observed','missing'),index=raw.index,dtype=object)
            if baseline is not None:
                companion=next((c for c in (source+'__observation_status',source+'_status') if c in baseline),None)
                tokens=baseline[companion].astype('string').str.lower().str.strip() if companion else pd.Series(pd.NA,index=raw.index,dtype='string')
                if source in baseline:
                    original=baseline[source].astype('string').str.lower().str.strip()
                    tokens=tokens.fillna(original.where(original.isin(['unknown','unknown_detail','partial','not_measured','not_requested','outside_window','source_absent','structural'])))
                state.loc[tokens.isin(['unknown','unknown_detail','indeterminate','uninterpretable'])]='unknown_detail'
                state.loc[tokens.eq('partial')]='partial'
                for absence in ('not_measured','not_requested','outside_window','source_absent','structural'):
                    state.loc[tokens.eq(absence)]=absence
            # Uncertainty cannot be converted into a measured clinical negative.
            observed=observed & state.isin(['observed','partial'])
            per_source.append(observed.astype(float)); supplied.append(state)
            lineage.append(dict(panel_id=reference+'_fixed',observation_unit=key,source_column=source,
                encoded_columns='|'.join(cols),n_encoded_columns=len(cols),instrument_id=row.get('instrument_id','UNKNOWN'),
                semantic_feature_id=row.get('semantic_feature_id',source),derivation_group=row.get('derivation_group',source)))
        block=pd.concat(per_source,axis=1)
        masks[key]=block.max(axis=1)
        state=pd.Series(np.where(block.min(axis=1).eq(1),'observed',np.where(block.max(axis=1).eq(1),'partial','missing')),index=raw.index)
        source_states=pd.concat(supplied,axis=1)
        state.loc[source_states.eq('unknown_detail').any(axis=1)]='unknown_detail'
        state.loc[source_states.eq('partial').any(axis=1) & ~state.eq('unknown_detail')]='partial'
        for absence in ('not_measured','not_requested','outside_window','source_absent','structural'):
            state.loc[source_states.eq(absence).all(axis=1)]=absence
        states[key]=state
        units.append(dict(observation_unit=key,columns=tuple(columns),sources=tuple(sources),
            family=row.get('technical_family',row.get('family','UNKNOWN')),clinical_domain=row.clinical_domain,
            subdomain=row.get('subdomain','UNKNOWN'),instrument_id=row.get('instrument_id','UNKNOWN'),
            n_encoded_columns=len(columns),n_source_variables=len(sources),
            administrative=bool(part.role.eq('administrative_confounder').any() or part.clinical_domain.eq('D09').any())))
    fingerprint=hashlib.sha256(json.dumps(meta.fillna('').reset_index().astype(str).to_dict('records'),sort_keys=True).encode()).hexdigest()
    return ObservationPanel(pd.DataFrame(masks,index=raw.index),pd.DataFrame(states,index=raw.index),
        pd.DataFrame(units).set_index('observation_unit'),pd.DataFrame(lineage),reference+'_fixed',fingerprint)


def observation_summary(panel):
    rows=[]
    for key,unit in panel.metadata.iterrows():
        for state,count in panel.states[key].value_counts().items():
            rows.append(dict(panel_id=panel.panel_id,observation_unit=key,state=state,n_patients=len(panel.mask),
                n_state=int(count),family=unit.family,clinical_domain=unit.clinical_domain,
                instrument_id=unit.instrument_id,n_source_variables=unit.n_source_variables,n_encoded_columns=unit.n_encoded_columns,
                source_columns='|'.join(unit.sources),availability_origin='original_baseline_before_imputation'))
    return rows


# Only explicit upstream codes carry a reason. In particular, "not_requested"
# alone does not say why the test was not requested.
REASON_CODES = {
    'not_requested_clinical_indication': 'clinical_documented',
    'clinical_not_requested': 'clinical_documented',
    'not_scheduled_by_protocol': 'administrative',
    'not_scheduled_by_era': 'administrative',
    'not_scheduled_for_visit_type': 'administrative',
    'protocol_not_scheduled': 'administrative',
    'administrative': 'administrative',
    'structural': 'structural',
    'not_applicable': 'structural',
    'partial': 'partial',
    'uninterpretable': 'invalid',
    'invalid': 'invalid',
    'outside_window': 'outside_window',
}


@dataclass
class ReasonPanel:
    observation: ObservationPanel
    cells: pd.DataFrame
    eligible: pd.DataFrame
    valid: pd.DataFrame
    masks: dict
    fingerprint: str


def classify_panel(raw, registry, reference, baseline, date_column='clinical_anchor_date'):
    """Read typed reasons and retain per-source evidence before reconstruction."""
    if not raw.index.equals(baseline.index):
        raise ValueError('BASELINE_PANEL_INDEX_MISMATCH')
    panel = build_panel(raw, registry, reference, baseline, date_column)
    meta = panel_metadata(raw, registry, reference)
    eligible = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    valid = raw.notna()
    source_reasons = {}
    cells = []
    for lineage in panel.lineage.itertuples(index=False):
        source = lineage.source_column
        columns = meta.index[meta.source_column.eq(source)]
        available = raw[columns].notna()
        observed = available.any(axis=1)
        reason = pd.Series('unknown', index=raw.index)
        evidence = pd.Series('', index=raw.index)
        original_state = pd.Series(np.where(observed, 'observed', 'missing'), index=raw.index)
        stem = source.rsplit('__', 1)[0] if source.startswith('lab__') and source.endswith(('__value','__text')) else source
        if source in baseline:
            token = baseline[source].astype('string').str.strip().str.lower()
            explicit = token.map(REASON_CODES)
            original_state = token.where(explicit.notna(), original_state)
            reason = explicit.where(explicit.notna(), reason)
            evidence = (source+'='+token).where(explicit.notna(), evidence)
        fields = [stem+'__observation_status', stem+'_status', stem+'__missingness_reason',
                  source+'__observation_status', source+'_status', source+'__missingness_reason']
        for field in dict.fromkeys(fields):
            if field not in baseline:
                continue
            token = baseline[field].astype('string').str.strip().str.lower()
            if not field.endswith('__missingness_reason'):
                original_state = token.where(token.notna(), original_state)
            mapped = token.map(REASON_CODES)
            conflict = reason.eq('conflicting_evidence') | (mapped.notna() & evidence.ne('') & mapped.ne(reason))
            reason = mapped.where(mapped.notna(), reason).mask(conflict, 'conflicting_evidence')
            evidence = (field+'='+token).where(mapped.notna(), evidence).mask(conflict, 'conflicting_companion_codes')
        disallowed = reason.isin(['invalid', 'outside_window', 'conflicting_evidence'])
        contradicts = observed & (disallowed | reason.isin(['clinical_documented', 'administrative', 'structural']))
        reason = reason.mask(observed & reason.eq('unknown'), 'observed').mask(contradicts, 'contradiction')
        target = reason.isin(['unknown', 'clinical_documented', 'partial', 'observed'])
        eligible[columns] = (~available).mul(target, axis=0)
        valid[columns] = available.mul(~contradicts, axis=0)
        source_reasons[source] = reason
        row = meta.loc[columns[0]]
        timestamp = next((column for column in (row.get('measurement_date_column'), stem+'__measurement_date',
                                                 stem+'_measurement_date') if isinstance(column,str) and column in baseline), None)
        cells.append(pd.DataFrame({
            'patient_key': baseline.patient_id if 'patient_id' in baseline else raw.index,
            'index_date': baseline[date_column] if date_column in baseline else pd.NaT,
            'source_column': source, 'observation_unit': lineage.observation_unit,
            'semantic_feature_id': row.get('semantic_feature_id', source),
            'derivation_group': row.get('derivation_group', source),
            'instrument_id': row.get('instrument_id', 'UNKNOWN'),
            'family': row.get('technical_family', row.get('family', 'UNKNOWN')),
            'domain': row.clinical_domain, 'subdomain': row.get('subdomain', 'UNKNOWN'),
            'original_observation_state': original_state, 'reason_class': reason,
            'reason_evidence': evidence,
            'reason_confidence': np.select([evidence.ne('').to_numpy(dtype=bool), reason.eq('observed').to_numpy(dtype=bool)],
                                           ['explicit_code','observed_source'],
                                           default='unverified'),
            'eligible_to_measure': target.astype('boolean').mask(reason.eq('unknown')),
            'eligible_to_impute': eligible[columns].any(axis=1),
            'observed_value_valid': valid[columns].any(axis=1),
            'source_timestamp': baseline[timestamp] if isinstance(timestamp, str) and timestamp in baseline else pd.NaT,
            'provenance': row.get('provenance', source),
        }))
    masks = {name: pd.DataFrame(index=raw.index) for name in
             ('clinical_documented', 'clinical_plus_unknown', 'administrative_only', 'unknown_only')}
    for unit, row in panel.metadata.iterrows():
        reason = pd.concat([source_reasons[source] for source in row.sources], axis=1)
        observed = reason.eq('observed').all(axis=1)
        clinical = reason.eq('clinical_documented').all(axis=1)
        uncertain = reason.isin(['unknown', 'partial', 'invalid']).all(axis=1)
        administrative = reason.isin(['administrative', 'structural']).all(axis=1)
        if not row.administrative:
            masks['clinical_documented'][unit] = np.select([observed, clinical], [0., 1.], default=np.nan)
            masks['clinical_plus_unknown'][unit] = np.select([observed, clinical, uncertain], [0., 1., 2.], default=np.nan)
            masks['unknown_only'][unit] = np.select([observed, uncertain], [0., 1.], default=np.nan)
        masks['administrative_only'][unit] = np.select([observed, administrative], [0., 1.], default=np.nan)
    # Keep the historical all-state definition as its own explicit comparator.
    masks['legacy_states'] = panel.states.loc[:, ~panel.metadata.administrative].copy()
    frame = pd.concat(cells, ignore_index=True)
    state_hash = pd.util.hash_pandas_object(frame[['patient_key', 'index_date', 'source_column',
                                                   'reason_class', 'reason_evidence']], index=False)
    signature = hashlib.sha256(panel.fingerprint.encode() + state_hash.to_numpy().tobytes()).hexdigest()
    return ReasonPanel(panel, frame, eligible, valid, masks, signature)
