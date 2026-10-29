"""Exact clinical source joins and separate technical/clinical review gates."""
import numpy as np
import pandas as pd

REQUIRED = {'source_column','semantic_feature_id','family','clinical_domain','subdomain',
            'role','approval','temporal_validity','unit','missingness_origin','release_id'}
CLINICAL = {f'D{i:02d}' for i in range(1, 8)}
RESERVED = {'context','administrative_confounder','post_baseline_outcome','independent_baseline_anchor'}
BASELINE = {'baseline','verified_asof_baseline'}


def clinical_registry(manifest, mapping, primary_features):
    records = mapping.get('features', [])
    for row in records:
        if REQUIRED - row.keys():
            raise ValueError('Clinical map lacks fields: ' + ','.join(sorted(REQUIRED-row.keys())))
        if row['clinical_domain'] not in mapping['domains']:
            raise ValueError('Unknown clinical domain in exact map')
    meta = pd.DataFrame(records)
    if meta.source_column.duplicated().any():
        raise ValueError('Duplicate exact source_column in clinical map')
    if manifest.feature.duplicated().any():
        raise ValueError('Ambiguous encoded feature in clinical catalog')
    data = manifest.merge(meta, on='source_column', how='left', suffixes=('_catalog',''), validate='many_to_one')
    data['technical_family'] = data.family_catalog.str.upper()
    data['clinical_domain'] = data.clinical_domain.fillna('UNMAPPED')
    data['approval_status'] = data.approval.fillna('unmapped')
    data['temporality'] = data.temporal_status
    data['coverage'] = data.pct_nonmissing_baseline
    data['selected_in_primary'] = data.feature.isin(primary_features)
    data['included_R0'] = data.selected_in_primary
    data['mapping_source'] = mapping.get('source', 'exact_versioned_map')
    data['mapping_review_status'] = mapping.get('mapping_review_status','pending_clinical_review')
    data['mapping_release_id'] = mapping.get('mapping_release_id', mapping.get('schema_version'))
    for column, fallback in [('derivation_group','semantic_feature_id'),('instrument_id','subdomain'),('assay_id','semantic_feature_id')]:
        if column not in data: data[column] = data[fallback]
        data[column] = data[column].fillna(data[fallback])
    reasons=[]; exceptions=[]
    for r in data.to_dict('records'):
        errors=[]; notes=[]
        if not r['included'] or not r.get('technical_eligible',False): errors.append('CATALOG_NOT_ELIGIBLE')
        if r['clinical_domain'] not in CLINICAL: errors.append('CONTEXT_ONLY' if r['clinical_domain'] in {'D08','D09'} else 'UNMAPPED')
        if r.get('role') in RESERVED: errors.append('RESERVED_VARIABLE_ROLE')
        if r['temporality'] not in BASELINE or r.get('temporal_validity') not in BASELINE | {'requires_source_verification','pending_review'}:
            errors.append('SOURCE_TEMPORAL_INVALID')
        elif r.get('temporal_validity') not in BASELINE:
            notes.append('BASELINE_VERIFIED_BY_CATALOG_POLICY;mapping_temporal_review_pending')
        if r.get('unit_review_status') in {'incompatible','invalid','multiple_units_not_harmonized','multiple_units_review','incomplete_metadata_review','conflict_detected'} or r.get('unit') in {'incompatible','invalid'}:
            errors.append('UNIT_INCOMPATIBLE')
        elif pd.isna(r.get('unit')) or r.get('unit') in {'','requires_source_review','unknown'}:
            notes.append('UNIT_REVIEW_PENDING;within_source_standardization_only_no_cross_source_unit_comparison')
        if r.get('family') != r['technical_family']: errors.append('FAMILY_MISMATCH')
        if pd.isna(r.get('subdomain')) or not r.get('subdomain'): errors.append('MISSING_SUBDOMAIN')
        reasons.append('|'.join(errors)); exceptions.append('|'.join(notes))
    data['technical_exclusion_reason_v2']=reasons
    data['technical_exceptions']=exceptions
    data['exploratory_technical_eligible']=data.technical_exclusion_reason_v2.eq('')
    data['unit_status']=data.unit_review_status
    data['temporal_status']=data.temporality
    data['source_clinical_approval_status']=data.get('clinical_approval_status',pd.Series('pending',index=data.index))
    source_approved=data.source_clinical_approval_status.eq('approved')
    data['clinically_approved']=(data.exploratory_technical_eligible & data.approval_status.eq('approved') &
        data.mapping_review_status.eq('approved') & data.temporal_validity.isin(BASELINE) &
        ~data.unit.isna() & ~data.unit.isin(['','requires_source_review','unknown']) &
        (data.technical_family.ne('LAB') | source_approved))
    data['clinical_approval_status']=np.where(data.clinically_approved,'approved','pending')
    data['mapping_eligibility']=np.where(data.clinically_approved,'ELIGIBLE',
        np.where(data.exploratory_technical_eligible,'NOT_APPROVED',data.technical_exclusion_reason_v2))
    data['included_R2']=False
    data['exclusion_reason']=data.technical_exclusion_reason_v2.where(~data.exploratory_technical_eligible,'')
    return data


def eligible_domain_map(registry, features, purpose='clinical_release'):
    if purpose not in {'exploratory','clinical_release'}: raise ValueError('Unknown mapping purpose')
    part=registry.loc[registry.feature.isin(features)].copy()
    if len(part)!=len(features) or part.feature.duplicated().any():
        raise ValueError('Incomplete or duplicate clinical feature map')
    part=part.loc[~part.clinical_domain.isin(['D08','D09'])]
    if 'role' in part: part=part.loc[~part.role.isin(RESERVED)]
    if part.empty: raise ValueError('NO_ELIGIBLE_CLINICAL_DOMAINS')
    if purpose=='exploratory':
        if 'exploratory_technical_eligible' not in part: raise ValueError('MISSING_TECHNICAL_MAPPING_AUDIT')
        part=part.loc[part.exploratory_technical_eligible]
        if part.empty: raise ValueError('NO_TECHNICALLY_SAFE_CLINICAL_SOURCES: inspect clinical_feature_mapping_audit')
    else:
        if part.mapping_eligibility.ne('ELIGIBLE').any():
            raise ValueError('UNAPPROVED_OR_UNMAPPED_CLINICAL_DOMAINS: exact reviewed map required')
        if not part.temporal_validity.isin(BASELINE).all(): raise ValueError('CLINICAL_DOMAIN_TEMPORAL_REVIEW_PENDING')
        if 'unit' in part and (part.unit.isna() | part.unit.isin(['','requires_source_review','unknown'])).any():
            raise ValueError('CLINICAL_DOMAIN_UNIT_REVIEW_PENDING')
    if not part.temporality.isin(BASELINE).all(): raise ValueError('CLINICAL_DOMAIN_SOURCE_TEMPORAL_INVALID')
    if part.family.str.upper().ne(part.technical_family).any(): raise ValueError('CLINICAL_DOMAIN_FAMILY_MISMATCH')
    if not part.clinical_domain.isin(CLINICAL).all(): raise ValueError('UNAPPROVED_OR_UNMAPPED_CLINICAL_DOMAINS')
    return part.set_index('feature')


def mapping_counts(registry, features):
    part=registry.loc[registry.feature.isin(features)]
    return dict(n_source_variables=part.source_column.nunique(),n_encoded_columns=len(features),
        n_semantic_constructs=part.derivation_group.nunique(),n_subdomains=part[['clinical_domain','subdomain']].drop_duplicates().shape[0],
        n_domains_effective=part.clinical_domain.nunique())

def domain_coverage(raw, registry, representation):
    rows=[]
    for domain, part in registry.loc[registry.feature.isin(raw)].groupby('clinical_domain'):
        block=raw[part.feature.tolist()]
        rows.append({'representation_id':representation,'clinical_domain':domain,'n_patients':len(raw),
            'n_features':len(block.columns),'n_patients_any_observed':int(block.notna().any(axis=1).sum()),
            'observed_fraction':float(block.notna().to_numpy().mean())})
    return rows


def verify_observed_baseline_dates(baseline,manifest,date_column='clinical_anchor_date'):
    """Use exact assay companion dates where supplied; never admit a future result."""
    index=pd.to_datetime(baseline[date_column],errors='coerce')
    for source in manifest.loc[manifest.included,'source_column'].unique():
        stem=source.rsplit('__',1)[0] if source.startswith('lab__') and source.endswith(('__value','__text')) else source
        for companion in (stem+'__measurement_date',stem+'_measurement_date'):
            if companion not in baseline: continue
            date=pd.to_datetime(baseline[companion],errors='coerce')
            observed=baseline[source].notna()
            if (observed & baseline[companion].notna() & date.isna()).any():
                raise ValueError('UNPARSEABLE_BASELINE_MEASUREMENT_DATE')
            if (observed & date.notna() & (index.isna() | date.gt(index))).any():
                raise ValueError('FUTURE_MEASUREMENT_IN_BASELINE')
