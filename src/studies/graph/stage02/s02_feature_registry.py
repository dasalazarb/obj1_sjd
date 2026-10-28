"""Exact clinical metadata joins, independent of technical-family heuristics."""
import numpy as np
import pandas as pd

REQUIRED = {'source_column','semantic_feature_id','family','clinical_domain','subdomain',
            'role','approval','temporal_validity','unit','missingness_origin','release_id'}


def clinical_registry(manifest, mapping, primary_features):
    records = mapping.get('features', [])
    for row in records:
        if REQUIRED - row.keys():
            raise ValueError('Clinical map lacks fields: ' + ','.join(sorted(REQUIRED-row.keys())))
        if row['clinical_domain'] not in mapping['domains']:
            raise ValueError('Unknown clinical domain in exact map')
    meta = pd.DataFrame(records, columns=sorted(REQUIRED))
    if meta.source_column.duplicated().any():
        raise ValueError('Duplicate exact source_column in clinical map')
    data = manifest.merge(meta, on='source_column', how='left', suffixes=('_catalog',''), validate='many_to_one')
    data['technical_family'] = data.family_catalog.str.upper()
    data['clinical_domain'] = data.clinical_domain.fillna('UNMAPPED')
    data['approval_status'] = data.approval.fillna('unmapped')
    data['temporality'] = data.temporal_status
    data['coverage'] = data.pct_nonmissing_baseline
    data['selected_in_primary'] = data.feature.isin(primary_features)
    data['mapping_source'] = mapping.get('source', 'exact_versioned_map')
    data['mapping_review_status'] = mapping.get('mapping_review_status','pending_clinical_review')
    data['mapping_eligibility'] = 'ELIGIBLE'
    data.loc[~data.included, 'mapping_eligibility'] = 'NOT_ELIGIBLE'
    data.loc[data.clinical_domain.eq('UNMAPPED'), 'mapping_eligibility'] = 'UNMAPPED'
    data.loc[data.clinical_domain.isin(['D08','D09']), 'mapping_eligibility'] = 'CONTEXT_ONLY'
    data.loc[data.included & ~data.clinical_domain.isin(['D08','D09','UNMAPPED']) &
             (data.approval_status.ne('approved') | data.mapping_review_status.ne('approved')),
             'mapping_eligibility'] = 'NOT_APPROVED'
    return data


def eligible_domain_map(registry, features):
    part = registry.loc[registry.feature.isin(features)].copy()
    if len(part) != len(features) or part.feature.duplicated().any():
        raise ValueError('Incomplete or duplicate clinical feature map')
    part = part.loc[~part.clinical_domain.isin(['D08','D09'])]
    if 'role' in part:
        part=part.loc[~part.role.isin(['independent_baseline_anchor','post_baseline_outcome','administrative_confounder','context'])]
    if part.empty:
        raise ValueError('NO_ELIGIBLE_CLINICAL_DOMAINS')
    if part.mapping_eligibility.ne('ELIGIBLE').any():
        raise ValueError('UNAPPROVED_OR_UNMAPPED_CLINICAL_DOMAINS: exact reviewed map required')
    if not part.temporal_validity.isin(['baseline','verified_asof_baseline']).all():
        raise ValueError('CLINICAL_DOMAIN_TEMPORAL_REVIEW_PENDING')
    if not part.temporality.isin(['baseline','verified_asof_baseline']).all():
        raise ValueError('CLINICAL_DOMAIN_SOURCE_TEMPORAL_INVALID')
    if 'unit' in part and (part.unit.isna() | part.unit.isin(['','requires_source_review','unknown'])).any():
        raise ValueError('CLINICAL_DOMAIN_UNIT_REVIEW_PENDING')
    if part.family.str.upper().ne(part.technical_family).any():
        raise ValueError('CLINICAL_DOMAIN_FAMILY_MISMATCH')
    if part.subdomain.isna().any() or part.subdomain.eq('').any():
        raise ValueError('Missing clinical subdomain')
    # Separate exact assays may share a construct only via an explicit approved subdomain.
    return part.set_index('feature')


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
        if not source.startswith('lab__') or not source.endswith(('__value','__text')):
            continue
        stem=source.rsplit('__',1)[0]
        companion=stem+'__measurement_date'
        if companion not in baseline:
            continue  # Curated episode lineage remains required; no date is invented.
        date=pd.to_datetime(baseline[companion],errors='coerce')
        observed=baseline[source].notna()
        if (observed & baseline[companion].notna() & date.isna()).any():
            raise ValueError('UNPARSEABLE_BASELINE_MEASUREMENT_DATE')
        if (observed & date.notna() & (index.isna() | date.gt(index))).any():
            raise ValueError('FUTURE_MEASUREMENT_IN_BASELINE')
