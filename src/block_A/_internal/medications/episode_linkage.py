"""Patient-specific calendar-year documentary proximity, never exposure."""
import pandas as pd
from .dates import exact_day
from .contracts import KEYS, key_set, spine_signature
from .documentation import pseudonym

LINK_COLUMNS = KEYS + ['documentation_record_id', 'source_kind', 'medication_name', 'canonical_medication_name', 'name_requires_review',
    'matching_reference_kinds', 'selected_reference_kind', 'selected_date',
    'relation_to_episode', 'evidence_level', 'post_anchor', 'qc_flags']


def match(documents, spine, secret):
    key_set(spine)
    spine_signature(spine)
    if documents.documentation_record_id.duplicated().any():
        raise ValueError('duplicate documentation record ids')
    episodes = spine.copy()
    for col in ['episode_start_date', 'episode_end_date', 'clinical_anchor_date']:
        episodes[col] = episodes[col].map(exact_day)
    if (episodes.episode_start_date > episodes.episode_end_date).any():
        raise ValueError('reversed Step00 episode bounds')
    episodes['token'] = episodes.patient_id.map(lambda p: pseudonym(secret, 'patient', str(p)))
    groups = {token: rows for token, rows in episodes.groupby('token')}
    records = []
    for doc in documents.itertuples(index=False):
        group = groups.get(doc.linked_patient_pseudonymous)
        if group is None:
            # Nonclinical original CTDB reports remain in the protected catalogue.
            continue
        for ep in group.itertuples(index=False):
            start, end = ep.episode_start_date, ep.episode_end_date
            if doc.source_kind == 'CTDB':
                if str(doc.original_episode_id) != str(ep.clinical_episode_id):
                    continue
                refs = {'ctdb_visit': exact_day(doc.visit_documentation_date)}
                selected, relation, evidence = 'ctdb_visit', 'original_visit', 'reported_at_original_visit'
            else:
                refs = {kind: exact_day(getattr(doc, kind + '_raw'))
                        for kind in ('order_start', 'order_end', 'initial_consent')}
                refs = {k: d for k, d in refs.items() if pd.notna(d) and
                        start - pd.DateOffset(years=1) <= d <= end + pd.DateOffset(years=1)}
                if not refs:
                    continue
                selected = next(k for k in ('order_start','order_end','initial_consent') if k in refs)
                day = refs[selected]
                relation = 'within_episode' if start <= day <= end else ('before_episode' if day < start else 'after_episode')
                evidence = 'consent_only_review' if selected == 'initial_consent' else 'order_date_proximity'
            day = refs[selected]
            records.append(dict(patient_id=ep.patient_id, clinical_episode_id=ep.clinical_episode_id,
                documentation_record_id=doc.documentation_record_id, source_kind=doc.source_kind,
                medication_name=doc.medication_name,
                canonical_medication_name=doc.canonical_medication_name,
                name_requires_review=bool(doc.name_requires_review), matching_reference_kinds='|'.join(refs),
                selected_reference_kind=selected, selected_date=day,
                relation_to_episode=relation, evidence_level=evidence,
                post_anchor=bool(pd.notna(day) and day > ep.clinical_anchor_date), qc_flags=doc.qc_flags))
    links = pd.DataFrame(records, columns=LINK_COLUMNS)
    if links.duplicated(KEYS + ['documentation_record_id']).any():
        raise ValueError('duplicate document-episode relation')
    return links
