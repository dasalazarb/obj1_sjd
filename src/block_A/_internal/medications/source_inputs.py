"""Recovered full BTRIS CSVs, collision-checked cohort and original CTDB sources."""
from pathlib import Path
import pandas as pd
from . import documentation as d
from .ctdb import build as ctdb_build
from .episode_features import review_name, canonical_name
from .dates import parse_date

SOURCES = {'11D': 'data_intermediate/BTRIS/11D/Medication288444-1.csv',
           '15D': 'data_intermediate/BTRIS/15D/Medication288476-1.csv'}
DATE_FIELDS = {'order_start': 'Order Date/Start Date', 'order_end': 'Order End Date',
               'initial_consent': 'Initial Consent Date'}


def load(eda_root, cohort_path, spine, all_spine, secret):
    from .contracts import key_set, STRUCTURE
    key_set(spine); key_set(all_spine)
    cohort = pd.read_csv(cohort_path, dtype='string')
    if 'patient_id' not in cohort or cohort.patient_id.isna().any() or cohort.patient_id.duplicated().any():
        raise ValueError('invalid cohort')
    eligible = set(cohort.patient_id.map(d.key))
    if '' in eligible or len(eligible) != len(cohort):
        raise ValueError('invalid cohort IDs')
    if set(spine.patient_id.astype(str)) != eligible:
        raise ValueError('cohort differs from clinical Step00 patients')
    if not key_set(spine).issubset(key_set(all_spine)):
        raise ValueError('clinical episodes absent from all-spine')
    authoritative = all_spine.merge(spine[STRUCTURE], on=['patient_id','clinical_episode_id'], suffixes=('_all','_clinical'), validate='one_to_one')
    for col in STRUCTURE[2:]:
        if not authoritative[col+'_all'].astype('string').equals(authoritative[col+'_clinical'].astype('string')):
            raise ValueError('clinical/all Step00 structure disagreement')
    frames = {}
    for protocol, relative in SOURCES.items():
        frame = pd.read_csv(Path(eda_root)/relative, dtype='string', keep_default_na=False)
        if not frame.columns.is_unique or not {'MRN','Medication Name','Drug Ingredient',*DATE_FIELDS.values()}.issubset(frame):
            raise ValueError('missing full BTRIS source columns')
        frames[protocol] = frame
    visits = pd.read_parquet(Path(eda_root)/'data_analytic/visits_long.parquet')
    row_map = pd.read_parquet(Path(eda_root)/'data_intermediate/clinical_episode_row_map.parquet')
    if not any(c in visits for c in d.CTDB_RX_NAMES):
        raise ValueError('CTDB current medication name fields missing')
    patient_col = next((c for c in ['patient_id',d.CTDB_PATIENT_ID,'patient_record_number'] if c in visits), None)
    if patient_col is None:
        raise ValueError('CTDB patient field missing')
    ctdb_ids = visits.rename(columns={patient_col:d.CTDB_PATIENT_ID}) if patient_col != d.CTDB_PATIENT_ID else visits
    linkage = d.validate_serology_style_linkage(frames, ctdb_ids, eligible)
    if linkage['status'] != 'validated':
        raise ValueError('patient linkage validation failed: ' + linkage['reason'])
    records=[]
    selected_counts={}
    parsing={kind:{} for kind in DATE_FIELDS}
    for protocol, frame in frames.items():
        # Filter using the same collision-checked normalization used by validated linkage.
        selected=frame.loc[frame.MRN.map(lambda x:d.serology_patient_id_match(x)[0]).isin(linkage['normalized_mapping'])].reset_index(drop=True)
        selected_counts[protocol]=len(selected)
        source_records=d.btris_records(selected,protocol,linkage,eligible,secret)
        for record, (_, raw) in zip(source_records,selected.iterrows()):
            for kind, field in DATE_FIELDS.items():
                record[kind+'_raw']=d.raw(raw[field])
                parsed=parse_date(raw[field]); status=parsed['status']+':'+parsed['precision']
                parsing[kind][status]=parsing[kind].get(status,0)+1
                for key in ('status','precision','lower_bound','upper_bound'):
                    record[kind+'_'+key]=parsed[key]
            records.append(record)
    if records:
        orders=d.classification.enrich(d.interval_qc(d.duplicate_qc(d.records_frame(records))),d.DRUG_RULES,linkage)
        orders['documentation_record_id']=orders.audit_record_id.map(lambda x:d.pseudonym(secret,'document',x))
        orders['medication_name_raw']=orders.drug_raw.where(orders.drug_raw.fillna('').str.strip().ne(''),orders.ingredient_raw)
        orders['original_episode_id']=''
        orders['visit_documentation_date']=''
    else:
        orders=pd.DataFrame()
    # Only eligible source visits enter reconstruction; preserve exact row-map coverage.
    visits=visits.loc[visits[patient_col].astype(str).isin(eligible)].copy()
    row_map=row_map.loc[row_map.row_id_raw.isin(visits.row_id_raw)].copy()
    ctdb,_,ctdb_qc=ctdb_build(visits,row_map,all_spine,secret)
    if not ctdb.empty:
        ctdb['source_kind']='CTDB'; ctdb['source']='CTDB'
        ctdb['linked_patient_pseudonymous']=ctdb.patient_pseudonym
        ctdb['documentation_record_id']=[d.pseudonym(secret,'ctdb_document',f'{r.row_pseudonym}:{r.slot}') for r in ctdb.itertuples()]
        ctdb['original_episode_id']=ctdb.clinical_episode_id
        ctdb['qc_flags']=ctdb.classification_review_reasons.fillna('')
        for kind in DATE_FIELDS: ctdb[kind+'_raw']=''
    catalog=pd.concat([orders,ctdb],ignore_index=True)
    required=['documentation_record_id','linked_patient_pseudonymous','source_kind','source','ingredient_canonical','classification_confidence','medication_name_raw','qc_flags','original_episode_id','visit_documentation_date',*[k+'_raw' for k in DATE_FIELDS]]
    for col in required:
        if col not in catalog: catalog[col]=pd.Series(dtype='string')
    catalog['medication_name']=[review_name(r.ingredient_canonical,r.medication_name_raw,r.classification_confidence) for r in catalog.itertuples()]
    catalog['canonical_medication_name']=[canonical_name(r.ingredient_canonical,r.classification_confidence) for r in catalog.itertuples()]
    catalog['name_requires_review']=catalog.canonical_medication_name.eq('')
    # Never retain identified MRN, subject name, or source free-text instructions in this catalogue.
    qc={'n_source_rows_by_protocol':{k:len(v) for k,v in frames.items()},
        'n_selected_rows_by_protocol':selected_counts,'n_btris':len(orders),'n_ctdb':len(ctdb),'n_documents':len(catalog), 'date_parsing':parsing,
        'n_unmapped_or_uncertain':int(catalog.name_requires_review.sum()), 'n_validated_canonical_documents':int(catalog.canonical_medication_name.ne('').sum()), 'ctdb':ctdb_qc,
        'n_btris_source_rows_outside_cohort':linkage['n_btris_rows_outside_eligibility']}
    return catalog,qc
