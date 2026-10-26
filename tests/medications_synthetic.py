"""Only invented source records. No patient data or stable production keys."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def producer():
    spec=importlib.util.spec_from_file_location('medication_producer',ROOT/'src/block_A/08_build_medications.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def sources(tmp_path):
    from src.block_A._internal.medications.source_inputs import SOURCES
    spine=pd.DataFrame({'patient_id':pd.Series(['SYNTHETIC_A','SYNTHETIC_A','SYNTHETIC_B'],dtype='string'),
        'clinical_episode_id':pd.Series(['synthetic_e1','synthetic_e2','synthetic_e3'],dtype='string'),
        'clinical_anchor_date':pd.to_datetime(['2020-02-29','2023-04-01','2020-02-29']),
        'episode_start_date':pd.to_datetime(['2020-02-29','2023-04-01','2020-02-29']),
        'episode_end_date':pd.to_datetime(['2020-03-02','2023-04-01','2020-03-02']),
        'clinical_visit':[True]*3})
    sp=tmp_path/'spine.parquet';spine.to_parquet(sp,index=False)
    all_sp=tmp_path/'all-spine.parquet';spine.to_parquet(all_sp,index=False)
    cohort=tmp_path/'cohort.csv';pd.DataFrame({'patient_id':['SYNTHETIC_A','SYNTHETIC_B']}).to_csv(cohort,index=False)
    eda=tmp_path/'eda'
    rows=[
        ['rituximab','2019-02-28','2020-03-01','2020-02-29'],
        ['Rituxan','2020-03-03','','2020-02-29'],
        ['Unknown Synthetic Product','','','2020-02-29'],
        ['Undated Synthetic Product','','',''],
        ['Plaquenil','2020-02-29T12:20:30.123+00:00','',''],
        ['Invalid Synthetic Date','bad date','','2020-02-29'],
        ['Partial Synthetic Date','2020-02','',''],
    ]
    for protocol,relative in SOURCES.items():
        frame=pd.DataFrame(rows,columns=['Medication Name','Order Date/Start Date','Order End Date','Initial Consent Date'])
        frame['MRN']='SYNTHETIC_A';frame['Drug Ingredient']=''
        frame['Protocol Number']='11-D-0172' if protocol=='11D' else '15-D-0051'
        frame['Order ID']=['synthetic-order-'+str(i) for i in range(len(frame))]
        path=eda/relative;path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False)
    visits=pd.DataFrame({'row_id_raw':[1,2], 'patient_id':['SYNTHETIC_A','SYNTHETIC_B'],
        'medications__rx_1_name':['Plaquenil','none'],
        'medications__antiplatelet_1_dose':['must not assign to a slot','']})
    path=eda/'data_analytic/visits_long.parquet';path.parent.mkdir(parents=True,exist_ok=True);visits.to_parquet(path,index=False)
    row_map=pd.DataFrame({'row_id_raw':[1,2], 'patient_id':['SYNTHETIC_A','SYNTHETIC_B'],
        'clinical_episode_id':['synthetic_e1','synthetic_e3'],'collection_date':pd.to_datetime(['2020-02-29','2020-02-29'])})
    row_map.to_parquet(eda/'data_intermediate/clinical_episode_row_map.parquet',index=False)
    audit=tmp_path/'audit';(audit/'restricted').mkdir(parents=True)
    key=audit/'restricted/.pseudonym_key';key.write_bytes(b'synthetic-key-only-never-use-real'+b'x'*32)
    return SimpleNamespace(eda_root=eda,audit_root=audit,spine=sp,all_spine=all_sp,cohort=cohort,
        output=tmp_path/'staging/08_medication_episode_features.parquet',pseudonym_key=key,
        force_rebuild=False,reuse_validated_inputs=False,dry_run=False),spine
