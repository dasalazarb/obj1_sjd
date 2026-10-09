from pathlib import Path
import pytest
from shared.loaders.registry import Registry,SchemaError,Entry
from shared.loaders.csv_loader import read_csv_bytes


@pytest.mark.parametrize('path',['../outside.csv','/tmp/outside.csv','data/raw/anything.csv','tables/12_patient_followup_metrics.csv','qc/12_intervisit_gaps.csv','tables/11_integrated_baseline_patient_level.csv','tables/06_overlap_episode_level.csv','anything.parquet'])
def test_patient_and_escape_paths_denied(tmp_path,path):
    registry=Registry(outputs=tmp_path,repo=tmp_path)
    with pytest.raises(SchemaError): registry.add('bad',{'path':path,'kind':'csv'})


def test_symlink_escape(tmp_path):
    (tmp_path/'external').symlink_to('/tmp')
    registry=Registry(outputs=tmp_path,repo=tmp_path)
    with pytest.raises(SchemaError): registry.add('bad',{'path':'external/any.csv','kind':'csv'})


@pytest.mark.parametrize('column',['patient_id','ids__sex','patient_dob','medical_record_number','date_of_birth'])
def test_identifier_headers_rejected(column):
    with pytest.raises(SchemaError): read_csv_bytes((column+'\nMOCK\n').encode(),Entry('mock',{'path':'MOCK.csv','kind':'csv'}))


def test_blank_na_and_exact_keys():
    entry=Entry('mock',{'path':'MOCK.csv','columns':['key','cell'],'key':['key']})
    frame=read_csv_bytes(b'key,cell\nA,\n A,NA\n',entry)
    assert frame.cell.tolist()==['','NA']; assert frame.key.tolist()==['A',' A']


def test_duplicates_and_missing_columns():
    entry=Entry('mock',{'path':'MOCK.csv','columns':['key','cell'],'key':['key']})
    with pytest.raises(SchemaError): read_csv_bytes(b'key,cell\nA,1\nA,2\n',entry)
    with pytest.raises(SchemaError): read_csv_bytes(b'key\nA\n',entry)


def test_duplicate_headers_cannot_be_silently_renamed():
    entry=Entry('mock',{'path':'MOCK.csv','columns':['key','cell'],'key':['key']})
    with pytest.raises(SchemaError,match='Duplicate source column'):
        read_csv_bytes(b'key,cell,cell\nA,1,2\n',entry)


def test_multiline_identifier_header_rejected_before_fingerprinting(tmp_path):
    from shared.loaders.manifest_loader import fingerprint
    path=tmp_path/'MOCK.csv'
    path.write_text('"note\ncontinued",patient_id\nMOCK,never-ingested\n')
    assert fingerprint(path)==('rejected','Patient identifier columns rejected')
