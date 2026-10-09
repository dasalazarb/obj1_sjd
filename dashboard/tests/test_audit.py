import json
import pytest
from scripts.audit_outputs import audit
from scripts.new_study import scaffold


def test_audit_maps_only_explicit_metadata(tmp_path):
    root=tmp_path/'MOCK_outputs';(root/'figures').mkdir(parents=True)
    figure=root/'figures/ambiguous.png';figure.write_bytes(b'MOCK image')
    catalogue=[{'variable_id':'mock_lab','display_name':'MOCK Laboratory'}]
    assert audit(root,tmp_path/'audit',catalogue)==[]
    figure.with_suffix('.meta.json').write_text(json.dumps({'variable_id':'mock_lab','view':'longitudinal','source_script':'MOCK.py'}))
    candidates=audit(root,tmp_path/'audit',catalogue)
    assert candidates[0]['status']=='candidate';assert figure.read_bytes()==b'MOCK image'


def test_audit_cannot_write_upstream(tmp_path):
    with pytest.raises(ValueError):audit(tmp_path,tmp_path/'audit',[])


def test_scaffold_preserves_existing(tmp_path):
    path=scaffold(tmp_path,'study','mock','MOCK Study')
    assert (path/'metadata/claims.yml').read_text()=='[]\n'
    with pytest.raises(ValueError):scaffold(tmp_path,'study','mock','MOCK changed')
