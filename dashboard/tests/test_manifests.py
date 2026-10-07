import json
from pathlib import Path
import pytest
from shared.loaders.manifest_loader import capture_run,current_key,digest
from shared.loaders.registry import MissingOutput,SchemaError
from shared.utils.provenance import evidence_hash,claim_is_reviewed


def test_frozen_cells_until_refresh(mock_run):
    before=mock_run.data('t12_ret').copy(); path=mock_run.registry.path('t12_ret')
    path.write_text(path.read_text().replace(',3,',',4,'))
    assert current_key(mock_run.registry)[0]!=mock_run.key
    assert mock_run.data('t12_ret').equals(before)
    assert capture_run(mock_run.registry).key!=mock_run.key


def test_figure_rejects_mid_session_change(mock_run):
    registry=mock_run.registry; registry.add('mock_figure',{'path':'figures/MOCK.png','kind':'figure'})
    path=registry.path('mock_figure');path.parent.mkdir();path.write_bytes(b'MOCK original')
    run=capture_run(registry);assert run.figure_bytes('mock_figure')==b'MOCK original'
    path.write_bytes(b'MOCK changed')
    with pytest.raises(MissingOutput):run.figure_bytes('mock_figure')


def test_claim_change_requires_human_review(mock_run):
    refs=['t11_overall:Cohort / demographics|N patients@Summary#raw']
    claim={'status':'reviewed','reviewed_by':'MOCK reviewer','reviewed_hash':evidence_hash(refs,mock_run),'evidence_refs':refs}
    assert claim_is_reviewed(claim,mock_run)
    mock_run.tables['t11_overall'].loc[mock_run.tables['t11_overall'].Variable.eq('N patients'),'Summary']='4'
    assert not claim_is_reviewed(claim,mock_run)


def test_incomplete_manifest_rejected(mock_run):
    path=mock_run.registry.path('run_manifest');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'run_id':'MOCK_run','status':'incomplete','assets':[]}))
    with pytest.raises(SchemaError):capture_run(mock_run.registry)


def test_manifest_checksums_and_coverage(mock_run):
    registry=mock_run.registry
    assets=[{'id':rid,'path':entry.path,'sha256':digest(registry.path(rid).read_bytes())} for rid,entry in registry.entries.items() if rid!='run_manifest' and registry.path(rid).is_file()]
    path=registry.path('run_manifest');path.write_text(json.dumps({'run_id':'MOCK_complete','status':'complete','assets':assets}))
    assert capture_run(registry).run_id=='MOCK_complete'
    assets[0]['sha256']='bad';path.write_text(json.dumps({'run_id':'MOCK_complete','status':'complete','assets':assets}))
    with pytest.raises(SchemaError):capture_run(registry)


@pytest.mark.parametrize('content',['{invalid JSON','{}','[]'])
def test_present_unreadable_manifest_never_falls_back_to_synthesized(mock_run,content):
    mock_run.registry.path('run_manifest').write_text(content)
    with pytest.raises(SchemaError,match='manifest cannot be read'):
        capture_run(mock_run.registry)


@pytest.mark.parametrize('assets',[{},[{}],[None],[{'id':'t11_overall','path':'MOCK.csv','sha256':False}]])
def test_malformed_assets_fail_with_schema_error(mock_run,assets):
    mock_run.registry.path('run_manifest').write_text(json.dumps({'run_id':'MOCK','status':'complete','assets':assets}))
    with pytest.raises(SchemaError): capture_run(mock_run.registry)


def test_duplicate_manifest_assets_rejected(mock_run):
    registry=mock_run.registry
    entry=registry.get('t11_overall')
    asset={'id':entry.id,'path':entry.path,'sha256':digest(registry.path(entry.id).read_bytes())}
    registry.path('run_manifest').write_text(json.dumps({'run_id':'MOCK','status':'complete','assets':[asset,asset]}))
    with pytest.raises(SchemaError,match='Duplicate'):
        capture_run(registry)


def test_invalid_text_output_degrades_without_breaking_other_cells(mock_run):
    mock_run.registry.path('t11_avail').write_bytes(b'\xffinvalid UTF-8')
    run=capture_run(mock_run.registry)
    assert 't11_avail' in run.errors
    assert not run.data('t11_overall').empty
