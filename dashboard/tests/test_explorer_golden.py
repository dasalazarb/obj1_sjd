"""MOCK image golden tests; never reconstruct the missing scientific figures."""
import base64
import csv
import hashlib
import io
import json
import re
from PIL import Image
from streamlit.testing.v1 import AppTest
from shared.loaders.catalog import load_catalog
from shared.loaders.manifest_loader import capture_run
from shared.utils.paths import APP_ROOT


def figure_run(mock_run):
    registry=mock_run.registry
    catalog=load_catalog();registry.catalog=catalog
    variables=['anti_ro_ssa','complement_c4','igg','rheumatoid_factor','wbc']
    image=Image.new('RGB',(1800,3),color='white');buffer=io.BytesIO();image.save(buffer,format='PNG');original=buffer.getvalue()
    registry.figures=[]
    for var in variables:
        row=next(row for row in catalog if row['variable_id']==var)
        fid='MOCK_'+var;view=row['default_view'];path='figures/'+fid+'.png'
        file=registry.outputs/path;file.parent.mkdir(exist_ok=True);file.write_bytes(original)
        registry.add(fid,{'path':path,'kind':'figure','producer':'MOCK.py'})
        registry.figures.append({'figure_id':fid,'variable_id':var,'view':view,'kind':'other','path':path,'title':'MOCK IMAGE — NOT RESULTS','source_script':'MOCK.py','format':'png','panel_groups':'','generated_run':'','status':'verified'})
    return capture_run(registry),original


def test_deep_links_and_original_png_hash(mock_run,monkeypatch):
    run,original=figure_run(mock_run)
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(run.registry.outputs));monkeypatch.setenv('SJD_REPO_ROOT',str(run.registry.repo))
    app=AppTest.from_file(str(APP_ROOT/'app.py'))
    app.session_state['sjd_run']=run
    app.query_params.update(obj='objective_01',var='complement_c4',view='longitudinal')
    app.run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='sjd_variable_Complement').value=='complement_c4'
    images=[trace['source'] for element in app.get('plotly_chart') for trace in json.loads(element.proto.spec)['data'] if trace['type']=='image']
    assert len(images)==1
    encoded=images[0].split(',',1)[1]
    assert hashlib.sha256(base64.b64decode(encoded)).digest()==hashlib.sha256(original).digest()


def test_all_golden_catalogue_domains_and_default_views():
    catalogue={row['variable_id']:row for row in load_catalog()}
    assert catalogue['anti_ro_ssa']['domain']=='Serology'
    assert catalogue['anti_ro_ssa']['default_view']=='categorical_longitudinal'
    for var in ['complement_c4','igg','wbc','rheumatoid_factor']:
        assert catalogue[var]['default_view']=='longitudinal'


def test_category_navigation_and_variable_next(mock_run,monkeypatch):
    run,_=figure_run(mock_run)
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(run.registry.outputs));monkeypatch.setenv('SJD_REPO_ROOT',str(run.registry.repo))
    app=AppTest.from_file(str(APP_ROOT/'app.py'));app.session_state['sjd_run']=run
    app.query_params.update(var='anti_ro_ssa',view='categorical_longitudinal');app.run(timeout=20)
    assert not app.exception
    app.button(key='next_variable').click().run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='sjd_variable_Serology').value=='rheumatoid_factor'
    assert app.query_params['var']=='rheumatoid_factor'


def explorer_app(mock_run,monkeypatch):
    run,_=figure_run(mock_run)
    monkeypatch.setenv('SJD_OUTPUTS_ROOT',str(run.registry.outputs))
    monkeypatch.setenv('SJD_REPO_ROOT',str(run.registry.repo))
    app=AppTest.from_file(str(APP_ROOT/'app.py'))
    app.session_state['sjd_run']=run
    app.query_params.update(var='complement_c4',view='longitudinal')
    return app.run(timeout=20)


def test_search_can_move_away_from_previous_deep_link(mock_run,monkeypatch):
    app=explorer_app(mock_run,monkeypatch)
    app.text_input(key='sjd_search').set_value('IgG').run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='sjd_variable_Immunoglobulins').value=='igg'
    assert app.query_params['var']=='igg'


def test_incoming_deep_link_replaces_existing_widget_selection(mock_run,monkeypatch):
    app=explorer_app(mock_run,monkeypatch)
    app.query_params.update(var='igg',view='longitudinal')
    app.run(timeout=20)
    assert not app.exception
    assert app.selectbox(key='sjd_category').value=='Immunoglobulins'
    assert app.selectbox(key='sjd_variable_Immunoglobulins').value=='igg'


def test_unavailable_deep_view_is_not_substituted(mock_run,monkeypatch):
    app=explorer_app(mock_run,monkeypatch)
    app.query_params['view']='categorical_longitudinal'
    app.run(timeout=20)
    assert not app.exception
    assert not app.get('plotly_chart')
    assert app.query_params['view']=='categorical_longitudinal'


def test_search_no_matches_then_recovers(mock_run,monkeypatch):
    app=explorer_app(mock_run,monkeypatch)
    app.text_input(key='sjd_search').set_value('no such MOCK label').run(timeout=20)
    assert not app.exception
    app.text_input(key='sjd_search').set_value('WBC').run(timeout=20)
    assert not app.exception
    assert app.query_params['var']=='wbc'
