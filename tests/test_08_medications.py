import json
import os
import pandas as pd
import pytest
from medications_synthetic import sources,producer
from src.block_A._internal.medications.contracts import COLUMNS,KEYS


def test_full_pipeline_without_step10_preserves_all_documents(tmp_path):
    args,spine=sources(tmp_path);qc=producer().run(args)
    features=pd.read_parquet(args.output)
    assert len(features)==len(spine) and features[KEYS].equals(spine[KEYS])
    first=features.iloc[0]
    assert first[COLUMNS[0]]=='hydroxychloroquine'
    assert 'rituximab' in first[COLUMNS[2]] and 'rituximab' in first[COLUMNS[3]]
    assert 'unknown synthetic product' in first[COLUMNS[5]]
    assert 'unknown synthetic product' not in first[COLUMNS[4]]
    assert features.iloc[2][COLUMNS].isna().all()
    assert not any(c.startswith(('meddoc__','medprox__','medcand__')) for c in features)
    restricted=args.audit_root/'restricted'
    catalog=pd.read_parquet(restricted/'08_medication_documentation_long.parquet')
    links=pd.read_parquet(restricted/'08_medication_candidate_episode_links.parquet')
    ctdb=catalog.loc[catalog.source_kind.eq('CTDB')]
    assert 'must not assign to a slot' in ctdb.unassigned_visit_doses_json.iloc[0]
    assert ctdb.dose_slot_mapping.eq('unassigned_not_inferred').all()
    assert len(catalog)==15  # all protocol rows, plus original CTDB report
    assert not links.duplicated(KEYS+['documentation_record_id']).any()
    ritux=links.loc[links.medication_name.eq('rituximab')]
    assert ritux.matching_reference_kinds.str.contains('order_end').any()
    assert 'MRN' not in catalog and 'Subject Name' not in catalog
    assert catalog.possible_cross_protocol_reexport.fillna(False).any()
    assert qc['date_parsing']['order_start']['unsupported_format:']==2
    undated=pd.read_parquet(restricted/'08_medication_unassigned.parquet')
    assert undated.medication_name.str.contains('undated synthetic product').any()
    assert qc['confirmed_medication_exposure'] is False
    assert os.stat(restricted).st_mode & 0o777 == 0o700
    assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in restricted.iterdir() if p.is_file() and p.name!='.pseudonym_key')
    assert os.stat(args.output).st_mode & 0o777 == 0o600


def test_dry_run_no_writes_and_missing_key_fails(tmp_path):
    args,_=sources(tmp_path);args.dry_run=True
    assert producer().run(args)['status']=='validated_dry_run_no_publication'
    assert not args.output.parent.exists()
    args.pseudonym_key.unlink()
    with pytest.raises(FileNotFoundError,match='HMAC'):producer().run(args)
    assert not args.pseudonym_key.exists()


def test_reuse_checks_inputs_code_key_and_rebuild_is_deterministic(tmp_path):
    args,_=sources(tmp_path);module=producer();module.run(args)
    original=args.output.read_bytes()
    args.reuse_validated_inputs=True
    assert module.run(args)['status']=='validated_reuse'
    original_key=args.pseudonym_key.read_bytes()
    args.pseudonym_key.write_bytes(b'changed-synthetic-key'+b'x'*32)
    with pytest.raises(ValueError,match='key changed'):module.run(args)
    args.pseudonym_key.write_bytes(original_key)
    args.reuse_validated_inputs=False;args.force_rebuild=True
    module.run(args)
    assert args.output.read_bytes()==original  # analytic names don't expose key-dependent tokens


@pytest.mark.parametrize('failure',['ctdb_patient','source_collision','spine_missing','upstream_overwrite'])
def test_identity_and_source_safety_fail_closed(tmp_path,failure):
    args,spine=sources(tmp_path)
    if failure=='ctdb_patient':
        path=args.eda_root/'data_intermediate/clinical_episode_row_map.parquet'
        frame=pd.read_parquet(path);frame.loc[0,'patient_id']='SYNTHETIC_B';frame.to_parquet(path,index=False)
    elif failure=='source_collision':
        path=args.cohort;frame=pd.read_csv(path);frame.loc[len(frame)]={'patient_id':'SYNTHETIC_A'};frame.to_csv(path,index=False)
    elif failure=='spine_missing':
        spine.iloc[:-1].to_parquet(args.spine,index=False)
    else:
        args.output=args.spine
    with pytest.raises(ValueError):producer().run(args)
    assert not (args.audit_root/'08_medication_publication_qc.json').exists()


def test_unmapped_ingredient_and_free_text_sanitization():
    from src.block_A._internal.medications.episode_features import review_name
    assert review_name('unmapped','New synthetic compound','unmapped')=='new synthetic compound'
    assert review_name('','Please contact research nurse@example.invalid','unmapped')==''


def test_archived_orchestrator_is_disabled(tmp_path):
    import subprocess,sys
    from medications_synthetic import ROOT
    script=ROOT/'src/block_A/audit/08_medications/legacy/08_rebuild_full_source_pipeline.py'
    result=subprocess.run([sys.executable,str(script),'--audit-root',str(tmp_path)],text=True,capture_output=True)
    assert result.returncode!=0 and 'Archived orchestrator' in result.stderr
    assert not list(tmp_path.iterdir())
