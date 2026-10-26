import json
import pandas as pd
import pytest
from medications_synthetic import sources,producer
from src.block_A._internal.step10_medications import load_contract
from src.block_A._internal.medications.persistence import manifest_path,sha256
from src.block_A._internal.medications.contracts import COLUMNS


def test_reader_only_opens_contract_manifest_lock_and_configured_spine(tmp_path,monkeypatch):
    args,spine=sources(tmp_path);producer().run(args)
    real=pd.read_parquet
    def read(path,*a,**kw):
        assert path==args.output
        return real(path,*a,**kw)
    monkeypatch.setattr(pd,'read_parquet',read)
    names,qc=load_contract(args.output,spine,args.spine)
    assert names[COLUMNS].notna().any().any() and qc['confirmed_medication_exposure'] is False

@pytest.mark.parametrize('kind',['hash','version','missing_keys','extra_keys','duplicate_keys','schema','spine','policy','manifest_counts'])
def test_invalid_publication_fails_closed(tmp_path,kind):
    args,spine=sources(tmp_path);producer().run(args)
    meta_path=manifest_path(args.output);meta=json.loads(meta_path.read_text())
    if kind=='hash':args.output.write_bytes(args.output.read_bytes()+b'corrupt')
    elif kind=='version':meta['contract_version']='old-version'
    elif kind=='spine':spine.loc[0,'clinical_anchor_date']=pd.Timestamp('2020-03-01')
    elif kind=='policy':meta['consent_is_not_medication_date']=False
    elif kind=='manifest_counts':meta['n_rows']+=1
    else:
        features=pd.read_parquet(args.output)
        if kind=='missing_keys':features=features.iloc[:-1]
        elif kind=='extra_keys':features.loc[0,'clinical_episode_id']='extra_episode'
        elif kind=='duplicate_keys':features=pd.concat([features,features.iloc[:1]])
        else:features=features.rename(columns={COLUMNS[0]:'meddoc__old_count'})
        features.to_parquet(args.output,index=False);meta['sha256']=sha256(args.output)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):load_contract(args.output,spine,args.spine)


def test_missing_contract_requires_explicit_clinical_only(tmp_path):
    args,spine=sources(tmp_path)
    with pytest.raises(FileNotFoundError,match='--no-medications'):load_contract(args.output,spine,args.spine)


def test_medications_cannot_be_pre_anchor_model_predictors():
    from src.studies._shared import validate_predictors
    for column in COLUMNS:
        for prefix in ('','from_'):
            with pytest.raises(ValueError,match='descriptive'):validate_predictors([prefix+column])
