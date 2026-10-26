import json
import pandas as pd
import pytest
from medications_synthetic import sources,producer
from src.block_A._internal.medications.contracts import validate,COLUMNS,KEYS
from src.block_A._internal.medications.persistence import publish,manifest_path


def test_empty_categories_are_missing_and_names_are_deterministic(tmp_path):
    args,spine=sources(tmp_path);producer().run(args)
    features=pd.read_parquet(args.output)
    assert features.iloc[1][COLUMNS].isna().all()
    for col in COLUMNS:
        for cell in features[col].dropna():assert cell.split('; ')==sorted(set(cell.split('; ')))
    for bad in [features.iloc[:-1],pd.concat([features,features.iloc[:1]])]:
        with pytest.raises(ValueError):validate(bad,spine)
    features.loc[0,COLUMNS[0]]='z; a'
    with pytest.raises(ValueError,match='noncanonical'):validate(features,spine)


def test_publication_rollback_keeps_old_complete_pair(tmp_path,monkeypatch):
    from src.block_A._internal.medications import persistence
    args,_=sources(tmp_path);module=producer();module.run(args)
    original=(args.output.read_bytes(),manifest_path(args.output).read_bytes())
    replace=persistence.os.replace
    def fail(source,destination):
        if destination==manifest_path(args.output):raise OSError('synthetic rename failure')
        return replace(source,destination)
    monkeypatch.setattr(persistence.os,'replace',fail)
    args.force_rebuild=True
    with pytest.raises(OSError,match='rename'):module.run(args)
    assert (args.output.read_bytes(),manifest_path(args.output).read_bytes())==original


def test_spine_grain_is_data_driven_including_empty_episodes():
    from src.block_A._internal.medications.episode_features import build
    from src.block_A._internal.medications.episode_linkage import LINK_COLUMNS
    # This is an invented size-regression fixture, never evidence of real-cohort validation.
    spine=pd.DataFrame({'patient_id':[f'SYNTHETIC_{i%159}' for i in range(497)],
                        'clinical_episode_id':[f'SYNTHETIC_E_{i}' for i in range(497)]})
    features=build(spine,pd.DataFrame(columns=LINK_COLUMNS))
    assert len(features)==497 and features.patient_id.nunique()==159
    assert features[COLUMNS].isna().all().all()
