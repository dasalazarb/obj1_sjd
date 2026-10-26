"""Regression: the same documented names appear in CSV, master, context and registry."""
import importlib.util
from pathlib import Path
import sys
import pandas as pd
import pytest
from medications_synthetic import sources,producer,ROOT
from src.block_A._internal.medications.contracts import COLUMNS,KEYS

spec=importlib.util.spec_from_file_location('step10_master_medications',ROOT/'src/block_A/10_build_integrated_longitudinal_dataset.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_medication_descriptors_present_in_both_outputs(tmp_path):
    args,spine=sources(tmp_path);producer().run(args)
    base=m.CuratedBuild(spine.copy(),spine.copy(),pd.DataFrame(),pd.DataFrame(),[],[],[],pd.DataFrame())
    built,qc=m._integrate_medication_context(base,medication_input=args.output,clinical_spine_path=args.spine)
    assert len(built.analytic)==len(built.context)==len(spine)
    pd.testing.assert_frame_equal(built.analytic[KEYS+COLUMNS],built.context[KEYS+COLUMNS])
    pd.testing.assert_frame_equal(built.analytic[spine.columns],spine)
    assert set(COLUMNS)==set(built.registry.variable)
    assert built.registry.included_in_analytic.all() and built.registry.included_in_context.all()
    assert not any(c.startswith(('meddoc__','medprox__','medcand__')) for c in built.analytic)
    assert qc['pre_anchor_treatment_exposure_established'] is False
    out=tmp_path/'synthetic_step10.csv';built.analytic.to_csv(out,index=False)
    restored=pd.read_csv(out,dtype='string')
    assert restored.loc[0,COLUMNS[0]]=='hydroxychloroquine'
    assert restored.loc[2,COLUMNS].isna().all()


def test_medication_descriptors_are_checked_by_episode_key_when_orders_differ(tmp_path):
    args,spine=sources(tmp_path);producer().run(args)
    context=spine.iloc[::-1].copy()
    context['patient_id']=context.patient_id.astype(object)
    base=m.CuratedBuild(spine.copy(),context,pd.DataFrame(),pd.DataFrame(),[],[],[],pd.DataFrame())
    result,qc=m._integrate_medication_context(base,medication_input=args.output,clinical_spine_path=args.spine)
    pd.testing.assert_frame_equal(result.analytic.set_index(KEYS)[COLUMNS].sort_index(),result.context.astype({'patient_id':'string'}).set_index(KEYS)[COLUMNS].sort_index())
    assert qc['n_descriptive_medication_columns_added_to_master']==len(COLUMNS)


def test_both_clis_publish_matching_names_and_explicit_omission(tmp_path,monkeypatch):
    """Exercise real writers/registry and the default medication requirement."""
    from test_integrated_longitudinal_dataset import frames
    args,spine=sources(tmp_path)
    upstream=list(frames())
    for col in ['patient_id','clinical_episode_id','clinical_anchor_date','episode_start_date','episode_end_date','clinical_visit']:
        upstream[0][col]=spine[col].to_numpy()
    upstream[0]['clinical_baseline_episode_id']=['synthetic_e1','synthetic_e1','synthetic_e3']
    upstream[0]['clinical_baseline_date']=pd.to_datetime(['2020-02-29']*3)
    upstream[0]['time_since_clinical_baseline_days']=[0,1127,0]
    upstream[0]['time_since_clinical_baseline_years']=[0.,1127/365.25,0.]
    for block in upstream[1:]:
        for col in ['patient_id','clinical_episode_id','clinical_anchor_date','clinical_visit_number']:
            block[col]=upstream[0][col].to_numpy()
    paths=[]
    for label,block in zip(['spine','pop','labs','overlap','pros','extended-clinical'],upstream):
        path=tmp_path/f'{label}.parquet';block.to_parquet(path,index=False);paths.append(path)
    args.spine=paths[0];args.all_spine=paths[0]
    # Full producer CLI parsing, no Step10 products exist yet.
    producer().main(['--eda-root',str(args.eda_root),'--audit-root',str(args.audit_root),
                     '--spine',str(args.spine),'--all-spine',str(args.all_spine),
                     '--cohort',str(args.cohort),'--pseudonym-key',str(args.pseudonym_key),
                     '--output',str(args.output)])
    argv=['step10']
    for label,path in zip(['spine','pop','labs','overlap','pros','extended-clinical'],paths):argv += ['--'+label,str(path)]
    out=tmp_path/'step10/clinical_episode.parquet';qc_dir=tmp_path/'step10/qc'
    argv += ['--medication-input',str(args.output),'--output',str(out),'--qc-dir',str(qc_dir)]
    monkeypatch.setattr(m.common,'ensure_output_dirs',lambda:None)
    monkeypatch.setattr(sys,'argv',argv);m.main()
    analytic=pd.read_parquet(out)
    context=pd.read_parquet(out.with_name('10_integrated_longitudinal_context.parquet'))
    registry=pd.read_csv(out.with_name('10_variable_registry.csv'))
    assert set(COLUMNS).issubset(analytic)
    pd.testing.assert_frame_equal(analytic[KEYS+COLUMNS],context[KEYS+COLUMNS])
    assert set(COLUMNS).issubset(registry.variable)
    assert out.with_suffix('.csv').exists()
    # The explicit clinical-only option is visible in the actual written QC.
    import json
    # Keep every clinical block argument; remove only the medication/output/QC options.
    clinical_only=tmp_path/'clinical_only/clinical_episode.parquet'
    clinical_only_qc=tmp_path/'clinical_only/qc'
    no_med=argv[:argv.index('--medication-input')]+['--no-medications','--output',str(clinical_only),'--qc-dir',str(clinical_only_qc)]
    monkeypatch.setattr(sys,'argv',no_med);m.main()
    clinical_only_data=pd.read_parquet(clinical_only)
    assert not set(COLUMNS).intersection(clinical_only_data)
    pd.testing.assert_frame_equal(analytic.drop(columns=COLUMNS),clinical_only_data)
    assert json.loads((clinical_only_qc/'10_medication_documentation_qc.json').read_text())['omitted_explicitly'] is True
