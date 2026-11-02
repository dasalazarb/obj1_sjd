"""Generate explicitly synthetic, aggregate-only evidence from the public fixture."""
from pathlib import Path
import json
import hashlib
import shutil
import time
from tests.test_graph_missingness_variants import synthetic
from src.studies.graph._pipeline import load_pipeline
from src.studies.graph.stage02.s02_missingness_study import run_missingness_study
from src.studies.graph.stage02.s02_representations import build_space
from src.studies.graph.stage02.s02_mapper_engine import analyze
raw,registry,baseline,config,policy,families=synthetic.__wrapped__()
policy['resampling'].update(main_subsampling_replicates_per_variant=2,iterative_bootstrap_replicates=2,parameter_sensitivity_subsampling_replicates=2)
cfg=load_pipeline();cfg['missingness_methods']=policy;cfg['execution']['resume']=False;cfg['experiments']['minimum_success_fraction']=.5
root=Path('work/focal_synthetic');root.mkdir(parents=True,exist_ok=True)
results={}
for reference in ['R0','R2']:
    space=build_space(reference,raw,families,config,registry,registry,baseline,{}, {})
    results[reference]=dict(space=space,mapped=analyze(space,config))
start=time.perf_counter()
report=run_missingness_study(results,registry,registry,baseline,families,config,cfg,root,{}, {},'public_synthetic_fixture_927')
import pandas as pd
out=Path('docs/graph02_focal_fix_synthetic');out.mkdir(exist_ok=True)
frame=pd.read_csv(root/'02_graph_missingness_method_registry.csv')
columns=['experiment_id','status','reason','n_patients','n_encoded_features','n_shared','n_covered','n_communities','n_valid_pair_distances','n_connected','n_successful_refits','estimator','panel_hash']
frame.loc[~frame.experiment_id.str.contains('/')].reindex(columns=columns).to_csv(out/'acceptance.csv',index=False)
for name in ['imputation_diagnostics','sparse_graph_connectivity','focal_comparison']:
    for extension in ['png','pdf','csv']:
        shutil.copyfile(root/'figures'/('02_graph_missingness_'+name+'.'+extension),out/(name+'.'+extension))
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
(out/'manifest.json').write_text(json.dumps(dict(data='PUBLIC_SYNTHETIC_ONLY; no clinical data',seed=config['random_seed'],
    fixture='tests/test_graph_missingness_variants.py::synthetic',fixture_seed=927,n_patients=len(raw),n_features=raw.shape[1],
    estimator_version=policy['estimator_version'],cache_version=policy['cache_version'],graph03_defining=False,
    main_refits=2,parameter_refits=2,bootstrap_refits=2,elapsed_seconds=time.perf_counter()-start,
    code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src/studies/graph/stage02').glob('s02_*.py'))},
    config_sha256=cfg['_config_sha256'],fixture_sha256=hashlib.sha256(Path('tests/test_graph_missingness_variants.py').read_bytes()).hexdigest(),
    sha256=files),indent=2)+'\n')
print(frame.loc[~frame.experiment_id.str.contains('/')].reindex(columns=columns).to_string(index=False))
