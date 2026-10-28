#!/usr/bin/env python3
"""Run the complete prespecified Graph02 -> Graph03 study without required flags."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from src.studies.graph._pipeline import (base_parser,load_pipeline,output_root,new_run,write_json,
    save_table,digest,utc_now,version_info,git_commit)


def run(cfg, *, root=None,run_id=None,preflight_only=False):
    from src.studies.graph.stage02.s02_pipeline import run as run02
    from src.studies.graph.stage03.s03_pipeline import run as run03
    run_dir=new_run(root or output_root(cfg,'study'),run_id)
    steps={}; status='FAILED'
    try:
        if not cfg['execution']['run_02'] or not cfg['execution']['run_03']:
            raise ValueError('Integrated runner requires both stages; use the standalone entrypoints for individual stages')
        steps['02']=run02(cfg,preflight_only=preflight_only)
        if preflight_only:
            status='PREFLIGHT_ONLY'
            return {'run_dir':run_dir,'execution_status':status}
        # Pass the exact release created by this run. Never rediscover it via glob/pointer.
        steps['03']=run03(cfg,graph02_release=steps['02']['manifest'])
        values=[r['execution_status'] for r in steps.values()]
        status='PARTIAL' if 'PARTIAL' in values else 'COMPLETE_WITH_LIMITATIONS' if 'COMPLETE_WITH_LIMITATIONS' in values else 'COMPLETE'
        index=[]; experiment_status=[]; qc=[]
        for stage,result in steps.items():
            for artifact in sorted(result['run_dir'].rglob('*')):
                if artifact.is_file():
                    index.append({'stage':stage,'relative_path':os.path.relpath(artifact,run_dir),
                                  'sha256':digest(artifact),'artifact_type':artifact.suffix.lstrip('.'),'stage_status':result['execution_status']})
            name='02_graph_experiment_registry.csv' if stage=='02' else '03_graph_validation_status.csv'
            frame=pd.read_csv(result['run_dir']/name); frame['stage']=stage; experiment_status.append(frame)
            qc.append({'stage':stage,'execution_status':result['execution_status'],'manifest_sha256':digest(result['manifest'])})
        save_table(run_dir/'graph_study_output_index.csv',index)
        save_table(run_dir/'graph_study_experiment_status.csv',pd.concat(experiment_status,ignore_index=True))
        save_table(run_dir/'graph_study_qc_summary.csv',qc)
        (run_dir/'graph_study_results_summary.md').write_text(f'# Graph study\n\nEXPLORATORY: clinical validity is not established.\n\nExecution: {status}. Graph02 freezes topology before Graph03 interprets it.\n\nGraph02 release: {os.path.relpath(steps["02"]["manifest"],run_dir)}\n\nGraph03 summary: {os.path.relpath(steps["03"]["manifest"],run_dir)}\n\nSee output index and experiment status for all data, PNG/PDF figures, negative results, and explicit not-estimable reasons.\n')
        print(f'Graph study {status}; index: {run_dir / "graph_study_output_index.csv"}')
        return {'run_dir':run_dir,'execution_status':status}
    finally:
        write_json(run_dir/'graph_study_execution_manifest.json',{'schema_version':'graph_study_v1',
            'execution_status':status,'created_at_utc':utc_now(),'git_commit':git_commit(),'config_sha256':cfg['_config_sha256'],
            'versions':version_info(),'stages':{s:{'manifest_path':os.path.relpath(r['manifest'],run_dir) if r.get('manifest') else None,
                'manifest_sha256':digest(r['manifest']) if r.get('manifest') else None,'execution_status':r['execution_status']} for s,r in steps.items()}})


def main(argv=None):
    parser=base_parser(__doc__); args=parser.parse_args(argv)
    return run(load_pipeline(args.pipeline_config),root=args.output_dir,run_id=args.run_id,preflight_only=args.preflight_only or args.dry_run)

if __name__=='__main__': main()
