#!/usr/bin/env python3
"""Run Graph02 and Graph03 without required flags."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph._pipeline import base_parser, load_pipeline, output_root, stage_destinations, stage_outputs

SUMMARY = 'graph_study_results_summary.md'


def run(cfg, *, root=None, preflight_only=False):
    from src.studies.graph.stage02.s02_pipeline import run as run02
    from src.studies.graph.stage03.s03_pipeline import run as run03

    root = Path(root).resolve() if root else output_root(cfg, 'study')
    graph02_root, graph03_root = (output_root(cfg, key) for key in ('02', '03'))
    if any(root.is_relative_to(path) or path.is_relative_to(root)
           for stage in (graph02_root, graph03_root) for path in stage_destinations(stage).values()):
        raise ValueError('Study summary directory overlaps a Graph stage')

    def owned(_, name):
        return name.startswith('graph_study_')

    with stage_outputs({'stage': root}, owned, failure=('stage', 'graph_study_failure.json'),
                       completion=root / SUMMARY, diagnostic=preflight_only) as staging:
        if not cfg['execution']['run_02'] or not cfg['execution']['run_03']:
            raise ValueError('Use the standalone entrypoints for individual stages')
        graph02 = run02(cfg, preflight_only=preflight_only)
        graph02_status = graph02['execution_status']
        if preflight_only:
            if graph02_status != 'PREFLIGHT_ONLY':
                raise RuntimeError('GRAPH02_PREFLIGHT_UNAVAILABLE')
            return {'run_dir': root, 'execution_status': graph02_status}
        publishable = {'COMPLETE', 'COMPLETE_WITH_LIMITATIONS', 'PARTIAL'}
        if graph02_status not in publishable or graph02.get('manifest') is None:
            raise RuntimeError('GRAPH02_RESULT_UNAVAILABLE')
        graph03 = run03(cfg, graph02_release=graph02['manifest'])
        graph03_status = graph03['execution_status']
        if graph03_status not in publishable or graph03.get('manifest') is None:
            raise RuntimeError('GRAPH03_RESULT_UNAVAILABLE')
        statuses = (graph02_status, graph03_status)
        if 'PARTIAL' in statuses:
            status = 'PARTIAL'
        elif 'COMPLETE_WITH_LIMITATIONS' in statuses:
            status = 'COMPLETE_WITH_LIMITATIONS'
        else:
            status = 'COMPLETE'
        (staging['stage'] / SUMMARY).write_text(
            '# Graph study\n\n'
            f'Execution status **{status}**.\n\n'
            '| Stage | Status | Current result |\n|---|---|---|\n'
            f'| Graph02 | {graph02_status} | `{graph02["manifest"]}` |\n'
            f'| Graph03 | {graph03_status} | `{graph03["manifest"]}` |\n')
    print(f'Graph study {status}; summary: {root / SUMMARY}')
    return {'run_dir': root, 'execution_status': status}


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    return run(load_pipeline(args.pipeline_config), root=args.output_dir,
               preflight_only=args.preflight_only or args.dry_run)


if __name__ == '__main__':
    main()
