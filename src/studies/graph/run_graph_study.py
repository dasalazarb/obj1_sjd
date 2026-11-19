#!/usr/bin/env python3
"""Run Graph02, Graph03 and exploratory Graph03 N3 without required flags."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.graph._pipeline import base_parser, load_pipeline, output_root, stage_outputs

SUMMARY = 'graph_study_results_summary.md'


def run(cfg, *, root=None, preflight_only=False):
    from src.studies.graph.stage02.s02_pipeline import run as run02
    from src.studies.graph.stage03.s03_n3_exploratory import run_exploratory_auto
    from src.studies.graph.stage03.s03_pipeline import run as run03

    root = Path(root).resolve() if root else output_root(cfg, 'study')
    graph02_root, graph03_root = (output_root(cfg, key) for key in ('02', '03'))
    n3_root = graph03_root.parent / '03_graph_n3_controlled_inference'
    if any(root.is_relative_to(stage) or stage.is_relative_to(root)
           for stage in (graph02_root, graph03_root, n3_root)):
        raise ValueError('Study summary directory overlaps a Graph stage')

    def owned(_, name):
        return name.startswith('graph_study_')

    with stage_outputs({'stage': root}, owned, failure=('stage', 'graph_study_failure.json'),
                       completion=root / SUMMARY, diagnostic=preflight_only) as staging:
        if not cfg['execution']['run_02'] or not cfg['execution']['run_03']:
            raise ValueError('Use the standalone entrypoints for individual stages')
        graph02 = run02(cfg, preflight_only=preflight_only)
        if preflight_only:
            return {'run_dir': root, 'execution_status': 'PREFLIGHT_ONLY'}
        graph03 = run03(cfg)
        n3 = run_exploratory_auto(ROOT, cfg)
        if n3['status'] in {'EXPLORATORY_RUN_FAILED', 'INPUTS_UNUSABLE'}:
            raise RuntimeError('GRAPH03_N3_INPUTS_UNAVAILABLE')
        statuses = (graph02['execution_status'], graph03['execution_status'], n3['status'])
        if 'PARTIAL' in statuses or 'PARTIAL_WITH_DATA_ISSUES' in statuses:
            status = 'PARTIAL'
        elif 'COMPLETE_WITH_LIMITATIONS' in statuses:
            status = 'COMPLETE_WITH_LIMITATIONS'
        else:
            status = 'COMPLETE'
        (staging['stage'] / SUMMARY).write_text(
            '# Graph study\n\n'
            f'Execution status **{status}**.\n\n'
            '| Stage | Status | Current result |\n|---|---|---|\n'
            f'| Graph02 | {statuses[0]} | `{graph02["manifest"]}` |\n'
            f'| Graph03 | {statuses[1]} | `{graph03["manifest"]}` |\n'
            f'| Graph03 N3 | {statuses[2]} | `{n3["report"]}` |\n\n'
            f'Exploratory primary model pairs estimated in 03e **{n3["n_model_pairs_estimable"]}**.\n')
    print(f'Graph study {status}; summary: {root / SUMMARY}')
    return {'run_dir': root, 'execution_status': status}


def main(argv=None):
    args = base_parser(__doc__).parse_args(argv)
    return run(load_pipeline(args.pipeline_config), root=args.output_dir,
               preflight_only=args.preflight_only or args.dry_run)


if __name__ == '__main__':
    main()
