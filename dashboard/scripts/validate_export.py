"""Validate real researcher-supplied aggregate fixtures, never fabricate them."""
import argparse
from pathlib import Path
import sys
import re
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.loaders.registry import Registry
from shared.loaders.csv_loader import read_csv_bytes, check_header
from shared.loaders.manifest_loader import Run
from shared.loaders.refs import resolve
from shared.utils.paths import APP_ROOT


def configured_refs(node):
    if isinstance(node,dict):
        for value in node.values(): yield from configured_refs(value)
    elif isinstance(node,list):
        for value in node: yield from configured_refs(value)
    elif isinstance(node,str):
        if node.startswith(('t11_','t12_')) and '@' in node and '#' in node:
            yield node.replace('{cohort}','Overall')
        for ref in re.findall(r'\{\{ref:(.*?)\}\}',node):
            yield ref.replace('{cohort}','Overall')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fixtures',type=Path,default=APP_ROOT/'tests/fixtures/objective_01'); args=parser.parse_args()
    registry=Registry(); run=Run('researcher-export-validation',registry,{})
    required=['t11_overall','t11_by_pop','t11_avail','t11_dict','t12_long','t12_ret']
    missing=[Path(registry.get(rid).path).name for rid in required if not (args.fixtures/Path(registry.get(rid).path).name).is_file()]
    if missing:
        print('BLOCKED: researcher-provided aggregate fixtures missing: '+', '.join(missing)); return 2
    for rid in required:
        entry=registry.get(rid)
        path=args.fixtures/Path(entry.path).name
        try:
            check_header(path)
            run.tables[rid]=read_csv_bytes(path.read_bytes(),entry)
        except (ValueError,OSError) as exc:
            print('INVALID aggregate fixture '+path.name+': '+str(exc)); return 1
    cfg=yaml.safe_load((APP_ROOT/'objectives/objective_01/config.yml').read_text())
    failures=[]
    for ref in configured_refs(cfg):
        try:
            value=resolve(ref,run)
            if value.kind=='failure': failures.append(ref+': parse failure')
        except ValueError as exc: failures.append(ref+': '+str(exc))
    # Historical expected values are test-only; never runtime data.
    expected={'t11_overall:Cohort / demographics|N patients@Summary#raw':'159','t11_overall:Disease activity|essdai_total@Summary#raw':'2.0 (0.0–4.0); n=148','t11_overall:Serology|Anti-Ro/SSA positive, n/N (%)@Summary#raw':'49/62 (79.0%)','t12_long:Clinical episodes|Overall@Value#raw':'497'}
    for ref,raw in expected.items():
        try:
            if resolve(ref,run).raw!=raw: failures.append(ref+': historical export differs')
        except ValueError as exc:
            failures.append(ref+': '+str(exc))
    if failures: print('\n'.join(failures)); return 1
    print('Configured references and historical expected cells verified against real aggregate fixtures.'); return 0


if __name__=='__main__': sys.exit(main())
