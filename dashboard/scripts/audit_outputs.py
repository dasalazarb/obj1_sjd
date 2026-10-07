"""Inventory existing outputs without reading patient data or altering upstream files."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.loaders.registry import Registry
from shared.loaders.catalog import load_catalog
from shared.utils.paths import APP_ROOT,contained

FIELDS=['figure_id','variable_id','view','kind','path','title','source_script','format','panel_groups','generated_run','status']


def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def audit(root,destination,catalog):
    root=Path(root).resolve(); destination=Path(destination).resolve()
    if destination.is_relative_to(root): raise ValueError('Audit destination must be outside upstream outputs')
    inventory=[]; candidates=[]; unmatched=[]
    allowed={row['variable_id']:row for row in catalog}
    figure_root=root/'figures'
    if figure_root.exists():
        for path in sorted(figure_root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in {'.png','.pdf','.svg'}: continue
            relative=path.relative_to(root).as_posix()
            contained(root,relative) # reject escaping symlinks
            stat=path.stat()
            item={'path':relative,'extension':path.suffix.lower(),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'producer_directory':path.parent.name}
            inventory.append(item)
            # Explicit metadata only. No OCR, image inspection or ambiguous filename inference.
            sidecar=path.with_suffix('.meta.json')
            if sidecar.is_file():
                contained(root,sidecar.relative_to(root))
                meta=json.loads(sidecar.read_text())
                variable=meta.get('variable_id'); view=meta.get('view')
                if variable in allowed and view in {'longitudinal','categorical_longitudinal'}:
                    candidates.append(dict(figure_id=meta.get('figure_id',path.stem),variable_id=variable,view=view,kind=meta.get('kind','other'),path=relative,title=allowed[variable]['display_name'],source_script=meta.get('source_script',''),format=path.suffix.lower()[1:],panel_groups='',generated_run=meta.get('run_id',''),status='candidate'))
                    continue
            unmatched.append({**item,'reason':'No unambiguous supported sidecar mapping; human curation required'})
    write_csv(destination/'figures_inventory.csv',['path','extension','size','mtime_ns','producer_directory'],inventory)
    write_csv(destination/'figures_unmatched.csv',['path','extension','size','mtime_ns','producer_directory','reason'],unmatched)
    write_csv(destination/'figures_candidates.csv',FIELDS,candidates)
    return candidates


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--study',default='objective_01'); parser.add_argument('--outputs-root',type=Path); parser.add_argument('--audit-dir',type=Path,default=APP_ROOT/'audit'); args=parser.parse_args()
    registry=Registry(outputs=args.outputs_root)
    audit(registry.outputs,args.audit_dir,load_catalog())
    rows=[]
    for rid,entry in registry.entries.items():
        path=registry.path(rid)
        rows.append({'registry_id':rid,'path':entry.path,'producer':entry.producer,'status':'present' if path.is_file() else 'missing'})
    write_csv(args.audit_dir/'registered_outputs.csv',['registry_id','path','producer','status'],rows)
    print('Audit written to '+str(args.audit_dir)+'. Candidates require human review; figures.csv was not overwritten.')


if __name__=='__main__': main()
