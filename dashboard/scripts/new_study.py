"""Create empty study scaffolding. Never generates scientific content."""
import argparse
from pathlib import Path
import re
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.utils.paths import APP_ROOT


def scaffold(root,kind,identifier,title):
    if not re.fullmatch(r'[a-z][a-z0-9_]*',identifier): raise ValueError('Invalid study id')
    destination=Path(root)/('objectives' if kind=='objective' else 'studies')/identifier
    if destination.exists(): raise ValueError('Study already exists; existing content preserved')
    (destination/'sections').mkdir(parents=True); (destination/'metadata').mkdir()
    (destination/'__init__.py').touch()
    (destination/'README.md').write_text('# '+title+'\n\nPlanned. Register verified aggregate outputs and editorial sections before enabling.\n')
    (destination/'page.py').write_text('import streamlit as st\n\ndef render():\n    st.title('+repr(title)+')\n    st.info("Coming soon")\n')
    (destination/'narrative.md').write_text('')
    (destination/'config.yml').write_text(yaml.safe_dump({'id':identifier,'title':title,'status':'planned','sections':[]}))
    (destination/'metadata/claims.yml').write_text('[]\n')
    (destination/'metadata/figures.csv').write_text('figure_id,variable_id,view,kind,path,title,source_script,format,panel_groups,generated_run,status\n')
    (destination/'metadata/tables.csv').write_text('registry_id,status\n')
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=['objective','study'],required=True); parser.add_argument('--id',required=True); parser.add_argument('--title',required=True); args=parser.parse_args()
    print(scaffold(APP_ROOT,args.kind,args.id,args.title))
