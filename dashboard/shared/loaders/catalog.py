import csv
from shared.utils.paths import APP_ROOT
from shared.loaders.registry import SchemaError


def load_catalog():
    with (APP_ROOT/'config/variable_catalog.csv').open(newline='') as handle:
        rows=list(csv.DictReader(handle))
    ids=[row['variable_id'] for row in rows]
    if len(ids)!=len(set(ids)):
        raise SchemaError('Duplicate variable catalogue id')
    return sorted([row for row in rows if row['active']=='true'],key=lambda row:int(row['sort_order']))


def load_figures():
    path=APP_ROOT/'objectives/objective_01/metadata/figures.csv'
    with path.open(newline='') as handle:
        reader=csv.DictReader(handle)
        required={'figure_id','variable_id','view','kind','path','title','source_script','format','panel_groups','generated_run','status'}
        if not required.issubset(reader.fieldnames or []):
            raise SchemaError('Incomplete figure manifest columns')
        rows=list(reader)
    ids=[row['figure_id'] for row in rows]
    if len(ids)!=len(set(ids)):
        raise SchemaError('Duplicate figure id')
    verified=[(row['variable_id'],row['view']) for row in rows if row['status']=='verified']
    if len(verified)!=len(set(verified)):
        raise SchemaError('Ambiguous verified variable/view mapping')
    for row in rows:
        if row['status'] not in {'verified','candidate','missing'}:
            raise SchemaError('Invalid figure review status')
        if row['format'] not in {'png','svg','pdf'}:
            raise SchemaError('Unsupported figure format')
    return rows


def attach_figures(registry):
    registry.catalog=load_catalog()
    registry.figures=load_figures()
    for row in registry.figures:
        if row['status']=='verified':
            registry.add(row['figure_id'],{'path':row['path'],'kind':'figure','producer':row['source_script'],'format':row['format']})
    return registry
