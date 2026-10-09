"""Software-only MOCK inputs, never real clinical results or UI fallback data."""
import csv
import io
import pytest
import yaml
from shared.loaders.registry import Registry
from shared.loaders.manifest_loader import capture_run
from shared.utils.paths import APP_ROOT
from scripts.validate_export import configured_refs


@pytest.fixture
def mock_run(tmp_path):
    cfg=yaml.safe_load((APP_ROOT/'objectives/objective_01/config.yml').read_text())
    registry=Registry(outputs=tmp_path/'MOCK_outputs',repo=tmp_path/'MOCK_repo')
    cells={}
    refs=list(configured_refs(cfg))
    # Explicitly include cells addressed dynamically by population and retention views.
    for row in cfg['pro_rows'] + cfg['lab_rows'] + cfg['overlap_segments']:
        refs.extend(f't11_by_pop:{row["row_key"]}@{group}#raw' for group in cfg['pop_columns'])
    refs.extend(f't12_ret:Overall|{time}@{column}#raw' for time in cfg['retention_times'] for column in ['pct_retained','n_retained','denominator'])
    refs.extend([
        't11_overall:Glandular / extended phenotype|Any sicca symptom present, n/N (%)@Summary#raw',
        't12_long:Patients with exactly 1 clinical episode|Overall@Value#k',
    ])
    for ref in refs:
        rid,key,tail=ref.split(':',1)[0],ref.split(':',1)[1].split('@',1)[0],ref.split('@',1)[1]
        col,field=tail.rsplit('#',1)
        row=cells.setdefault(rid,{}).setdefault(key,{})
        if rid in {'t11_overall','t11_by_pop'}:
            raw='3' if key.endswith('|N patients') else '1/3 (33.3%)' if ('n/N (%)' in key or 'race:' in key or 'ethnicity:' in key or 'sex:' in key or 'overlap_status:' in key) else '1.0 (0.0–2.0); n=3'
            if col in {'N available','N missing'}: raw='3' if col=='N available' else '0'
        elif rid=='t11_avail': raw='33.3' if col.startswith('pct_') else '3'
        else:
            raw='1 (33.3%)' if field in {'k','pct'} else '1.0 (0.0–2.0)' if field in {'median','q1','q3'} else '1.0–2.0' if key.startswith('IQR ') else '3'
        row[col]=raw
    for rid in ['t11_overall','t11_by_pop','t11_avail','t12_long','t12_ret']:
        entry=registry.get(rid); rows=[]
        for key,provided in cells.get(rid,{}).items():
            row={col:'MOCK' for col in entry.spec['columns']}
            row.update(dict(zip(entry.spec['key'],key.split('|'))))
            row.update(provided)
            if rid=='t11_overall': row['N available']='3'; row['N missing']='0'
            if rid=='t11_by_pop':
                sample=next(iter(provided.values()))
                for col in ['Overall','Pop1','Pop2','Pop3','Unclassifiable']: row.setdefault(col,sample); row[col]=provided.get(col,sample)
            rows.append(row)
        path=registry.path(rid); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=entry.spec['columns']); writer.writeheader(); writer.writerows(rows)
    entry=registry.get('t11_dict'); path=registry.path('t11_dict'); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=entry.spec['columns']);writer.writeheader();writer.writerow({**{col:'MOCK' for col in entry.spec['columns']},'variable':'MOCK_variable','is_canonical':'True','include_in_table1':'True'})
    registry.catalog=[]; registry.figures=[]
    return capture_run(registry)
