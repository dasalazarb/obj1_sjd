#!/usr/bin/env python3
"""Inventory Graph03 source schemas without reading clinical rows or parquet metadata.

Only Arrow field names/types and an allowlist of variable-registry metadata are
serialized. In particular, parquet statistics, row counts, pandas metadata,
registry coverage counts, patient values and dates are never inspected.
"""
import argparse
import csv
import json
from pathlib import Path

REGISTRY_FIELDS = ('variable', 'source', 'family', 'role', 'destination',
                   'canonical_variable', 'alias_of', 'dtype', 'producer_script',
                   'original_variable', 'public_variable', 'temporal_scope')


def schema_inventory(base, graph02_run, master=None):
    import pyarrow.parquet as pq
    base, graph02_run = Path(base), Path(graph02_run)
    master = Path(master) if master else base / (
        'data/analytic/10_build_integrated_longitudinal_dataset/'
        '10_integrated_longitudinal_clinical_episode.parquet')
    sources = {
        'Step10_analytic_master': master,
        'Step10_context': master.with_name('10_integrated_longitudinal_context.parquet'),
        'Step10_variable_registry': master.with_name('10_variable_registry.csv'),
        'Graph02_baseline': graph02_run / '02_graph_baseline_raw.csv',
    }
    result = dict(schema_version='graph03_schema_inventory_v1',
                  patient_rows_read=False, patient_counts_exported=False, sources={})
    for label, path in sources.items():
        entry = dict(file_exists=path.is_file())
        if path.is_file():
            if path.suffix == '.parquet':
                schema = pq.read_schema(path)
                entry['fields'] = [dict(name=f.name, type=str(f.type), nullable=f.nullable)
                                   for f in schema]
            else:
                with path.open(newline='', encoding='utf-8-sig') as handle:
                    reader = csv.DictReader(handle)
                    entry['fields'] = list(reader.fieldnames or [])
                    if label == 'Step10_variable_registry':
                        entry['variable_metadata'] = [
                            {key: row[key] for key in REGISTRY_FIELDS if key in row}
                            for row in reader]
        result['sources'][label] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--graph02-run', required=True, type=Path)
    parser.add_argument('--master', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = schema_inventory(args.base, args.graph02_run, args.master)
    # An inventory must not replace a source or an earlier inventory.
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')


if __name__ == '__main__':
    main()
