"""Collector admits only canonical, checked and aggregate files."""
import csv
from hashlib import sha256
from io import StringIO
import json
from zipfile import ZipFile

import pytest

from src.studies.pharma.stage05 import deck_collection as deck


def put(root, path, data):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(data)
    return file


def test_collector_paths_statuses_hashes_and_atomic_replacement(tmp_path, monkeypatch):
    data = b'marker,n_patients\nvalid,2\n'
    table = f'outputs/tables/{deck.STAGE05}/05_pharma_trial_eligibility_proxy.csv'
    missing_pdf = f'outputs/figures/{deck.STAGE05}/05_pharma_absolute_risk_by_biomarker.pdf'
    stale = f'outputs/tables/{deck.STAGE05}/05_pharma_essdai_decomposition.csv'
    multiple = f'outputs/tables/{deck.STAGE05}/05_pharma_domain_cooccurrence.csv'
    select = lambda path, status, content: dict(path=path, status=status,
                                                sha256=sha256(content).hexdigest() if content else None,
                                                schema=deck.SCHEMA, producer=deck.PRODUCER)
    manifest = {'schema': deck.SCHEMA, 'producer': deck.PRODUCER,
                'artifacts': [select(table, 'estimated', data),
                              select(missing_pdf, 'not_estimable', None),
                              select(stale, 'estimated', b'different'),
                              select(multiple, 'estimated', data), select(multiple, 'estimated', data)]}
    put(tmp_path, deck.QC_MANIFEST, json.dumps(manifest).encode())
    put(tmp_path, table, data)
    put(tmp_path, stale, data)
    put(tmp_path, multiple, data)
    put(tmp_path, 'outputs/tables/BLOCK_b/01_pop_distribution/01_pop_distribution_counts.csv', data)
    put(tmp_path, 'outputs/tables/BlockB/01_pop_distribution_counts.csv', data)
    put(tmp_path, 'outputs/tables/blockA/backup/01_pop_distribution_counts.csv', data)
    put(tmp_path, 'outputs/tables/other/01_pop_distribution_counts.csv', data)
    outside = put(tmp_path, 'outside.csv', data)
    symlink = tmp_path / 'outputs/tables/blockA/01_pop_distribution/01_pop_distribution_by_visit.csv'
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(outside)
    identifiable = 'outputs/tables/blockA/01_pop_distribution/01_table1_by_pop.csv'
    put(tmp_path, identifiable, b'patient_id,count\nsynthetic-100,50\n')
    old = put(tmp_path, deck.ARCHIVE, b'previous deck')
    archive, _ = deck.collect(tmp_path)
    assert archive == old and archive.read_bytes() != b'previous deck'
    with ZipFile(archive) as zip_file:
        members = zip_file.namelist()
        rows = {row['path']: row for row in csv.DictReader(StringIO(zip_file.read('MANIFEST.csv').decode()))}
        assert set(members) == {deck.QC_MANIFEST, table, 'MANIFEST.csv', 'PENDING.md', 'README.md'}
        assert rows[table]['sha256'] == sha256(data).hexdigest()
        assert rows[table]['minimum_positive_count'] == '2'
        assert rows[missing_pdf]['status'] == 'NOT_ESTIMABLE'
        for name in (stale, str(symlink.relative_to(tmp_path)), identifiable):
            assert rows[name]['status'] == 'STALE_OR_INCOMPATIBLE'
        assert rows[multiple]['status'] == 'MULTIPLE_REVIEW'
        assert any(row['status'] == 'OPTIONAL_MISSING' for row in rows.values())
        assert any(row['status'] == 'MISSING_REQUIRED' for row in rows.values())
    previous = archive.read_bytes()
    def fail_replace(src, dst):
        raise OSError('synthetic replacement failure')
    monkeypatch.setattr(deck.os, 'replace', fail_replace)
    with pytest.raises(OSError):
        deck.collect(tmp_path)
    assert archive.read_bytes() == previous
    assert not list(tmp_path.glob('.pharma-deck-*.zip'))
