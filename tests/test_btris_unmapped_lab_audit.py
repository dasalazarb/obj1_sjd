"""Focused tests for the descriptive BTRIS unmapped-lab audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "src" / "20_btris_visit_date_match_report.py"
SPEC = importlib.util.spec_from_file_location("btris_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _labs() -> pd.DataFrame:
    rows = []
    for patient in range(1, 11):
        repeats = 3 if patient == 1 else 1
        for repeat in range(repeats):
            rows.append({
                "patient_id": patient,
                "order_name_original": "Free light chains",
                "cluster_name_original": "Free Kappa Light Chain",
                "lab_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=repeat),
                "result_original": "<2.0" if repeat == 0 else "2.5",
                "result_numeric": pd.NA if repeat == 0 else 2.5,
                "result_text": pd.NA,
                "unit_original": "mg/L",
                "mapping_status": "unexpected_unmapped",
                "semantic_mapping_status": "unexpected_unmapped",
                "canonical_analyte": pd.NA,
            })
    rows.extend([
        {
            "patient_id": 1, "order_name_original": "Comment",
            "cluster_name_original": "Administrative note", "lab_date": pd.Timestamp("2021-01-01"),
            "result_original": "see note", "result_numeric": pd.NA,
            "result_text": "see note", "unit_original": pd.NA,
            "mapping_status": "unexpected_unmapped", "semantic_mapping_status": "unexpected_unmapped",
            "canonical_analyte": pd.NA,
        },
        {
            "patient_id": 99, "order_name_original": "Already mapped",
            "cluster_name_original": "Creatinine", "lab_date": pd.Timestamp("2021-01-01"),
            "result_original": "1.0", "result_numeric": 1.0, "result_text": pd.NA,
            "unit_original": "mg/dL", "mapping_status": "expected_mapped",
            "semantic_mapping_status": "mapped", "canonical_analyte": "creatinine",
        },
    ])
    return pd.DataFrame(rows)


def test_candidates_measure_repeats_and_keep_suggestions_in_audit_only():
    labs = _labs()
    candidates = MODULE.build_unmapped_lab_candidates(
        labs,
        pd.DataFrame({"order_name": ["Free light chains"], "cluster_name": ["Free Kappa Light Chain"]}),
        cluster_semantics={
            "Free Kappa Light Chain": {
                "canonical_analyte": "free_kappa_light_chain",
                "lab_family": "immunology",
                "analytic_role": "numeric",
            }
        },
    )
    kappa = candidates.iloc[0]
    assert (kappa.n_rows, kappa.n_patients, kappa.n_patients_ge2, kappa.n_patients_ge3) == (12, 10, 1, 1)
    assert (kappa.n_numeric_exact, kappa.pct_numeric_exact) == (2, 100 / 6)
    assert kappa.present_in_reference == True  # noqa: E712
    assert kappa.suggested_mapping_source == "cluster_semantic_fallback"
    assert kappa.audit_priority == "HIGH_PRIORITY_MAP"
    assert labs.canonical_analyte.isna().sum() == 13


def test_summary_counts_unique_patients_per_priority_and_writer_outputs(tmp_path, capsys):
    labs = _labs()
    candidates, summary = MODULE.write_unmapped_lab_audit(
        labs, pd.DataFrame(), tmp_path,
        cluster_semantics={"Free Kappa Light Chain": ("free_kappa_light_chain", "immunology", "numeric")},
        pair_semantics={}, semantic_overrides={},
    )
    high = summary.set_index("audit_priority").loc["HIGH_PRIORITY_MAP"]
    admin = summary.set_index("audit_priority").loc["ADMINISTRATIVE_OR_TEXT"]
    assert (high.n_pairs, high.n_rows, high.n_patients_unique) == (1, 12, 10)
    assert (admin.n_pairs, admin.n_rows, admin.n_patients_unique) == (1, 1, 1)
    assert (tmp_path / MODULE.UNMAPPED_CANDIDATES_NAME).exists()
    assert (tmp_path / MODULE.UNMAPPED_SUMMARY_NAME).exists()
    output = capsys.readouterr().out
    assert "BTRIS UNMAPPED LAB AUDIT" in output
    assert "Mapped pairs: 1" in output
    assert list(candidates.audit_priority) == ["HIGH_PRIORITY_MAP", "ADMINISTRATIVE_OR_TEXT"]
