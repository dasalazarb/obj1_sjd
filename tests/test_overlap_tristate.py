"""Synthetic contract tests for observed inactivity versus non-evaluation."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("overlap_step", ROOT / "src/block_A/06_overlap_glandular.py")
step = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(step)

from src.derivations.overlap_flags import (  # noqa: E402
    EXTRAGLANDULAR_DOMAINS,
    derive_domain_active,
    derive_extraglandular_flags,
    derive_glandular_flags,
    derive_overlap_flags,
)


def test_domain_source_states_and_absent_columns():
    active, evaluable, score = derive_domain_active(pd.Series(["no activity", None, "moderate activity", "nonsense"]))
    assert active.tolist() == [False, pd.NA, True, pd.NA]
    assert evaluable.tolist() == [True, False, True, False]
    assert score.tolist() == [0.0, pd.NA, 2.0, pd.NA]
    flags = derive_extraglandular_flags(pd.DataFrame({"essdai__constitutional": ["no activity"]}))
    assert flags.eg_constitutional_active.iloc[0] == False  # noqa: E712
    assert pd.isna(flags.eg_renal_active.iloc[0])
    assert pd.isna(flags.extraglandular_active.iloc[0])
    assert pd.isna(flags.n_extraglandular_domains_active.iloc[0])


@pytest.mark.parametrize(
    ("g", "e", "status", "overlap"),
    [
        (True, True, "overlap", True), (True, False, "glandular_only", False),
        (False, True, "extraglandular_only", False), (False, False, "neither", False),
        (pd.NA, pd.NA, "unclassifiable", pd.NA), (True, pd.NA, "unclassifiable", pd.NA),
        (pd.NA, True, "unclassifiable", pd.NA), (False, pd.NA, "unclassifiable", False),
        (pd.NA, False, "unclassifiable", False),
    ],
)
def test_overlap_truth_matrix(g, e, status, overlap):
    frame = pd.DataFrame({
        "glandular_active": pd.Series([g], dtype="boolean"),
        "extraglandular_active": pd.Series([e], dtype="boolean"),
        "n_glandular_manifestations_active": pd.Series([pd.NA], dtype="Int64"),
        "n_extraglandular_domains_active": pd.Series([pd.NA], dtype="Int64"),
    })
    result = derive_overlap_flags(frame)
    assert result.overlap_status.iloc[0] == status
    assert result.overlap_evaluable.iloc[0] == (pd.notna(g) and pd.notna(e))
    assert (pd.isna(result.overlap_active.iloc[0]) and pd.isna(overlap)) or result.overlap_active.iloc[0] == overlap


def test_partial_glandular_group_is_unknown_not_negative():
    col = "visit_summary_-_2016_classification_criteria__ic_dry_eye_3month"
    result = derive_glandular_flags(pd.DataFrame({col: ["no"]}))
    assert pd.isna(result.glandular_dry_eye_subjective_active.iloc[0])
    assert pd.isna(result.glandular_active.iloc[0])
    assert pd.isna(result.n_glandular_manifestations_active.iloc[0])


def test_consumers_exclude_unknown_from_domain_denominators():
    meta = EXTRAGLANDULAR_DOMAINS["constitutional"]
    baseline = pd.DataFrame({
        "patient_id": ["synthetic-a", "synthetic-b", "synthetic-c"],
        "glandular_active": pd.Series([True, False, pd.NA], dtype="boolean"),
        "glandular_evaluable": [True, True, False],
        meta["active_col"]: pd.Series([True, False, pd.NA], dtype="boolean"),
        "eg_constitutional_evaluable": [True, True, False],
    })
    # Add other dynamic domains so the existing table contract remains intact.
    for key, domain in EXTRAGLANDULAR_DOMAINS.items():
        if domain["active_col"] not in baseline:
            baseline[domain["active_col"]] = pd.Series([pd.NA] * 3, dtype="boolean")
            baseline[f"eg_{key}_evaluable"] = False
    row = step.associations(baseline).set_index("domain").loc[meta["label"]]
    assert row.n_complete == 2
    assert row.n_missing_or_not_evaluable == 1
