"""Integration assertions for generated canonical visit-spine artifacts."""
import pytest

pd = pytest.importorskip("pandas")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "data/intermediate/00_patient_visit_spine.parquet"

@pytest.mark.skipif(not SPINE.exists(), reason="visit spine is generated from the protected analytic input")
def test_spine_integrity_and_output_visit_ids():
    spine = pd.read_parquet(SPINE)
    assert not spine.duplicated(["patient_id", "visit_date"]).any()
    assert spine["visit_id"].is_unique
    assert (spine.groupby("patient_id")["visit_number"].min() == 0).all()
    baseline = spine[spine.visit_number.eq(0)]
    assert (baseline.visit_date == baseline.observed_baseline_date).all()
    assert (spine.time_since_observed_baseline_days >= 0).all()
    valid_ids = set(spine.visit_id)
    outputs = [
        ROOT / "data/intermediate/01_visit_level_classification.parquet",
        ROOT / "data/intermediate/06_overlap_baseline_patient_level.parquet",
        ROOT / "data/intermediate/06_overlap_longitudinal_dx_temporal_anchor_patient_visit.parquet",
    ]
    for output in outputs:
        if output.exists():
            frame = pd.read_parquet(output)
            assert set(frame["visit_id"].dropna()) <= valid_ids
