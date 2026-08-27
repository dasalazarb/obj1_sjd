import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src" / "block_A" / "09_pros_longitudinal.py"
SPEC = importlib.util.spec_from_file_location("pros_longitudinal", MODULE_PATH)
pros = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pros)


def test_longitudinal_descriptives_are_labelled_as_available_measurements(monkeypatch):
    frame = pd.DataFrame(
        {
            "patient_id": [1, 1, 2, 2],
            "clinical_anchor_date": pd.to_datetime(
                ["2020-06-01", "2022-06-01", "2020-03-01", "2021-03-01"]
            ),
            "time_since_clinical_baseline_years": [0.5, 2.5, 0.25, 1.25],
            "esspri_total": [2.0, 5.0, 4.0, 3.0],
        }
    )
    monkeypatch.setattr(pros, "measures", lambda: [("ESSPRI", "esspri_total")])
    monkeypatch.setattr(
        pros,
        "fit_model",
        lambda *_: {
            "model_used": "LME",
            "model_status": "fitted",
            "annual_change_estimate": 0.0,
            "CI95_low": -1.0,
            "CI95_high": 1.0,
            "p_value": 1.0,
        },
    )

    summary, model_qc, eligible = pros.longitudinal_tables(frame)

    assert eligible == {"ESSPRI": 2}
    assert summary.loc[0, "first_available_mean"] == 3.0
    assert summary.loc[0, "last_available_mean"] == 4.0
    assert summary.loc[0, "mean_first_to_last_change"] == 1.0
    assert summary.loc[0, "median_observed_pro_span_years"] == 1.5
    assert not {
        "baseline_mean_or_median",
        "last_mean_or_median",
        "mean_or_median_change",
        "median_followup_years",
    }.intersection(summary.columns)
    assert model_qc.loc[0, "longitudinal_summary_basis"] == (
        "first_and_last_available_measurements"
    )


def test_visit_number_output_is_explicitly_descriptive(monkeypatch):
    frame = pd.DataFrame(
        {"patient_id": [1], "clinical_visit_number": [7], "esspri_total": [2.0]}
    )
    monkeypatch.setattr(pros, "measures", lambda: [("ESSPRI", "esspri_total")])

    result = pros.by_visit_table(frame)

    assert result.loc[0, "n_available"] == 1
    assert result.loc[0, "methodological_note"] == pros.CLINICAL_VISIT_DESCRIPTIVE_NOTE
    assert "descriptive summaries only" in result.loc[0, "methodological_note"]
