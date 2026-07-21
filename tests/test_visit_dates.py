import pandas as pd
from src.derivations.visit_dates import add_parsed_visit_dates, add_visit_timing, collapse_patient_visit_rows, normalize_patient_id, parse_visit_date_fragments

def test_date_fragments():
    assert parse_visit_date_fragments("2024-01-15")["date_parse_status"] == "single_valid_date"
    assert parse_visit_date_fragments("2024-01-15 | 2024-01-15")["date_parse_status"] == "multiple_valid_dates_same_day"
    different = parse_visit_date_fragments("2024-01-15 | 2024-01-20")
    assert different["visit_date"] == pd.Timestamp("2024-01-15") and different["visit_date_max"] == pd.Timestamp("2024-01-20")
    assert different["date_parse_status"] == "multiple_valid_dates_different_days"
    assert parse_visit_date_fragments("2024-01-15 | unknown")["date_parse_status"] == "partially_parsed"
    invalid = parse_visit_date_fragments("unknown")
    assert pd.isna(invalid["visit_date"]) and invalid["date_parse_status"] == "no_valid_date"

def test_ids_timing_and_duplicates():
    assert normalize_patient_id("00123") == "00123" and normalize_patient_id("123.0") == "123"
    source = pd.DataFrame({"patient_id": ["001", "001", "001", "001"], "visit_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-07-01", "2025-01-01"])})
    keys, audit = collapse_patient_visit_rows(source)
    assert len(keys) == 3 and audit.loc[audit.n_source_rows.eq(2), "had_multiple_source_rows"].all()
    timed = add_visit_timing(keys)
    assert timed.visit_number.tolist() == [0, 1, 2]
    assert timed.observed_baseline_date.iloc[0] == pd.Timestamp("2024-01-01")
    assert timed.time_since_observed_baseline_days.is_monotonic_increasing


def test_add_parsed_visit_dates_replaces_existing_helper_columns():
    source = pd.DataFrame({
        "ids__patient_record_number": ["123.0"],
        "ids__visit_date": ["2024-01-15 | 2024-01-20"],
        "visit_date": ["stale value"],
        "visit_date_raw": ["also stale"],
    })

    result = add_parsed_visit_dates(source, "ids__patient_record_number", "ids__visit_date")

    assert result.columns.tolist().count("visit_date") == 1
    assert result.loc[0, "patient_id"] == "123"
    assert result.loc[0, "visit_date"] == pd.Timestamp("2024-01-15")
    assert result.loc[0, "visit_date_raw"] == "2024-01-15 | 2024-01-20"
