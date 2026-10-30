"""Privacy and ordering regressions for the Step-10 share candidate."""
import pandas as pd

from src.block_A._internal.step10_share import (
    SHARE_DROP_COLUMNS, build_share_dataset, harmonize_interval_names,
    organize_analytic_columns,
)


def example():
    return pd.DataFrame({
        "lab__crp__value": [1.1, 4.2, 2.0],
        "patient_id": ["100", "100", "200"],
        "clinical_episode_id": ["x1", "x2", "x3"],
        "clinical_baseline_episode_id": ["x1", "x1", "x3"],
        "clinical_visit": [True, True, True],
        "visit_type": ["clinical"] * 3,
        "clinical_visit_number": [1, 2, 1],
        "clinical_anchor_date": pd.to_datetime(["2020-02-29", "2021-02-28", "2020-04-04"]),
        "clinical_baseline_date": pd.to_datetime(["2020-02-29", "2020-02-29", "2020-04-04"]),
        "episode_start_date": pd.to_datetime(["2020-02-29", "2021-02-28", "2020-04-04"]),
        "episode_end_date": pd.to_datetime(["2020-02-29", "2021-02-28", "2020-04-04"]),
        "is_clinical_baseline": [True, False, True],
        "time_since_clinical_baseline_days": [0, 365, 0],
        "spine__sjd_ever_1_2_4": [1, 1, 1],
        "spine__interval_name": [
            "prelude | Natural History Protocol 478 Interval | detail",
            "Optional Evaluation | Phase 2: 4th Full Evaluation",
            "Phase 1: Initial Full Evaluation | Optional Evaluation",
        ],
        "pop__status": ["Pop1", "Pop2", "Pop3"],
        "demo__age_at_visit": [91, 92, 39],
        "lab__clinical_note__text": ["Name: John Doe", "2020-05-01", "private"],
        "medication__current_names": ["rare_x, rare_y", "rare_x", "rare_z"],
        "ovl__status": ["overlap", "neither", "glandular_only"],
    })


def test_harmonized_labels_preserve_valid_phase():
    labels = harmonize_interval_names(example().spine__interval_name)
    assert labels.tolist() == [
        "Natural History Protocol 478 Interval",
        "Phase 2: 4th Full Evaluation",
        "Phase 1: Initial Full Evaluation",
    ]
    assert harmonize_interval_names(pd.Series(["Optional Evaluation", pd.NA])).iloc[0] == "Optional Evaluation"


def test_share_preserves_episodes_but_not_identifiers_or_exact_dates():
    original = example()
    unchanged = original.copy(deep=True)
    share, qc = build_share_dataset(original)
    pd.testing.assert_frame_equal(original, unchanged)
    assert len(share) == 3 and share.patient_id.nunique() == 2
    assert share.patient_id.tolist() == ["pt1", "pt1", "pt2"]
    assert not share.patient_id.isin(["100", "200"]).any()
    assert not set(SHARE_DROP_COLUMNS) & set(share.columns)
    assert "lab__clinical_note__text" in share
    assert share.medication__current_names.tolist() == original.medication__current_names.tolist()
    assert "clinical_anchor_date" in share and "clinical_baseline_date" in share
    shifted = pd.to_datetime(share.clinical_anchor_date)
    assert ((shifted - original.clinical_anchor_date).dt.days >= 36525).all()
    assert (shifted.iloc[1] - shifted.iloc[0]) == (
        original.clinical_anchor_date.iloc[1] - original.clinical_anchor_date.iloc[0]
    )
    shifted_baseline = pd.to_datetime(share.clinical_baseline_date)
    assert shifted_baseline.iloc[0] == shifted_baseline.iloc[1] == shifted.iloc[0]
    assert share.demo__age_at_visit.tolist() == [90, 90, 39]
    assert share.spine__interval_name.iloc[1] == "Phase 2: 4th Full Evaluation"
    assert qc["patient_id_method"].startswith("sequential pt1")
    assert qc["not_hipaa_safe_harbor"] is True
    assert qc["n_share_columns"] == len(share.columns)


def test_ordering_is_repeatable_and_groups_medication_last():
    original = example()
    a = organize_analytic_columns(original)
    b = organize_analytic_columns(original[original.columns[::-1]])
    assert list(a.columns) == list(b.columns)
    assert a.columns[0] == "patient_id"
    assert a.columns.get_loc("spine__interval_name") < a.columns.get_loc("pop__status")
    assert a.columns.get_loc("pop__status") < a.columns.get_loc("lab__crp__value")
    assert a.columns.get_loc("lab__crp__value") < a.columns.get_loc("medication__current_names")


def test_unknown_string_and_source_dates_never_leak():
    original = example().assign(
        source_protocol=["11-D-0172"] * 3,
        patient_subject_number=["912001"] * 3,
        calendar_year=[2020, 2021, 2020],
        free_text=["contains identifying text"] * 3,
        unexpected_date=["2020-01-01 | 2020-02-01"] * 3,
    )
    share, qc = build_share_dataset(original)
    for forbidden in ("source_protocol", "patient_subject_number", "calendar_year",
                      "free_text", "unexpected_date"):
        assert forbidden not in share
    assert set(("patient_subject_number", "source_protocol")) <= set(qc["removed_columns"])



def test_curated_text_preserves_findings_nulls_and_masks_identifiers():
    import json
    original = example().assign(**{
        "lab__ana__text": ["positive", "negative", None],
        "lab__clinical_note__text": [
            "ANA positive; MRN: ABC123; email a@example.org; 2020-05-01",
            "sicca reported; phone (212) 555-1234; Name: Jane Doe",
            None,
        ],
        "ext__sicca_notes": ["sicca reported", None, "ANA positive"],
        "spine__source_notes": ["administrative"] * 3,
        "lab__mrn": ["ABC123"] * 3,
        "demo__dob": ["1970-01-01"] * 3,
        "unapproved_notes": ["unreviewed"] * 3,
    })
    before = original.copy(deep=True)
    share, qc = build_share_dataset(original)
    pd.testing.assert_frame_equal(original, before)
    assert share["lab__ana__text"].iloc[:2].tolist() == ["positive", "negative"]
    assert pd.isna(share["lab__ana__text"].iloc[2])
    assert share["lab__clinical_note__text"].iloc[0] == "ANA positive; [IDENTIFIER]; email [EMAIL]; [DATE]"
    assert share["lab__clinical_note__text"].iloc[1] == "sicca reported; phone [PHONE]; [NAME]"
    assert pd.isna(share["lab__clinical_note__text"].iloc[2])
    assert share["ext__sicca_notes"].iloc[0] == "sicca reported"
    for name in ["spine__source_notes", "lab__mrn", "demo__dob", "unapproved_notes"]:
        assert name not in share
    assert qc["text_cells_with_identifier_patterns_masked"]["lab__clinical_note__text"] == 2
    assert "unapproved_notes" in qc["text_columns_review_required"]
    assert qc["contains_unstructured_text"] is True
    assert qc["manual_text_privacy_review_required"] is True
    assert qc["status"] == "PSEUDONYMIZED_REVIEW_ONLY_NOT_ANONYMIZED"
    assert not any(value in json.dumps(qc) for value in ["ABC123", "a@example.org", "Jane Doe", "ANA positive"])
    assert list(share.columns) == list(organize_analytic_columns(share).columns)
    assert share["lab__crp__value"].tolist() == original["lab__crp__value"].tolist()


def test_no_free_text_flags_and_unknown_only_review():
    source = pd.DataFrame({"patient_id": ["a"], "lab__crp__value": [1.2], "demo__sex": ["F"]})
    _, qc = build_share_dataset(source)
    assert qc["contains_unstructured_text"] is False
    assert qc["manual_text_privacy_review_required"] is False
    assert qc["text_columns_retained"] == []
    assert qc["text_cells_with_identifier_patterns_masked"] == {}
    _, qc = build_share_dataset(source.assign(unknown_notes="review"))
    assert qc["contains_unstructured_text"] is False
    assert qc["manual_text_privacy_review_required"] is True


def test_ambiguous_interval_is_counted_without_inventing_phase():
    label = "Optional Evaluation | Phase 1 | Phase 2"
    assert harmonize_interval_names(pd.Series([label])).iloc[0] == label
    _, qc = build_share_dataset(pd.DataFrame({"patient_id": ["a"], "spine__interval_name": [label]}))
    assert qc["interval_labels_ambiguous_count"] == 1
