from pathlib import Path

import numpy as np
import pandas as pd

from src.block_A._serology_longitudinal_plots import (
    adjust_global_lab_pvalues_bh,
    assign_exclusive_class_group,
    build_classification,
    build_lab_episode_plot_frame,
    compute_shared_axes,
    export_figure_pair,
    normalize_class_history,
    render_lab_categorical_panels,
    render_lab_trajectory_panels,
    safe_lab_slugs,
    shorten_category_label,
    summarize_categorical_by_group_visit,
    summarize_observed_by_group_visit,
)


def _spine(patient="p1"):
    return pd.DataFrame({
        "patient_id": [patient] * 3,
        "clinical_episode_id": ["e1", "e2", "e3"],
        "clinical_visit_number": pd.Series([1, 2, 3], dtype="Int64"),
        "clinical_anchor_date": pd.to_datetime(["2020-01-01", "2021-01-01", "2023-01-01"]),
        "time_since_clinical_baseline_years": [0.0, 1.0, 3.0],
    })


def _selected(types=("exact_numeric", "qualitative", "exact_numeric"), values=(2.0, np.nan, 3.4)):
    return pd.DataFrame({
        "patient_id": ["p1"] * 3, "clinical_episode_id": ["e1", "e2", "e3"], "lab_id": ["Lab X"] * 3,
        "canonical_analyte": ["Analyte"] * 3, "selected_value_type": types,
        "selected_value_numeric_harmonized": values, "selected_value_numeric": values,
        "selected_value_text": [pd.NA, "Texto de nada que ver con los valores numericos", pd.NA],
        "selected_unit_harmonized": ["mg/dL"] * 3,
        "unit_harmonization_status": ["same_as_canonical"] * 3,
        "result_conflict": [False] * 3, "unit_conflict": [False] * 3,
        "selected_operator": [pd.NA] * 3, "n_measurements_in_episode": [1] * 3,
    })


def test_class_histories_are_normalized_and_exclusive():
    assert normalize_class_history("1.0 | 2") == ("1", "2")
    assert assign_exclusive_class_group([1, 1, 1]) == "A"
    assert assign_exclusive_class_group([1, 2]) == "B"
    assert assign_exclusive_class_group([1, 4]) == "A"
    assert assign_exclusive_class_group([1, 2, 4]) == "B"
    assert assign_exclusive_class_group([4]) == "C"
    assert assign_exclusive_class_group([1, 3]) == "unclassified_or_other"
    assert assign_exclusive_class_group("Pop1") == "unclassified_or_other"


def test_classification_uses_full_spine_and_reports_overlap():
    all_spine = pd.DataFrame({"patient_id": ["a", "a", "b", "b"],
                              "visit_summary_form__sjogrens_class": [1, 1, 2, 4]})
    classes, qc = build_classification(all_spine)
    assert classes.set_index("patient_id").class_group.to_dict() == {"a": "A", "b": "B"}
    assert bool(classes.set_index("patient_id").loc["b", "class_4_trajectory_highlight"])
    assert qc.loc[0, "n_ever_2_and_4"] == 1


def test_class_4_is_highlight_not_group_override_when_1_or_2_exists():
    all_spine = pd.DataFrame(
        {
            "patient_id": ["four_to_one", "four_to_one", "one_to_four",
                           "one_to_four", "two_to_four", "two_to_four", "only_four"],
            "visit_summary_form__sjogrens_class": [4, 1, 1, 4, 2, 4, 4],
        }
    )
    classes, qc = build_classification(all_spine)
    indexed = classes.set_index("patient_id")
    assert indexed.class_group.to_dict() == {
        "four_to_one": "A", "one_to_four": "A", "two_to_four": "B", "only_four": "C"
    }
    assert not bool(indexed.loc["four_to_one", "class_4_trajectory_highlight"])
    assert indexed.loc[["one_to_four", "two_to_four"],
                       "class_4_trajectory_highlight"].all()
    assert not bool(indexed.loc["only_four", "class_4_trajectory_highlight"])
    assert qc.loc[0, "n_group_A_or_B_later_transition_to_4"] == 2


def test_text_between_numeric_values_is_not_imputed():
    frame = build_lab_episode_plot_frame(_spine(), _selected(), pd.DataFrame({"patient_id": ["p1"], "class_group": ["A"]}))
    assert frame.value_status.tolist() == ["valid_numeric", "documented_nonnumeric", "valid_numeric"]
    assert frame.plot_value.dropna().tolist() == [2.0, 3.4]
    assert 2.7 not in frame.plot_value.tolist()
    summary = summarize_observed_by_group_visit(frame, reps=20)
    assert summary.set_index("clinical_visit_number").loc[2, "n_patients_numeric"] == 0


def test_internal_missing_and_censored_are_never_numeric():
    selected = _selected(types=("exact_numeric", "censored_numeric", "exact_numeric"))
    selected.loc[1, "selected_operator"] = "<"
    frame = build_lab_episode_plot_frame(_spine(), selected, pd.DataFrame({"patient_id": ["p1"], "class_group": ["A"]}))
    assert frame.loc[1, "value_status"] == "censored_numeric"
    assert np.isnan(frame.loc[1, "plot_value"])
    missing = build_lab_episode_plot_frame(_spine(), selected.drop(index=1), pd.DataFrame({"patient_id": ["p1"], "class_group": ["A"]}))
    assert missing.loc[1, "value_status"] == "not_measured"
    assert bool(missing.loc[1, "draw_missing_marker"])


def test_conflicts_and_nonfinite_values_are_excluded():
    selected = _selected(); selected.loc[0, "unit_conflict"] = True; selected.loc[2, "selected_value_numeric_harmonized"] = np.inf
    frame = build_lab_episode_plot_frame(_spine(), selected, pd.DataFrame({"patient_id": ["p1"], "class_group": ["A"]}))
    assert frame.loc[0, "value_status"] == "noncomparable_or_conflict"
    assert frame.loc[2, "value_status"] == "documented_nonnumeric"
    assert not frame.included_in_mean.any()


def test_shared_axis_bh_and_collision_safe_slugs():
    assert compute_shared_axes([0, 10, 0, 15, 2, 22]) == (-1.1, 23.1)
    models = pd.DataFrame({"statistical_status": ["ok", "ok", "missing_group"], "p_raw": [.01, .04, np.nan]})
    adjusted = adjust_global_lab_pvalues_bh(models)
    assert adjusted.q_BH.iloc[:2].tolist() == [.02, .04]
    assert pd.isna(adjusted.q_BH.iloc[2])
    slugs = safe_lab_slugs(["A+B", "A B"])
    assert len(set(slugs.values())) == 2


def test_renderer_exports_vector_pdf_and_high_resolution_png(tmp_path: Path):
    import matplotlib.pyplot as plt
    from PIL import Image
    frames = []
    for patient, group, shift in [("p1", "A", 0), ("p2", "B", 2), ("p3", "C", 4)]:
        selected = _selected(); selected["patient_id"] = patient
        frames.append(build_lab_episode_plot_frame(_spine(patient), selected,
                                                    pd.DataFrame({"patient_id": [patient], "class_group": [group]})))
    data = pd.concat(frames, ignore_index=True); summary = summarize_observed_by_group_visit(data, reps=10)
    fig = render_lab_trajectory_panels(data, summary)
    pdf, png = tmp_path / "plot.pdf", tmp_path / "plot.png"
    export_figure_pair(fig, pdf, png, dpi=300); plt.close(fig)
    assert pdf.read_bytes().startswith(b"%PDF")
    with Image.open(png) as image:
        assert image.width >= 3000 and image.height >= 1000


def test_categorical_summary_keeps_all_classes_and_shortens_only_display_label():
    selected = _selected(
        types=("qualitative", "qualitative", "qualitative"),
        values=(np.nan, np.nan, np.nan),
    )
    selected["selected_value_text"] = ["Positive", "Negative", "Indeterminate"]
    frame = build_lab_episode_plot_frame(
        _spine(),
        selected,
        pd.DataFrame({"patient_id": ["p1"], "class_group": ["A"]}),
    )
    summary = summarize_categorical_by_group_visit(frame)
    assert set(summary.category_full) == {"Positive", "Negative", "Indeterminate"}
    assert shorten_category_label("Positive") == "Posit..."
    assert shorten_category_label("Neg") == "Neg"
    assert set(summary.category_label) == {"Posit...", "Negat...", "Indet..."}


def test_categorical_renderer_exports_all_observed_categories(tmp_path: Path):
    import matplotlib.pyplot as plt

    frames = []
    for patient, group, labels in [
        ("p1", "A", ("Positive", "Negative", "Positive")),
        ("p2", "B", ("Negative", "Indeterminate", "Negative")),
        ("p3", "C", ("Borderline long text", "Positive", "Borderline long text")),
    ]:
        selected = _selected(
            types=("qualitative", "qualitative", "qualitative"),
            values=(np.nan, np.nan, np.nan),
        )
        selected["patient_id"] = patient
        selected["selected_value_text"] = list(labels)
        frames.append(
            build_lab_episode_plot_frame(
                _spine(patient),
                selected,
                pd.DataFrame({"patient_id": [patient], "class_group": [group]}),
            )
        )
    data = pd.concat(frames, ignore_index=True)
    summary = summarize_categorical_by_group_visit(data)
    assert summary.category_full.nunique() == 4
    fig = render_lab_categorical_panels(data, summary)
    pdf, png = tmp_path / "categorical.pdf", tmp_path / "categorical.png"
    export_figure_pair(fig, pdf, png, dpi=150)
    plt.close(fig)
    assert pdf.exists()
    assert png.exists()
