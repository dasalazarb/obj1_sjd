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
    GLANDULAR_COLS,
    derive_domain_active,
    derive_extraglandular_flags,
    derive_glandular_flags,
    derive_overlap_flags,
    parse_sicca,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sicca absent", False),
        ("0 - Sicca absent", False),
        ("absence of sicca", False),
        ("sicca present", True),
        ("1 - Sicca present", True),
    ],
)
def test_parse_sicca_observed_states(raw, expected):
    assert parse_sicca(raw) is expected
    result = derive_glandular_flags(pd.DataFrame({
        "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth": [raw]
    }))
    assert result.sicca_active.iloc[0] == expected
    assert result.sicca_evaluable.iloc[0] == True  # noqa: E712


@pytest.mark.parametrize("raw", [None, "unknown"])
def test_parse_sicca_missing_remains_not_evaluable(raw):
    assert pd.isna(parse_sicca(raw))
    result = derive_glandular_flags(pd.DataFrame({
        "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth": [raw]
    }))
    assert pd.isna(result.sicca_active.iloc[0])
    assert result.sicca_evaluable.iloc[0] == False  # noqa: E712


def test_aggregate_sicca_absent_overrides_missing_components():
    result = derive_glandular_flags(pd.DataFrame({
        "visit_summary_-_2016_classification_criteria__ic_symptom_dry_eye_or_dry_mouth": ["sicca absent"],
        "visit_summary_-_2016_classification_criteria__ic_dry_eye_3month": [pd.NA],
        "visit_summary_-_2016_classification_criteria__ic_dry_mouth_3month": [pd.NA],
    }))
    assert result.sicca_active.iloc[0] == False  # noqa: E712
    assert result.sicca_evaluable.iloc[0] == True  # noqa: E712


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
    assert pd.isna(result.glandular_eye_dryness_active.iloc[0])
    assert pd.isna(result.glandular_active.iloc[0])
    assert pd.isna(result.n_glandular_manifestations_active.iloc[0])


def _association_frame(glandular, domain):
    size = len(glandular)
    meta = EXTRAGLANDULAR_DOMAINS["constitutional"]
    baseline = pd.DataFrame({
        "patient_id": [f"synthetic-{i}" for i in range(size)],
        "glandular_active": pd.Series(glandular, dtype="boolean"),
        "glandular_evaluable": pd.Series(glandular, dtype="boolean").notna(),
        meta["active_col"]: pd.Series(domain, dtype="boolean"),
        "eg_constitutional_evaluable": pd.Series(domain, dtype="boolean").notna(),
    })
    for key, item in EXTRAGLANDULAR_DOMAINS.items():
        if item["active_col"] not in baseline:
            baseline[item["active_col"]] = pd.Series([pd.NA] * size, dtype="boolean")
            baseline[f"eg_{key}_evaluable"] = False
    return baseline


def test_consumers_exclude_unknown_from_domain_denominators():
    meta = EXTRAGLANDULAR_DOMAINS["constitutional"]
    baseline = _association_frame([True, False, pd.NA], [True, False, True])
    row = step.associations(baseline).set_index("domain").loc[meta["label"]]
    assert row.n_complete == 2
    assert row.n_missing_or_not_evaluable == 1


def test_glandular_absent_is_pr_reference_not_missing():
    meta = EXTRAGLANDULAR_DOMAINS["constitutional"]
    baseline = _association_frame(
        [True, True, False, False, pd.NA], [True, False, True, False, True]
    )
    row = step.associations(baseline).set_index("domain").loc[meta["label"]]
    assert (row.glandular_pos_domain_pos, row.glandular_pos_domain_neg) == (1, 1)
    assert (row.glandular_neg_domain_pos, row.glandular_neg_domain_neg) == (1, 1)
    assert row.n_complete == 4
    assert row.n_missing_or_not_evaluable == 1
    assert row.prevalence_ratio == pytest.approx(1.0)


def test_pairwise_uses_glandular_not_sicca():
    meta = EXTRAGLANDULAR_DOMAINS["constitutional"]
    baseline = _association_frame([True, False], [True, False])
    baseline["sicca_active"] = pd.Series([False, False], dtype="boolean")
    baseline["sicca_evaluable"] = True
    row = step.associations(baseline).set_index("domain").loc[meta["label"]]
    assert row.glandular_pos_domain_pos == 1
    assert row.glandular_neg_domain_neg == 1
    assert row.n_complete == 2


def _glandular_row(**values):
    raw = {column: [pd.NA] for column in GLANDULAR_COLS.values()}
    for key, value in values.items():
        raw[GLANDULAR_COLS[key]] = [value]
    return derive_glandular_flags(pd.DataFrame(raw)).iloc[0]


def test_eye_dryness_tri_state_and_positive_fallback_sicca():
    positive = _glandular_row(dry_eye_3month="yes")
    negative = _glandular_row(dry_eye_3month="no", sand_gravel_eye="no", tear_subsit="no")
    incomplete = _glandular_row(dry_eye_3month="no", sand_gravel_eye="no")
    assert positive.glandular_eye_dryness_active == True  # noqa: E712
    assert positive.sicca_active == True and positive.glandular_active == True  # noqa: E712
    assert negative.glandular_eye_dryness_active == False  # noqa: E712
    assert pd.isna(incomplete.glandular_eye_dryness_active)


def test_mouth_dryness_tri_state():
    negative = _glandular_row(dry_mouth_3month="no", difficulty_swallowing_dry_food="no")
    positive = _glandular_row(difficulty_swallowing_dry_food="yes")
    assert negative.glandular_mouth_dryness_active == False  # noqa: E712
    assert positive.glandular_mouth_dryness_active == True  # noqa: E712


def test_aggregate_sicca_positive_does_not_localize_or_inflate_count():
    result = _glandular_row(symptom_dry_eye_or_mouth="sicca present")
    assert result.sicca_active == True and result.glandular_active == True  # noqa: E712
    assert pd.isna(result.glandular_eye_dryness_active)
    assert pd.isna(result.glandular_mouth_dryness_active)
    assert pd.isna(result.n_glandular_manifestations_active)


@pytest.mark.parametrize(
    ("source", "derived"),
    [
        ("ocular_stain", "glandular_objective_eye_active"),
        ("salivary_gland_movement", "glandular_objective_mouth_active"),
        ("gland_swell", "glandular_salivary_gland_swelling_active"),
    ],
)
def test_objective_or_swelling_positive_activates_glandular(source, derived):
    result = _glandular_row(**{source: "low activity" if source == "gland_swell" else "positive"})
    assert result[derived] == True  # noqa: E712
    assert result.glandular_active == True  # noqa: E712


def test_glandular_negative_requires_all_sources_negative():
    negative = _glandular_row(
        symptom_dry_eye_or_mouth="sicca absent", ocular_stain="negative",
        lacrimal_dysfunction="negative", salivary_gland_movement="negative",
        gland_swell="no activity",
    )
    unknown = _glandular_row(
        ocular_stain="negative", lacrimal_dysfunction="negative",
        salivary_gland_movement="negative", gland_swell="no activity",
    )
    assert negative.glandular_active == False and negative.glandular_evaluable == True  # noqa: E712
    assert pd.isna(unknown.glandular_active)


def test_zero_observed_reference_prevalence_is_infinite_not_missing():
    pr, lower, upper = step._ratio_ci(a=1, b=1, c=0, d=2)
    assert pr == float("inf")
    assert pd.isna(lower)
    assert pd.isna(upper)


@pytest.mark.parametrize(
    ("eye", "mouth", "expected"),
    [(True, pd.NA, True), (False, False, False), (False, pd.NA, pd.NA)],
)
def test_objective_composite_has_kleene_or_semantics(eye, mouth, expected):
    result = _glandular_row(
        ocular_stain="positive" if eye is True else "negative",
        lacrimal_dysfunction="positive" if eye is True else "negative",
        salivary_gland_movement=(
            "positive" if mouth is True else "negative" if mouth is False else pd.NA
        ),
    )
    observed = result.objective_glandular_dysfunction_active
    assert (pd.isna(observed) and pd.isna(expected)) or observed == expected


def test_objective_or_swelling_can_resolve_unknown_objective():
    positive = _glandular_row(gland_swell="low activity")
    negative = _glandular_row(
        ocular_stain="negative", lacrimal_dysfunction="negative",
        salivary_gland_movement="negative", gland_swell="no activity",
    )
    assert positive.objective_or_swelling_glandular_active == True  # noqa: E712
    assert negative.objective_or_swelling_glandular_active == False  # noqa: E712


def test_generic_association_excludes_unknown_and_keeps_zero_event_reference():
    baseline = _association_frame([True, True, False, False, pd.NA], [True, False, False, False, True])
    result = step.binary_exposure_domain_associations(
        baseline, "glandular_active", "glandular_evaluable", "synthetic", "test"
    )
    row = result.set_index("domain").loc["Constitutional"]
    assert row.n_complete == 4
    assert row.n_complete == sum([
        row.component_pos_domain_pos, row.component_pos_domain_neg,
        row.component_neg_domain_pos, row.component_neg_domain_neg,
    ])
    assert row.n_component_negative == 2
    assert row.prevalence_ratio == float("inf")
    assert "reference group exists" in row.estimability_note


def _sensitivity_frame(domain_values, glandular_values=None):
    size = (
        len(glandular_values)
        if glandular_values is not None
        else max((len(values) for values in domain_values.values()), default=1)
    )
    data = {}
    for key, meta in EXTRAGLANDULAR_DOMAINS.items():
        values = domain_values.get(key, [False] * size)
        data[meta["active_col"]] = pd.Series(values, dtype="boolean")
        data[f"eg_{key}_evaluable"] = pd.Series(values, dtype="boolean").notna()
    glandular_values = glandular_values or [[False] * 5 for _ in range(size)]
    for index, (_, column, _) in enumerate(step.GLANDULAR_COMPONENTS):
        data[column] = pd.Series([row[index] for row in glandular_values], dtype="boolean")
    return pd.DataFrame(data)


def test_no_bio_heme_count_excludes_domains_and_preserves_missingness():
    frame = _sensitivity_frame({
        "biological": [True, True],
        "hematologic": [True, True],
        "articular": [False, pd.NA],
    })
    result = step.add_sensitivity_phenotypes(frame)
    assert result.n_extraglandular_domains_active_no_bio_heme.iloc[0] == 0
    assert pd.isna(result.n_extraglandular_domains_active_no_bio_heme.iloc[1])

    raw = {
        meta["col"]: ["low activity" if key in {"biological", "hematologic"} else "no activity"]
        for key, meta in EXTRAGLANDULAR_DOMAINS.items()
    }
    derived = derive_extraglandular_flags(pd.DataFrame(raw)).iloc[0]
    assert derived.n_extraglandular_domains_active == 2
    assert derived.n_extraglandular_domains_active_no_bio_heme == 0


def test_glandular_completeness_is_independent_of_aggregate_activity():
    frame = _sensitivity_frame(
        {},
        glandular_values=[
            [True, False, True, False, False],
            [True, False, pd.NA, False, False],
        ],
    )
    frame["glandular_active"] = pd.Series([True, True], dtype="boolean")
    result = step.add_sensitivity_phenotypes(frame)
    assert result.glandular_phenotype_complete.tolist() == [True, False]
    assert result.glandular_active.tolist() == [True, True]


def test_leave_one_domain_out_removes_biological_activity():
    frame = _sensitivity_frame({"biological": [True, True, True]})
    frame = step.add_sensitivity_phenotypes(frame)
    frame["n_glandular_manifestations_active"] = pd.Series([0, 1, 2], dtype="Int64")
    result = step.leave_one_domain_out(frame).set_index("excluded_domain")
    assert result.loc["Biological", "n_complete"] == 3
    # Directly verify the synthetic count contract used by the leave-one-out routine.
    remaining = [
        meta["active_col"] for key, meta in EXTRAGLANDULAR_DOMAINS.items()
        if key != "biological"
    ]
    assert frame[remaining].sum(axis=1).eq(0).all()
