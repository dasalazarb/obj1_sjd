import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "src" / "block_A" / "07_comorbidities.py"
SPEC = importlib.util.spec_from_file_location("comorbidities", MODULE_PATH)
comorbidities = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comorbidities
SPEC.loader.exec_module(comorbidities)


def all_conditions():
    return [*comorbidities.PAST_MEDICAL_HISTORY_CONDITIONS,
            *comorbidities.RHEUMATOLOGIC_NON_SAID_CONDITIONS,
            *comorbidities.CONCOMITANT_SAID_CONDITIONS]


def test_families_exclude_anxiety_sjd_manifestations_and_prohibited_sources():
    names = {c.name for c in all_conditions()}
    assert "anxiety" not in names
    excluded = {"ild", "lung_fibrosis", "pulmonary_hypertension", "pleuritis",
                "pulmonary_granulomas", "raynaud", "peripheral_neuropathy",
                "renal_tubular_acidosis", "glomerulonephritis", "vasculitis",
                "cryoglobulinemia", "lymphoma", "myositis"}
    assert names.isdisjoint(excluded)
    sources = [source for c in all_conditions() for source in (*c.primary, *c.detail_columns)]
    assert not any(source.startswith(comorbidities.PROHIBITED_SOURCE_PREFIXES) for source in sources)
    assert "polymyositis" in names and "dermatomyositis" in names


def test_pmh_is_documented_history_and_blank_is_not_current_disease_evidence():
    spec = comorbidities.PAST_MEDICAL_HISTORY_CONDITIONS[0]
    raw = pd.DataFrame({spec.primary[0]: ["Yes", None, "No"]})
    result = comorbidities.derive_comorbidity_indicators(raw)
    assert result[spec.name].tolist() == [True, False, False]
    assert f"{spec.name}_status" not in result
    summary = comorbidities.summarize_historical_family(result, [spec]).iloc[0]
    assert summary.n_documented_history == 1
    assert summary.n_total_patients == 3
    assert summary.percent_documented_total_cohort == 100 / 3
    assert "documented" in summary.summary_label.lower()


def test_rheumatologic_status_priority_and_primary_exposure():
    g = pd.Series([True, False, True, None], dtype="boolean")
    h = pd.Series([True, True, False, None], dtype="boolean")
    c = pd.Series([True, False, False, None], dtype="boolean")
    status = comorbidities.derive_condition_status(g, h, c)
    assert status.tolist() == ["confirmed_present", "history_only", "status_uncertain", "no_comorbidity"]
    spec = comorbidities.RHEUMATOLOGIC_NON_SAID_CONDITIONS[0]
    frame = pd.DataFrame({f"{spec.name}_status": status})
    exposure = comorbidities.apply_exposure_definition(frame, spec).exposure
    assert exposure.iloc[0] == 1
    assert exposure.iloc[3] == 0
    assert exposure.iloc[1:3].isna().all()


def test_family_burdens_are_separate_and_no_combined_total_exists():
    pmh = comorbidities.PAST_MEDICAL_HISTORY_CONDITIONS[0]
    rheum = comorbidities.RHEUMATOLOGIC_NON_SAID_CONDITIONS[0]
    said = comorbidities.CONCOMITANT_SAID_CONDITIONS[0]
    raw = pd.DataFrame({pmh.primary[0]: [True], rheum.primary[2]: [True], said.primary[2]: [True]})
    result = comorbidities.derive_comorbidity_indicators(raw)
    assert result[pmh.name].iloc[0] and result[rheum.name].iloc[0] and result[said.name].iloc[0]
    assert set(comorbidities.PROGRESSION_CONDITION_NAMES) == {c.name for c in all_conditions()}
    forbidden_totals = {"n_prespecified_comorbidities", "n_comorbidities", "total_comorbidity_burden"}
    assert forbidden_totals.isdisjoint(result.columns)


def test_progression_models_cover_all_families_and_retain_rheum_primary_contrast():
    spec = comorbidities.RHEUMATOLOGIC_NON_SAID_CONDITIONS[0]
    frame = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        f"{spec.name}_status": ["confirmed_present", "no_comorbidity",
                                "history_only", "status_uncertain"],
    })
    result = comorbidities.restrict_to_primary_exposure(frame, spec)

    assert result.patient_id.tolist() == ["P1", "P2"]
    assert result[spec.name].tolist() == [1, 0]
    assert comorbidities.PROGRESSION_CONDITIONS == all_conditions()
    assert [stem for stem, _ in comorbidities.PROGRESSION_FAMILIES] == [
        "general_medical_comorbidities", "rheumatologic_comorbidities",
        "concomitant_said"]


def test_pmh_progression_exposure_uses_documented_history_flag():
    spec = comorbidities.PAST_MEDICAL_HISTORY_CONDITIONS[0]
    frame = pd.DataFrame({"patient_id": ["P1", "P2", "P3"],
                          spec.name: pd.Series([True, False, None], dtype="boolean")})
    result = comorbidities.restrict_to_primary_exposure(frame, spec)
    assert result.patient_id.tolist() == ["P1", "P2"]
    assert result[spec.name].tolist() == [1, 0]
    assert comorbidities.progression_exposure_column(spec) == spec.name


def test_progression_exposure_selection_does_not_depend_on_display_family_label():
    pmh = comorbidities.PAST_MEDICAL_HISTORY_CONDITIONS[0]
    renamed = comorbidities.Condition(
        pmh.name, pmh.label, pmh.primary, "GENERAL_MEDICAL_COMORBIDITIES",
        pmh.clinical_category, pmh.detail_columns, pmh.notes)
    frame = pd.DataFrame({renamed.name: pd.Series([True, False], dtype="boolean")})

    result = comorbidities.restrict_to_primary_exposure(frame, renamed)

    assert comorbidities.progression_exposure_column(renamed) == renamed.name
    assert result[renamed.name].tolist() == [1, 0]


def _pop_frame(positives):
    rows=[]
    spec=comorbidities.RHEUMATOLOGIC_NON_SAID_CONDITIONS[0]
    for i,(positive,N) in enumerate(zip(positives,[100,100,100]),1):
        rows.extend({"baseline_pop":f"Pop{i}",spec.name:j<positive} for j in range(N))
    return pd.DataFrame(rows),spec


def test_adequate_pop_table_uses_pearson_and_fdr():
    frame,spec=_pop_frame([20,30,40])
    result=comorbidities.summarize_family_by_pop(frame,[spec]).iloc[0]
    assert result.global_test == "Pearson chi-square"
    assert pd.notna(result.global_p_value) and pd.notna(result.fdr_bh_q_value)
    assert result.minimum_expected_cell >= 5


def test_sparse_pop_table_is_descriptive_but_pairwise_fisher_runs():
    frame,spec=_pop_frame([1,2,3])
    result=comorbidities.summarize_family_by_pop(frame,[spec]).iloc[0]
    assert result.global_test == "descriptive only - sparse table"
    assert pd.isna(result.global_p_value) and pd.isna(result.fdr_bh_q_value)
    assert result.sparse_table_flag
    assert pd.notna(result.fisher_exact_p_value)


def test_fdr_preserves_missing_values():
    result=comorbidities.apply_fdr(pd.Series([0.01,np.nan,0.2]))
    assert pd.isna(result.iloc[1])
    assert result.iloc[[0,2]].notna().all()
