"""Focused tests for representation-sensitivity comparison contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


SCRIPT = (Path(__file__).parents[1] / "src/studies/graph" /
          "02_graph_representation_sensitivity.py")
SPEC = importlib.util.spec_from_file_location("graph_representation_sensitivity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PHARMA_SPEC = importlib.util.spec_from_file_location(
    "pharma_lab_profile", SCRIPT.parents[1] / "pharma" / "00_profile_labs.py")
PHARMA = importlib.util.module_from_spec(PHARMA_SPEC)
PHARMA_SPEC.loader.exec_module(PHARMA)


@pytest.mark.parametrize("pop_columns", [
    {"pop__status": ["Pop1", "Pop2", "Pop3"]},
    {"pop_status": ["Pop1", "Pop2", "Pop3"]},
    {"pop__status": ["Pop1", "Pop2", "Pop3"], "pop_status": ["Pop3"] * 3},
])
def test_patient_membership_resolves_pop_labels(pop_columns):
    baseline = pd.DataFrame({"patient_id": ["p1", "p2", "p3"], **pop_columns})
    mapped = {"hard": np.array([0., 1., np.nan]),
              "probabilities": np.array([[.8, .2], [.1, .9], [np.nan, np.nan]]),
              "covered": np.array([True, True, False]),
              "ties": np.array([False, False, False]), "supported": [True, True]}

    result = MODULE.patient_membership("S0", baseline, mapped)

    assert result.baseline_pop.tolist() == ["Pop1", "Pop2", "Pop3"]
    assert result.patient_id.tolist() == ["p1", "p2", "p3"]
    assert result.community_supported.tolist() == [True, True, False]
    assert result.hard_community.isna().tolist() == [False, False, True]


def test_patient_membership_requires_pop_concept():
    with pytest.raises(ValueError, match="Missing required concept 'pop'"):
        MODULE.patient_membership("S0", pd.DataFrame({"patient_id": ["p1"]}),
                                  {"hard": np.array([0.]), "probabilities": np.ones((1, 1)),
                                   "covered": [True], "ties": [False], "supported": [True]})


def test_run_all_scenarios_with_public_episode_schema(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    n = 24
    master = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "clinical_episode_id": [f"e{i}" for i in range(n)],
        "clinical_anchor_date": pd.Timestamp("2020-01-01"),
        "clinical_visit_number": 1, "clinical_visit": True, "is_clinical_baseline": True,
        "pop__status": ["Pop1", "Pop2", "Pop3"] * (n // 3),
        "essdai__total": rng.uniform(0, 10, n), "esspri__total": rng.uniform(0, 10, n),
        "ovl__status": ["overlap", "glandular_only", "neither"] * (n // 3),
        **{f"pro__{feature}": rng.uniform(1, 10, n) for feature in MODULE.CANON.PRO_FEATURES},
        "lab__albumin__value": rng.uniform(1, 10, n),
        "lab__acute_care_panel__chloride_blood__value": rng.uniform(90, 110, n),
        "lab__anti_ro_ssa__text": ["positive"] * (n // 2) + ["negative"] * (n // 2),
    })
    integrated = tmp_path / "integrated.parquet"
    master.to_parquet(integrated, index=False)
    original_parquet = integrated.read_bytes()
    candidates = tmp_path / "candidates.csv"
    # Exercise the actual Pharma -> Graph boundary, including lab__ in lab keys.
    PHARMA.profile_labs(master).to_csv(candidates, index=False)
    metrics = tmp_path / "baseline.csv"
    master[["patient_id"]].to_csv(metrics, index=False)
    config = MODULE.CANON.load_config(SCRIPT.with_name("config_sensitivity.yaml"))
    config["bootstrap"]["replicates"] = 1
    config["permutation"]["ari_replicates"] = 5
    config["persistent_homology"]["enabled"] = False
    config["mapper"].update(n_cubes=2, min_samples=2, minimum_node_support=2)
    config["community_detection"]["minimum_patients"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    dirs = {name: tmp_path / name for name in ("tables", "logs", "figures")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr(MODULE, "create_study_dirs", lambda _: dirs)

    MODULE.run(MODULE.parse_args([
        "--integrated", str(integrated), "--graph-lab-candidates", str(candidates),
        "--baseline-metrics", str(metrics), "--config", str(config_path)]))

    membership = pd.read_csv(dirs["tables"] / "02_graph_scenario_patient_membership.csv")
    assert len(membership) == n * len(MODULE.SCENARIOS)
    for scenario in MODULE.SCENARIOS:
        assert membership.loc[membership.scenario.eq(scenario), "baseline_pop"].tolist() == master["pop__status"].tolist()
    summary = pd.read_csv(dirs["tables"] / "02_graph_representation_sensitivity_summary.csv")
    assert summary.scenario.tolist() == list(MODULE.SCENARIOS)
    assert summary.n_pro_features.eq(4).all()
    assert summary.n_clinical_features.eq(1).all()
    assert summary.n_lab_features.eq(3).all()
    mapping_audit = pd.read_csv(dirs["tables"] / "02_graph_S4_lab_group_mapping_audit.csv")
    assert mapping_audit.mapped.all()
    assert set(mapping_audit.lab) == {"albumin", "acute_care_panel__chloride_blood", "anti_ro_ssa"}
    assert mapping_audit.feature.str.startswith("lab__").all()
    crosswalk = pd.read_csv(dirs["tables"] / "02_graph_scenario_community_pop_crosswalk.csv")
    assert set(crosswalk.baseline_pop) <= {"Pop1", "Pop2", "Pop3"}
    assert set(crosswalk.scenario) == set(MODULE.SCENARIOS)
    features = pd.read_csv(dirs["tables"] / "02_graph_scenario_feature_sets.csv")
    assert not features.feature.isin(["pop_status", "pop__status", "essdai_total", "essdai__total", "esspri__total"]).any()
    assert len(list(dirs["figures"].glob("*.png"))) == 3
    assert integrated.read_bytes() == original_parquet


def test_strict_coverage_preserves_full_one_hot_and_missing_values():
    values = {
        "category__a": pd.Series([1.0, 0.0, np.nan, 1.0]),
        "category__b": pd.Series([0.0, 1.0, np.nan, 0.0]),
        "continuous": pd.Series([1.0, 2.0, 3.0, 4.0]),
    }
    families = {name: "CLINICAL" for name in values}
    config = {"feature_selection": {"lab_minimum_baseline_coverage": .7,
                                     "pro_minimum_baseline_coverage": .6,
                                     "clinical_minimum_baseline_coverage": .7},
              "redundancy": {"conceptual_pairs": [], "spearman_abs_threshold": .95}}

    selected, audit, _ = MODULE.select_features(
        values, families, "S1", config, pd.DataFrame())

    assert set(selected) == set(values)
    assert selected["category__a"].isna().sum() == 1
    assert selected["category__b"].isna().sum() == 1
    assert audit.included_after_coverage_filter.all()


def _membership(scenario, assignments, supported=None):
    supported = supported or [True] * len(assignments)
    return pd.DataFrame({"scenario": scenario, "patient_id": ["a", "b", "c", "d"],
        "mapper_covered": True, "community_membership_tie": False,
        "hard_community": pd.array(assignments, dtype="Int64"),
        "community_supported": supported})


def test_cross_scenario_overlap_and_ari_use_patient_assignments():
    memberships = {
        "S0": _membership("S0", [0, 0, 1, 1]),
        "S1": _membership("S1", [0, 0, 1, 1]),
        "S2": _membership("S2", [0, 1, 1, 1]),
        "S3": _membership("S3", [0, 0, 1, 1], [True, False, True, True]),
        "S4": _membership("S4", [0, 0, 1, 1]),
    }
    overlap, stable, ari = MODULE.cross_scenario_tables(
        memberships, {scenario: 0 for scenario in memberships})

    s0_s1 = stable.query("scenario_a == 'S0' and scenario_b == 'S1'").iloc[0]
    assert s0_s1.jaccard == 1.0
    assert s0_s1.overlap_coefficient == 1.0
    assert len(overlap) == 40
    assert ari.query("scenario_a == 'S0' and scenario_b == 'S1'").iloc[0].ari == 1.0
    assert ari.query("scenario_a == 'S0' and scenario_b == 'S3'").iloc[0].n_patients_used == 3
    assert len(ari) == 10
    assert len(stable) == 10
    assert len(ari.query("scenario_a == 'S3' and scenario_b == 'S4'")) == 1


def test_s4_hierarchical_balance_preserves_features_and_audits_weights():
    raw = pd.DataFrame({
        "lab_a__value": [1.0, 2.0, np.nan, 4.0, 5.0],
        "lab_b__value": [5.0, 2.0, 3.0, 1.0, 4.0],
        "lab_c__binary": [0.0, 1.0, 0.0, 1.0, 1.0],
        "pro_score": [2.0, 4.0, 1.0, 5.0, 3.0],
        "age": [10.0, 15.0, 20.0, 25.0, 30.0],
    })
    families = {"lab_a__value": "LAB", "lab_b__value": "LAB",
                "lab_c__binary": "LAB", "pro_score": "PRO", "age": "CLINICAL"}
    transformed, names, audit = MODULE.prepare_matrix(
        raw, families, balance=True, lab_group_balance=True,
        lab_group_by_lab={"lab_a": "HEMATOLOGY", "lab_b": "HEMATOLOGY",
                          "lab_c": "COAGULATION"},
        feature_to_lab={"lab_a__value": "lab_a", "lab_b__value": "lab_b",
                        "lab_c__binary": "lab_c"}, scenario="S4")

    assert set(names) == set(raw)
    assert transformed.shape == raw.shape
    assert set(audit.level) == {"LAB_GROUP", "FAMILY"}
    assert set(audit.query("level == 'LAB_GROUP'").block) == {"HEMATOLOGY", "COAGULATION"}
    assert (audit.sigma1_before_weighting > 0).all()
    assert np.isfinite(audit.weight_applied).all()


def test_lab_group_map_validation_and_retained_mapping_audit(tmp_path):
    path = tmp_path / "groups.csv"
    pd.DataFrame({"lab": [" Lab_A ", "lab_b"], "s4_group": [" HEM ", "COAG"],
                  "s4_subgroup": ["CBC", "PT"],
                  "mapping_status": ["MAPPED", "UNMAPPED"]}).to_csv(path, index=False)
    mapping = MODULE.load_lab_group_map(path)

    assert mapping.lab.tolist() == ["lab_a", "lab_b"]
    assert MODULE.lab_group_lookup(mapping) == {"lab_a": "HEM"}
    values = {"lab_a__value": pd.Series([1.0]), "lab_b__value": pd.Series([2.0])}
    families = {name: "LAB" for name in values}
    feature_to_lab = {"lab_a__value": "lab_a", "lab_b__value": "lab_b"}
    try:
        MODULE.validate_s4_mapping(
            values, families, mapping, feature_to_lab, included=set(values))
    except ValueError as exc:
        assert "lab_b__value" in str(exc)
    else:
        raise AssertionError("An included UNMAPPED lab must be rejected")


def test_group_map_normalizes_public_namespace_and_rejects_collisions(tmp_path):
    path = tmp_path / "groups.csv"
    pd.DataFrame({"lab": [" LAB__Albumin "], "s4_group": ["LIVER_PANCREAS"],
                  "s4_subgroup": ["PROTEINS"], "mapping_status": ["MAPPED"]}).to_csv(path, index=False)
    assert MODULE.load_lab_group_map(path).lab.tolist() == ["albumin"]
    pd.DataFrame({"lab": ["albumin", "lab__albumin"], "s4_group": ["LIVER_PANCREAS"] * 2,
                  "s4_subgroup": ["PROTEINS"] * 2, "mapping_status": ["MAPPED"] * 2}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate lab values"):
        MODULE.load_lab_group_map(path)


def test_default_group_map_covers_labs_reported_by_s4():
    # Regression fixture from the real S4 failure; compound stems stay intact.
    labs = """
        acute_care_panel__chloride_blood acute_care_panel__co2_total_blood
        acute_care_panel__potassium_blood acute_care_panel__sodium_blood
        albumin alkaline_phosphatase alpha1_globulin alpha2_globulin alt amylase anc
        anti_la_ssb anti_rnp anti_ro_ssa anti_scl70 anti_sm ast basophil_absolute
        basophil_percent beta1_globulin beta2_globulin bilirubin_total bun complement_c3
        complement_c4 creatine_kinase creatinine eosinophil_absolute eosinophil_percent
        esr estimated_average_glucose ferritin gamma_globulin glucose hdl_cholesterol
        hematocrit hemoglobin hepatitis_b_core_antibody hepatitis_c_antibody iga igg igm
        immature_granulocyte_absolute immature_granulocyte_percent
        immunofixation_electrophoresis_serum__albumin_ord_electrophoresis_pro_or_ife_blood
        iron_saturation lactate_dehydrogenase ldl_cholesterol_calculated lymphocyte_count
        lymphocyte_percent mch mchc mcv monocyte_absolute monocyte_percent mpv
        neutrophil_percent phosphorus platelet_count protein_total protein_total_electrophoresis
        rbc rdw serum_iron total_cholesterol transferrin triglycerides uric_acid
        urine_nitrite urine_squamous_cells wbc
    """.split()
    mapping = MODULE.load_lab_group_map(MODULE.DEFAULT_LAB_GROUP_MAP)
    lookup = MODULE.lab_group_lookup(mapping)
    assert set(labs) <= set(lookup)
    values = {f"lab__{lab}__value": pd.Series([1., 2.]) for lab in labs}
    audit = MODULE.validate_s4_mapping(
        values, {feature: "LAB" for feature in values}, mapping,
        {f"lab__{lab}__value": f"lab__{lab}" for lab in labs}, included=set(values))
    assert len(audit) == len(labs)
    assert audit.mapped.all()


@pytest.mark.parametrize("feature, expected", [
    ("lab__hemoglobin_a1c__value", "hemoglobin_a1c"),
    ("lab__anti_ro_ssa__binary", "anti_ro_ssa"),
    ("lab__ana__cat_positive", "ana"),
    ("lab__acute_care_panel__chloride_blood__value", "acute_care_panel__chloride_blood"),
    ("acute_care_panel__chloride_blood__value", "acute_care_panel__chloride_blood"),
])
def test_feature_stem_preserves_compound_lab_concept(feature, expected):
    assert MODULE.feature_stem(feature) == expected


def test_configured_redundancy_pair_matches_public_lab_features():
    values = {"lab__hemoglobin_a1c__value": pd.Series([1., 2., 3., 4.]),
              "lab__estimated_average_glucose__value": pd.Series([3., 1., 4., 2.])}
    config = {"redundancy": {"conceptual_pairs": [["hemoglobin_a1c", "estimated_average_glucose"]],
                              "spearman_abs_threshold": .95}}

    selected, audit = MODULE.reduce_redundancy(values, config)

    assert set(selected) == {"lab__hemoglobin_a1c__value"}
    assert audit.reason.str.startswith("configured_conceptual_pair").all()


def test_s4_uses_manifest_metadata_for_compound_lab_names():
    feature = "acute_care_panel__chloride_blood__value"
    lab = "acute_care_panel__chloride_blood"
    raw = pd.DataFrame({feature: [100.0, 101.0, 102.0],
                        "pro_score": [1.0, 3.0, 2.0]})
    families = {feature: "LAB", "pro_score": "PRO"}

    transformed, names, audit = MODULE.prepare_matrix(
        raw, families, balance=True, lab_group_balance=True,
        lab_group_by_lab={lab: "METABOLIC_ENDOCRINE"},
        feature_to_lab={feature: lab}, scenario="S4")

    assert names == [feature, "pro_score"]
    assert transformed.shape == raw.shape
    assert audit.query("level == 'LAB_GROUP'").block.tolist() == ["METABOLIC_ENDOCRINE"]


def test_s4_rejects_lab_feature_absent_from_manifest_metadata():
    values = {"lab_a__value": pd.Series([1.0])}
    families = {"lab_a__value": "LAB"}
    mapping = pd.DataFrame({"lab": ["lab_a"], "s4_group": ["HEM"],
                            "s4_subgroup": ["CBC"], "mapping_status": ["MAPPED"]})

    try:
        MODULE.validate_s4_mapping(values, families, mapping, {}, included=set(values))
    except ValueError as exc:
        assert "missing canonical lab metadata" in str(exc)
    else:
        raise AssertionError("Missing manifest metadata must be rejected")
