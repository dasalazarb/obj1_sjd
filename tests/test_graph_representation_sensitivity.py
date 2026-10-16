"""Focused tests for representation-sensitivity comparison contracts."""
import importlib.util
import json
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


def _profile_with_synthetic_followups_and_approvals(baseline):
    """Keep Graph scenario fixtures eligible under Pharma's explicit new gates.

    These are test-only repeated observations and approvals, never production
    clinical decisions. The Graph input remains the original baseline fixture.
    """
    first = baseline.copy()
    if "clinical_episode_id" not in first:
        first["clinical_episode_id"] = [f"baseline_{i}" for i in range(len(first))]
    if "clinical_anchor_date" not in first:
        first["clinical_anchor_date"] = pd.Timestamp("2020-01-01")
    followup = first.copy()
    followup["clinical_episode_id"] = "synthetic_followup_" + followup.clinical_episode_id.astype(str)
    followup["clinical_anchor_date"] = pd.to_datetime(followup.clinical_anchor_date).astype("datetime64[ns]") + pd.Timedelta(days=365)
    followup["is_clinical_baseline"] = False
    repeated = pd.concat([first, followup], ignore_index=True)
    registry = PHARMA.load_registry()
    for column in repeated:
        lab = PHARMA.lab_stem(column)
        if lab:
            registry["assays"][lab] = {
                "approval_status": "approved", "decision_id": "SYNTHETIC_TEST_ONLY",
                "reviewer": "synthetic_fixture", "evidence": "test_only_not_clinical_approval",
                "numeric_approved": True, "binary_approved": True, "allow_unassessed_units": True}
    return PHARMA.profile_labs(repeated, registry=registry)


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


@pytest.mark.parametrize("mode", ["healthy", "invalid_bootstrap", "unmapped"])
def test_run_all_scenarios_with_public_episode_schema(tmp_path, monkeypatch, mode):
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
    if mode == "unmapped":
        # Eligible in a subset, but below the full-sample S4 coverage threshold.
        master["lab__unapproved__value"] = rng.normal(size=n)
    integrated = tmp_path / "integrated.parquet"
    master.to_parquet(integrated, index=False)
    original_parquet = integrated.read_bytes()
    candidates = tmp_path / "candidates.csv"
    # Exercise the actual Pharma -> Graph boundary, including lab__ in lab keys.
    PHARMA.run(PHARMA.parse_args(["--integrated", str(integrated), "--output", str(tmp_path / "profile.csv")]))
    candidates = tmp_path / PHARMA.GRAPH_CANDIDATES_NAME
    metrics = tmp_path / "baseline.csv"
    master[["patient_id"]].to_csv(metrics, index=False)
    config = MODULE.CANON.load_config(SCRIPT.with_name("config_sensitivity.yaml"))
    config["bootstrap"]["replicates"] = 0 if mode == "invalid_bootstrap" else 1
    config["permutation"]["ari_replicates"] = 5
    config["persistent_homology"]["enabled"] = False
    config["mapper"].update(n_cubes=2, min_samples=2, minimum_node_support=2)
    config["community_detection"]["minimum_patients"] = 2
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    canonical = MODULE.CANON.load_config(SCRIPT.with_name("config.yaml"))
    for key in ("mapper", "community_detection", "random_seed"):
        canonical[key] = config[key]
    canonical_path = tmp_path / "canonical.yaml"
    canonical_path.write_text(yaml.safe_dump(canonical))
    dirs = {name: tmp_path / name for name in ("tables", "logs", "figures")}
    for path in dirs.values():
        path.mkdir()
    monkeypatch.setattr(MODULE.CANON, "run_dirs", lambda *a: dirs)

    args = MODULE.parse_args([
        "--integrated", str(integrated), "--graph-lab-candidates", str(candidates),
        "--baseline-metrics", str(metrics), "--config", str(config_path),
        "--feature-policy", "legacy_with_overlap", "--canonical-config", str(canonical_path)])
    if mode == "unmapped":
        with pytest.raises(MODULE.MappingError, match="unapproved"):
            MODULE.run(args)
        audit = pd.read_csv(dirs["tables"] / "02_graph_S4_lab_group_mapping_audit.csv")
        missing = audit.loc[audit.lab.eq("unapproved")].iloc[0]
        assert missing.eligible_for_bootstrap and missing.included_S4
        assert missing.mapping_validation_result == "MAPPING_ERROR"
        assert (dirs["tables"] / "02_graph_S0_pca_variance.csv").exists()
        completed = pd.read_csv(dirs["tables"] / "02_graph_representation_sensitivity_summary.csv")
        assert completed.scenario.tolist() == ["S0", "S1", "S2", "S3"]
        return
    MODULE.run(args)

    membership = pd.read_csv(dirs["tables"] / "02_graph_scenario_patient_membership.csv")
    assert len(membership) == n * len(MODULE.SCENARIOS)
    for scenario in MODULE.SCENARIOS:
        assert membership.loc[membership.scenario.eq(scenario), "baseline_pop"].tolist() == master["pop__status"].tolist()
    summary = pd.read_csv(dirs["tables"] / "02_graph_representation_sensitivity_summary.csv")
    assert summary.scenario.tolist() == list(MODULE.SCENARIOS)
    assert summary.n_pro_features.eq(4).all()
    assert summary.n_clinical_features.eq(1).all()
    assert summary.n_lab_features.eq(3).all()
    qc = pd.read_csv(dirs["tables"] / "02_graph_bootstrap_qc_summary.csv")
    replicates = pd.read_csv(dirs["tables"] / "02_graph_bootstrap_replicate_audit.csv")
    stability = pd.read_csv(dirs["tables"] / "02_graph_scenario_community_stability.csv")
    assert qc.n_requested.eq(qc.n_success + qc.n_failed).all()
    assert len(replicates) == 5 * config["bootstrap"]["replicates"]
    assert {"matched_successes", "stability_conditional", "stability_conservative", "bootstrap_qc_status"} <= set(stability)
    if mode == "invalid_bootstrap":
        assert summary.n_stable_communities.isna().all()
        assert summary.bootstrap_qc_status.eq("INVALID_BOOTSTRAP").all()
        assert stability.bootstrap_stability.isna().all()
    else:
        assert qc.bootstrap_qc_status.eq("BOOTSTRAP_QC_PASS").all()
    report = json.loads((dirs["tables"] / "02_graph_interpretation_qc.json").read_text())
    assert report["GLOBAL_PARTITION_ROBUSTNESS"] == "NOT_EVALUABLE"
    assert (dirs["tables"] / "02_graph_common_intersection_ari.csv").exists()
    assert (dirs["tables"] / "02_graph_stable_core_node_union_overlap.csv").exists()
    assert (dirs["tables"] / "02_graph_bootstrap_balance_audit.csv").exists()
    mapping_audit = pd.read_csv(dirs["tables"] / "02_graph_S4_lab_group_mapping_audit.csv")
    assert mapping_audit.mapped.all()
    assert set(mapping_audit.lab) == {"albumin", "acute_care_panel__chloride_blood", "anti_ro_ssa"}
    assert mapping_audit.feature.str.startswith("lab__").all()
    crosswalk = pd.read_csv(dirs["tables"] / "02_graph_scenario_community_pop_crosswalk.csv")
    assert set(crosswalk.baseline_pop) <= {"Pop1", "Pop2", "Pop3"}
    assert set(crosswalk.scenario) == set(MODULE.SCENARIOS)
    features = pd.read_csv(dirs["tables"] / "02_graph_scenario_feature_sets.csv")
    assert not features.loc[features.included, "feature"].isin(["pop_status", "pop__status", "essdai_total", "essdai__total", "esspri__total"]).any()
    assert len(list(dirs["figures"].glob("*.png"))) == 3
    assert integrated.read_bytes() == original_parquet
    selection = pd.read_csv(dirs["tables"] / "02_graph_feature_selection_audit.csv")
    assert set(selection.scenario) == set(MODULE.SCENARIOS)
    assert selection.loc[selection.family.eq("LAB"), "present_in_integrated_parquet"].all()
    assert not selection.loc[selection.family.eq("LAB"), "approved_by_pharma00"].any()
    assert selection.loc[selection.feature.eq("age_dx"), "present_in_pca"].eq(False).all()
    pd.testing.assert_frame_equal(
        features[["scenario", "feature", "included"]].sort_values(["scenario", "feature"]).reset_index(drop=True),
        selection[["scenario", "feature", "present_in_pca"]]
        .rename(columns={"present_in_pca": "included"})
        .sort_values(["scenario", "feature"]).reset_index(drop=True))


def test_five_audited_mappings_preserve_selection_stages():
    rng = np.random.default_rng(29)
    labs = ["acute_care_panel__anion_gap_blood", *[
        f"urinalysis_includes_microscopic__{name}_urinalysis" for name in
        ("hyalin_casts", "renal_epithelial_cells", "transitional_epith", "unclassified_casts")]]
    master = pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(24)], "is_clinical_baseline": True,
        **{f"lab__{lab}__value": rng.normal(size=24) for lab in labs},
        "lab__uncatalogued__value": rng.normal(size=24),
        "pro__sf36_pcs": rng.normal(size=24), "pro__sf36_mcs": rng.normal(size=24),
    })
    master[f"lab__{labs[1]}__value"] = [1., 2., 3., 4., 5.] + [np.nan] * 19
    master[f"lab__{labs[2]}__value"] = master[f"lab__{labs[0]}__value"]
    master[f"lab__{labs[4]}__value"] = 1.
    candidates = _profile_with_synthetic_followups_and_approvals(master)
    candidates = candidates.loc[~candidates.lab.eq("lab__uncatalogued")].copy()
    candidates.loc[candidates.lab.eq(f"lab__{labs[3]}"), ["graph_candidate_status", "graph_encoding_hint"]] = ["review_before_graph", "review"]
    # An approved catalog entry can lose variability in a later baseline extract.
    candidates.loc[candidates.lab.eq(f"lab__{labs[4]}"),
                   ["graph_candidate_status", "graph_encoding_hint", "primary_result_source"]] = [
                       "include_numeric", "continuous", "value_column"]
    candidates.loc[candidates.lab.eq(f"lab__{labs[4]}"),
        ["clinical_review_required", "clinically_approved", "primary_result_source", "source_column"]] = [
            False, "approved", "value_numeric", f"lab__{labs[4]}__value"]
    manifest, values, _, _ = MODULE.CANON._build_legacy_feature_manifest(master, candidates, 0.0)
    families = {row.feature: MODULE.family_name(row.family)
                for row in manifest.loc[manifest.included].itertuples()}
    feature_to_lab = {row.feature: MODULE.canonical_lab_name(row.lab)
                      for row in manifest.loc[manifest.family.eq("lab") & manifest.included].itertuples()}
    mapping = MODULE.load_lab_group_map(MODULE.DEFAULT_LAB_GROUP_MAP)
    lookup = MODULE.lab_group_lookup(mapping)
    indexed = mapping.set_index("lab")
    assert indexed.loc[labs, "mapping_status"].eq("MAPPED").all()
    assert indexed.loc[labs[0], ["s4_group", "s4_subgroup"]].tolist() == ["METABOLIC_ENDOCRINE", "ELECTROLYTES"]
    assert indexed.loc[labs[1:], "s4_group"].eq("RENAL_URINARY").all()
    assert indexed.loc[labs[1:], "s4_subgroup"].eq("URINALYSIS").all()
    preflight = MODULE.validate_s4_mapping(values, families, mapping, feature_to_lab, require_all=True)
    assert set(preflight.lab) == set(labs[:3])
    assert preflight.mapped.all()
    config = MODULE.CANON.load_config(SCRIPT.with_name("config_sensitivity.yaml"))
    audits = {}
    for scenario in ("S3", "S4"):
        selected, coverage, redundancy = MODULE.select_features(values, families, scenario, config, manifest)
        rep = MODULE.fit_embedding(pd.DataFrame(selected), families, scenario, config,
                                   lab_group_by_lab=lookup, feature_to_lab=feature_to_lab)
        audits[scenario] = MODULE.feature_selection_audit(
            master, master, candidates, manifest, values, coverage, redundancy, rep, mapping, scenario)
    audit = audits["S4"].set_index("feature")
    assert audit.loc[f"lab__{labs[0]}__value", "present_in_pca"]
    sparse = audit.loc[f"lab__{labs[1]}__value"]
    assert sparse.encoded_in_expanded_manifest and sparse.eligible_in_scenario and sparse.mapped_s4
    assert not sparse.passed_coverage_filter and not sparse.present_in_pca
    redundant = audit.loc[f"lab__{labs[2]}__value"]
    assert redundant.passed_coverage_filter and not redundant.selected_after_redundancy
    assert redundant.exclusion_reason == "removed_by_redundancy"
    review = audit.loc[f"lab__{labs[3]}__value"]
    assert review.catalogued_by_pharma00 and not review.approved_by_pharma00 and review.mapped_s4
    assert not review.eligible_in_scenario and not review.present_in_pca
    constant = audit.loc[f"lab__{labs[4]}__value"]
    assert constant.approved_by_pharma00 and not constant.encoded_in_expanded_manifest
    assert constant.exclusion_reason == "no_baseline_variability"
    uncatalogued = audit.loc["lab__uncatalogued__value"]
    assert uncatalogued.present_in_integrated_parquet and not uncatalogued.catalogued_by_pharma00
    assert not uncatalogued.present_in_pca
    assert uncatalogued.exclusion_reason == "not_catalogued_by_pharma00"
    assert audit.present_in_pca.equals(audits["S3"].set_index("feature").present_in_pca)


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


def test_default_group_map_covers_either_member_of_configured_redundancy_pairs():
    # Selection can retain either member depending on its baseline coverage.
    config = MODULE.CANON.load_config(SCRIPT.with_name("config_sensitivity.yaml"))
    mapping = MODULE.load_lab_group_map(MODULE.DEFAULT_LAB_GROUP_MAP)
    lookup = MODULE.lab_group_lookup(mapping)
    for pair in config["redundancy"]["conceptual_pairs"]:
        assert set(pair) <= set(lookup), f"S4 lacks mapping for configured alternatives: {pair}"


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
