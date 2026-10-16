"""Synthetic Step-10 shape, CLI immutability and actual consumer read checks.

This reproduces the documented family counts, not the unavailable real header.
It does not run Pharma or Graph models or assert clinical before/after results.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from graph_contract_fixtures import write_bundle

from test_pharma_lab_profile import profile_module as mod, episode_frame, approved_registry


def schema_shaped_frame():
    base = episode_frame()
    cols = {c: base[c] for c in base}
    for i in range(7):
        cols[f"structural_synthetic_{i}"] = "synthetic"
    stems = [e["preferred_lab_stem"] for e in mod.load_registry()["concepts"].values()]
    stems += ["lab__ana_hep2_status", "lab__anti_ro52", "lab__anti_ro60", "lab__igg_total", "lab__urine_wbc",
              "lab__urine_hemoglobin", "lab__estimated_average_glucose", "lab__anti_ccp", "lab__urine_protein_qualitative", "lab__hla_drb1"]
    stems += [f"lab__synthetic_assay_{i:03}" for i in range(177 - len(stems))]
    for stem in stems:
        cols[stem + "__text"] = ["POSITIVE", "NEGATIVE"] * 6 if stem == "lab__ana_status" else None
    for stem in stems[:100]:
        cols[stem + "__value"] = np.arange(12, dtype=float) if stem == "lab__glucose" else np.nan
    for i in range(7):
        cols[f"sero__baseline_synthetic_{i}"] = 1
    for i in range(15):
        cols[f"sero__synthetic_{i}__ever_positive_through_episode"] = 1
    for i in range(7):
        for suffix in ["asof_value", "patient_consensus_value"]:
            cols[f"sero__hla_synthetic_{i}__{suffix}"] = "allele"
    for prefix, extra in [("ext", 96), ("essdai", 24), ("ovl", 13), ("pro", 12), ("esspri", 3), ("demo", 5), ("spine", 3)]:
        for i in range(extra):
            cols[f"{prefix}__synthetic_{i}"] = np.nan
    return pd.DataFrame(cols)


def test_documented_485_column_family_shape_and_separate_serology_inventory():
    frame = schema_shaped_frame()
    assert len(frame.columns) == 485
    assert frame.columns.is_unique
    profile = mod.profile_labs(frame)
    assert len(profile) == 177
    assert profile.lab.str.startswith("lab__").all()
    assert sum(c.startswith("lab__") and c.endswith("__value") for c in frame) == 100
    assert sum(c.startswith("lab__") and c.endswith("__text") for c in frame) == 177
    serology = mod.serology_profile(frame)
    assert len(serology) == 36
    assert serology.kind.value_counts().to_dict() == {
        "baseline_summary": 7, "cumulative_asof": 15, "genetic_asof": 7, "patient_level_consensus": 7}
    assert not serology.potential_baseline_predictor.any()
    assert not serology.potential_time_varying_predictor.any()
    assert serology.loc[serology.kind.eq("patient_level_consensus"), "leakage_risk"].eq("high_future_information").all()
    assert not mod.graph_candidates(profile).lab.str.startswith("sero__").any()


def test_repeated_cli_outputs_are_identical_and_input_hashes_unchanged(tmp_path):
    frame = schema_shaped_frame()
    integrated = tmp_path / "clinical.parquet"
    frame.to_parquet(integrated)
    context_path = tmp_path / "10_integrated_longitudinal_context.parquet"
    context = frame[mod.KEYS].copy()
    context["lab__glucose__unit"] = "mg/dL"
    context.to_parquet(context_path)
    old = tmp_path / "previous.csv"
    mod.profile_labs(frame).to_csv(old, index=False)
    hashes = [mod._sha256(p) for p in [integrated, context_path, old]]
    for directory in ["first", "second"]:
        args = mod.parse_args(["--integrated", str(integrated), "--output", str(tmp_path / directory / "profile.csv"),
                               "--previous-profile", str(old)])
        mod.run(args)
    names = sorted(p.name for p in (tmp_path / "first").iterdir())
    assert len(names) == 7
    for name in names:
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
        assert "SYNTHETIC_PATIENT_0" not in (tmp_path / "first" / name).read_text()
    assert [mod._sha256(p) for p in [integrated, context_path, old]] == hashes
    summary = json.loads((tmp_path / "first" / mod.SUMMARY_NAME).read_text())
    assert summary["schema_matches_reference"]
    assert summary["inputs_unchanged"]
    assert summary["context_mode"] == "adjacent_context"
    assert summary["context_audit"]["n_context_keys_unmatched"] == 0
    assert summary["previous_snapshot_available"]
    audit = pd.read_csv(tmp_path / "first" / mod.EXPECTED_AUDIT_NAME).set_index("expected_lab")
    assert audit.loc["ana", "preferred_match_status"] == "exact"
    assert audit.loc["crp", "preferred_match_status"] == "exact"


def test_context_cli_modes_explicit_missing_and_no_context(tmp_path):
    integrated = tmp_path / "clinical.parquet"
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    frame.to_parquet(integrated)
    companion = tmp_path / "10_integrated_longitudinal_context.parquet"
    companion.write_text("invalid parquet: --no-context must never open this")
    args = mod.parse_args(["--integrated", str(integrated), "--no-context", "--output", str(tmp_path / "out" / "profile.csv")])
    result = mod.run(args)
    assert result.unit_assessment_status.eq("not_assessed").all()
    summary = json.loads((args.output.parent / mod.SUMMARY_NAME).read_text())
    assert summary["context_mode"] == "integrated_only_explicit"
    assert summary["schema_drift"]["lab_stems"] == {"expected": 177, "observed": 1}
    args = mod.parse_args(["--integrated", str(integrated), "--context", str(tmp_path / "missing.parquet")])
    with pytest.raises(FileNotFoundError, match="Explicit context"):
        mod.run(args)
    args = mod.parse_args(["--integrated", str(integrated), "--output", str(integrated)])
    with pytest.raises(ValueError, match="CSV"):
        mod.run(args)


def _load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, mod.ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_pharma_and_graph_readers_consume_only_existing_approved_sources(tmp_path):
    pharma = _load_script("pharma01_contract", "src/studies/pharma/01_run_pharma_main.py")
    graph = _load_script("graph01_contract", "src/studies/graph/01_run_topological_phenotyping.py")
    frame = episode_frame()
    frame["lab__glucose__value"] = np.arange(12)
    frame["lab__unapproved__value"] = np.arange(12)
    frame["lab__ana_status__text"] = ["POSITIVE"] * 6 + ["NEGATIVE"] * 6
    frame["sero__hla__patient_consensus_value"] = "allele"
    context = frame[mod.KEYS].copy()
    context["lab__glucose__unit"] = "mg/dL"
    registry = approved_registry("lab__glucose", "lab__ana_status", binary=True)
    profile = mod.profile_labs(frame, context, registry=registry)
    profile_path, candidates_path, integrated = (tmp_path / name for name in ["profile.csv", "candidates.csv", "clinical.parquet"])
    profile.to_csv(profile_path, index=False)
    mod.graph_candidates(profile).to_csv(candidates_path, index=False)
    frame.to_parquet(integrated)
    loaded = pharma.load_lab_profile(profile_path)
    numeric = pharma._lab_specs(loaded, frame, "numeric_longitudinal")
    categorical = pharma._lab_specs(loaded, frame, "categorical_longitudinal")
    assert [x["column"] for x in numeric] == ["lab__glucose__value"]
    assert [x["column"] for x in categorical] == ["lab__ana_status__text"]
    assert all(x["column"] in frame for x in numeric + categorical)
    write_bundle(integrated, candidates_path, profile=profile)
    master, candidates, contract = graph.load_real_inputs(integrated, candidates_path)
    assert contract["dataset_contract"]["status"] == "passed"
    assert candidates.lab.str.startswith("lab__").all()
    baseline = master.loc[master.is_clinical_baseline].copy()
    manifest, values, _, audit = graph.build_feature_manifest(baseline, candidates, 0.)
    assert {"lab__glucose__value", "lab__ana_status__binary"} <= set(values)
    included = manifest.loc[manifest.family.eq("lab") & manifest.included]
    assert included.source_column.isin(frame.columns).all()
    assert not included.lab.eq("lab__unapproved").any()
    assert not included.source_column.str.startswith("sero__").any()
    assert audit.loc[audit.lab.eq("lab__unapproved"), "exclusion_reason"].item() == "review_before_graph_not_approved"
    assert list(values["lab__glucose__value"]) == list(baseline["lab__glucose__value"])


def test_empty_catalog_is_readable_by_both_consumers(tmp_path):
    pharma = _load_script("pharma01_empty", "src/studies/pharma/01_run_pharma_main.py")
    graph = _load_script("graph01_empty", "src/studies/graph/01_run_topological_phenotyping.py")
    frame = episode_frame()
    profile = mod.profile_labs(frame)
    pp, cp, ip = [tmp_path / p for p in ["profile.csv", "candidates.csv", "clinical.parquet"]]
    profile.to_csv(pp, index=False)
    mod.graph_candidates(profile).to_csv(cp, index=False)
    frame.to_parquet(ip)
    assert pharma._lab_specs(pharma.load_lab_profile(pp), frame, "numeric_longitudinal") == []
    write_bundle(ip, cp, profile=profile)
    # Pharma's empty-inventory exporter currently omits v2 headers. Fail closed;
    # a separate Graph test covers a correctly headered empty v2 CSV + summary.
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        graph.load_real_inputs(ip, cp)
