"""Synthetic software validation only; no real-cohort assertions or outputs."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from matplotlib.collections import LineCollection

from src.block_A import observed_patterns_v2 as v2
from src.block_A import secondary_trajectory_analysis as analysis
from src.block_A.secondary_trajectory_catalog import build_catalog
from src.block_A.secondary_trajectory_figures import build_figure
from src.block_A.observed_patterns_v2_figures import COLORS, LABELS

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/"src/block_A"/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = load("secondary_step15", "15_secondary_trajectories_by_essdai_pattern.py")
step13 = load("secondary_step13_fixture", "13_disease_activity_progression.py")


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-upstream")
    frames = []
    for i, scores in enumerate(([2, 4, 3], [2, 5, 7], [8, 5, 3], [2, 6, 2], [2, 5])):
        times = pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"][:len(scores)])
        frame = pd.DataFrame(dict(patient_id=f"synthetic_{i:03d}", clinical_episode_id=[f"e{j:03d}" for j in range(len(scores))],
            clinical_anchor_date=times, clinical_baseline_date=times[0], clinical_baseline_episode_id="e000",
            is_clinical_baseline=[True]+[False]*(len(scores)-1), clinical_visit=True,
            clinical_visit_number=np.arange(1, len(scores)+1), time_since_clinical_baseline_days=(times-times[0]).days,
            time_since_clinical_baseline_years=(times-times[0]).days/365.25,
            **{"essdai__total": scores, "esspri__total": [10, 8, 6][:len(scores)],
               "lab__complement_c3__value": [np.nan, 80, 60][:len(scores)],
               "ext__ocular_staining_positive": [True, pd.NA, False][:len(scores)],
               "ovl__status": ["overlap", "glandular_only", "neither"][:len(scores)],
               "sero__anti_ro_ssa__ever_positive_through_episode": [pd.NA, False, True][:len(scores)]}))
        for domain in step13.DOMAINS:
            frame[step13.DOMAIN_ALIASES[domain][0]] = 0
        frames.append(frame)
    x = pd.concat(frames, ignore_index=True)
    x["ext__ocular_staining_positive"] = x["ext__ocular_staining_positive"].astype("boolean")
    x["sero__anti_ro_ssa__ever_positive_through_episode"] = x["sero__anti_ro_ssa__ever_positive_through_episode"].astype("boolean")
    b = x.loc[x.is_clinical_baseline].copy()
    inputs = root/"inputs"
    inputs.mkdir()
    integrated, baseline = inputs/"step10.parquet", inputs/"step11.parquet"
    x.to_parquet(integrated, index=False)
    b.to_parquet(baseline, index=False)
    context = x[analysis.KEYS+["clinical_anchor_date"]].copy()
    context["lab__complement_c3__measurement_date"] = x.clinical_anchor_date.where(x["lab__complement_c3__value"].notna())
    context["lab__complement_c3__unit"] = "mg/dL"
    context["source_protocol"] = ["A" if i % 2 else "B" for i in range(len(context))]
    context.to_parquet(inputs/"10_integrated_longitudinal_context.parquet", index=False)
    endpoints = step13.build_patient_absolute_change_endpoints(x)
    v2.write_observed_pattern_outputs_v2(x, b, endpoints, root, input_origin="synthetic_validation",
                                        input_paths={"step10": integrated, "step11": baseline})
    dirs = v2.output_directories(root)
    patterns = dirs["analytic"]/"13_essdai_patient_observed_patterns_v2.parquet"
    manifest = dirs["qc"]/"13_essdai_observed_pattern_manifest_v2.json"
    return dict(root=root, integrated=integrated, baseline=baseline, patterns=patterns, manifest=manifest,
                x=x, b=b, context=context, p=pd.read_parquet(patterns),
                e=pd.read_parquet(dirs["analytic"]/"13_essdai_pattern_patient_eligibility_v2.parquet"),
                s=pd.read_parquet(dirs["analytic"]/"13_essdai_pattern_selected_dates_v2.parquet"))


def args_for(upstream, output, *extra):
    return cli.parse_args(["--integrated", str(upstream["integrated"]), "--baseline", str(upstream["baseline"]),
        "--patterns", str(upstream["patterns"]), "--patterns-manifest", str(upstream["manifest"]),
        "--output-root", str(output), "--input-origin", "synthetic_validation", "--dpi", "60", "--data-cutoff", "2025-01-01", *extra])


def measure(upstream, variable="esspri_total", x=None, groups=None, **kwargs):
    x = analysis.attach_context(upstream["x"], upstream["context"]) if x is None else x
    specs, _ = build_catalog(x)
    spec = next(v for v in specs.values() if v.variable_id == variable)
    groups = analysis.fixed_groups(upstream["p"], 3) if groups is None else groups
    return analysis.observations(x, groups, spec, "2025-01-01", **kwargs), spec, groups


def test_real_producer_contract_and_fixed_groups(upstream):
    manifest, _ = analysis.verified_manifest(upstream["manifest"], upstream["patterns"], upstream["integrated"], upstream["baseline"], "synthetic_validation")
    assert manifest["qc_status"] == "PASS"
    _, _, p = analysis.validate_inputs(upstream["x"], upstream["b"], upstream["p"], upstream["e"], upstream["s"], "2025-01-01")
    (frame, status, _, _), _, groups = measure(upstream)
    assert status == "plot_ready"
    for pid, g in frame.groupby("patient_id"):
        assert g.observed_pattern.nunique() == 1
        assert g.observed_pattern.iloc[0] == groups.set_index("patient_id").loc[pid, "observed_pattern"]
        assert g.delta.tolist() == [0, -2, -4][:len(g)]
    assert set(p.observed_pattern) == set(LABELS)
    x = analysis.attach_context(upstream["x"], upstream["context"])
    x["sero__anti_ro_ssa__ever_positive_through_episode"] = True
    (ssa, _, _, _), _, _ = measure(upstream, "anti_ro_ssa", x=x)
    assert ssa.groupby("patient_id").observed_pattern.first().to_dict() == groups.set_index("patient_id").observed_pattern.to_dict()


def test_threshold_bundles_are_independent(upstream, tmp_path):
    report = cli.run(args_for(upstream, tmp_path/"output", "--threshold", "both", "--dry-run"))
    assert report["groups"]["3"] != report["groups"]["5"]
    assert sum(report["groups"]["3"].values()) == sum(report["groups"]["5"].values()) == 5
    assert not (tmp_path/"output").exists()


def test_delayed_reference_and_two_clocks(upstream):
    (frame, _, _, unit), _, groups = measure(upstream, "complement_c3")
    assert unit == "mg/dL"
    repeated = frame.loc[frame.patient_id.eq("synthetic_000")]
    assert repeated.delta.tolist() == [0, -20]
    assert repeated.time.iloc[0] == pytest.approx(2, abs=.01)
    assert repeated.reference_source.eq("first_observed_after_clinical_baseline").all()
    groups["essdai_index_date"] = pd.Timestamp("2023-01-01")
    (sensitivity, _, _, _), _, _ = measure(upstream, "complement_c3", groups=groups, time_axis="essdai_index")
    assert sensitivity.time.min() < 0
    assert sensitivity.time_from_clinical_baseline_years.min() > 1.99
    assert sensitivity.loc[sensitivity.patient_id.eq("synthetic_000"), "delta"].tolist() == [0, -20]


def test_same_day_discordance_and_official_priority(upstream):
    x = analysis.attach_context(upstream["x"], upstream["context"])
    duplicate = x.loc[x.patient_id.eq("synthetic_000") & x.clinical_episode_id.eq("e001")].copy()
    duplicate["clinical_episode_id"] = "e999"
    duplicate["esspri__total"] = 3
    x = pd.concat([x, duplicate], ignore_index=True)
    (frame, _, qc, _), _, _ = measure(upstream, x=x)
    assert frame.loc[frame.patient_id.eq("synthetic_000"), "value"].tolist() == [10, 6]
    assert any(r["check"] == "same_day_discordant" for r in qc)
    duplicate = x.loc[x.patient_id.eq("synthetic_000") & x.clinical_episode_id.eq("e000")].copy()
    duplicate["clinical_episode_id"] = "e888"
    duplicate["is_clinical_baseline"] = False
    duplicate["esspri__total"] = 9
    (frame, _, _, _), _, _ = measure(upstream, x=pd.concat([x, duplicate], ignore_index=True))
    assert frame.loc[frame.patient_id.eq("synthetic_000"), "reference_value"].eq(10).all()


def test_missing_values_profiles_and_unchanged_denominators(upstream):
    x = analysis.attach_context(upstream["x"], upstream["context"])
    x.loc[x.patient_id.eq("synthetic_001"), "esspri__total"] = np.nan
    x.loc[x.patient_id.eq("synthetic_002") & ~x.is_clinical_baseline, "esspri__total"] = np.nan
    (frame, _, _, _), spec, groups = measure(upstream, x=x)
    support = analysis.support_table(frame, groups, spec, 3)
    worsen = support.loc[support.observed_pattern.eq("worsening_observed")].iloc[0]
    assert worsen.N_group == 2 and worsen.n_any == 1
    singleton = frame.loc[frame.patient_id.eq("synthetic_002")]
    assert len(singleton) == 1 and singleton.delta.isna().all()
    x.loc[~x.is_clinical_baseline, "esspri__total"] = np.nan
    (frame, status, _, _), _, _ = measure(upstream, x=x)
    assert status == "baseline_only_plotted"
    assert frame.delta.isna().all()


def test_binary_evaluable_denominators_and_cumulative_profile(upstream):
    (frame, status, _, unit), spec, _ = measure(upstream, "ocular_staining_positive")
    _, category, _ = analysis.summarize(frame, spec, 3, "clinical_baseline", unit)
    assert not category.window.eq(2).any()  # missing observations never become negatives
    assert (category.n_evaluable > 0).all()
    (ssa, status, _, unit), spec, _ = measure(upstream, "anti_ro_ssa")
    assert status == "baseline_only_plotted"
    assert spec.observation_semantics == "cumulative_through_episode"
    assert not spec.expected_longitudinal
    first = ssa.groupby("patient_id").head(1)
    assert first.value.eq(0).all()  # NA at clinical baseline did not become False


def test_patient_window_deduplication(upstream):
    (frame, _, _, unit), spec, _ = measure(upstream)
    extra = frame.copy()
    extra["variable_measurement_date"] += pd.Timedelta(days=1)
    extra["value"] = 1
    both = pd.concat([frame, extra], ignore_index=True)
    numeric, _, _ = analysis.summarize(both, spec, 3, "clinical_baseline", unit)
    assert numeric.n_patients_unique.max() == 2
    assert numeric.n_observations_raw.max() == 4
    assert numeric.absolute_median.eq(1).all()
    assert not analysis.window_contributions(both).duplicated(["patient_id", "window"]).any()


def test_registry_scans_every_column_and_unresolved_domains(upstream):
    x = upstream["x"].assign(unrecognized_numeric=1, **{"lab__new_analyte__value": 9})
    specs, catalog = build_catalog(x)
    assert set(x).issubset(set(catalog.canonical_column))
    assert "lab__new_analyte__value" in specs
    assert catalog.loc[catalog.canonical_column.eq("unrecognized_numeric"), "status"].iloc[0] == "excluded"
    assert catalog.loc[catalog.variable_id.eq("salivary_flow_unstimulated"), "exclusion_or_limitation_reason"].iloc[0] == "not_in_integrated"
    assert {"esspri", "labs", "serology", "ocular", "salivary", "pro", "pathology", "sgus", "overlap"}.issubset(set(catalog.domain))


@pytest.mark.parametrize("fault", ["duplicate_group", "unknown_pattern", "baseline_conflict", "date_after_cutoff", "selected_date_identity", "numeric_keys", "mismatched_eligibility"])
def test_blocking_contract_errors(upstream, fault):
    x, b, p, e, s = [upstream[k].copy(deep=True) for k in ("x", "b", "p", "e", "s")]
    if fault == "duplicate_group":
        p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    elif fault == "unknown_pattern":
        p.loc[0, "observed_pattern"] = "made_up"
    elif fault == "baseline_conflict":
        b.loc[b.index[0], "clinical_baseline_date"] += pd.Timedelta(days=1)
    elif fault == "date_after_cutoff":
        x.loc[1, "clinical_anchor_date"] = pd.Timestamp("2030-01-01")
    elif fault == "selected_date_identity":
        s.loc[0, "clinical_episode_id"] = "unknown"
    elif fault == "numeric_keys":
        x["patient_id"] = 123
    else:
        p.loc[0, "observed_pattern"] = "worsening_observed"
    with pytest.raises(analysis.ContractError):
        analysis.validate_inputs(x, b, p, e, s, "2025-01-01")


def test_artifact_input_and_origin_verification(upstream, tmp_path):
    for origin in ("canonical", "unknown"):
        with pytest.raises(analysis.ContractError, match="origin"):
            analysis.verified_manifest(upstream["manifest"], upstream["patterns"], upstream["integrated"], upstream["baseline"], origin)
    fake = tmp_path/"altered.parquet"
    fake.write_bytes(upstream["integrated"].read_bytes()+b"altered")
    with pytest.raises(analysis.ContractError, match="SHA256"):
        analysis.verified_manifest(upstream["manifest"], upstream["patterns"], fake, upstream["baseline"], "synthetic_validation")
    replica = tmp_path/"copy"
    shutil.copytree(upstream["root"], replica)
    relative = upstream["manifest"].relative_to(upstream["root"])
    manifest = json.loads((replica/relative).read_text())
    artifact = replica/manifest["artifacts"][0]["path"]
    artifact.write_bytes(artifact.read_bytes()+b"changed")
    with pytest.raises(analysis.ContractError, match="SHA256"):
        analysis.verified_manifest(replica/relative, replica/upstream["patterns"].relative_to(upstream["root"]), upstream["integrated"], upstream["baseline"], "synthetic_validation")


@pytest.mark.parametrize("fault,reason", [("unit_missing", "unknown_units"), ("unit_change", "incompatible"), ("date_missing", "invalid_date")])
def test_no_unverified_lab_units_or_dates(upstream, fault, reason):
    x = analysis.attach_context(upstream["x"], upstream["context"])
    if fault == "unit_missing":
        x = x.drop(columns="lab__complement_c3__unit")
    elif fault == "unit_change":
        x.loc[1, "lab__complement_c3__unit"] = "g/L"
    else:
        x = x.drop(columns="lab__complement_c3__measurement_date")
    (frame, status, _, _), _, _ = measure(upstream, "complement_c3", x=x)
    assert frame.empty and reason in status


def test_actual_four_panel_figure_semantics(upstream):
    (frame, _, _, unit), spec, groups = measure(upstream)
    support = analysis.support_table(frame, groups, spec, 3)
    fig = build_figure(frame, groups, support, spec, 3, "delta", "clinical_baseline", unit, "synthetic_validation")
    assert len(fig.axes) == 4
    assert "SYNTHETIC" in " ".join(t.get_text() for t in fig.texts)
    assert "±3" in fig._suptitle.get_text()
    for ax, pattern in zip(fig.axes, LABELS):
        assert "N_group=" in ax.get_title()
        assert all(line.get_color() == COLORS[pattern] or line.get_color() == "0.6" for line in ax.lines)
        horizontals = [line for line in ax.lines if len(set(line.get_ydata())) == 1]
        assert all(set(line.get_ydata()) == {0} for line in horizontals)
    assert any(collection.get_linestyle()[0][1] for ax in fig.axes for collection in ax.collections
               if isinstance(collection, LineCollection))
    plt.close(fig)


def test_full_png_bundles_manifests_and_readonly_upstream(upstream, tmp_path):
    hashes = {p: analysis.sha256(p) for p in upstream["root"].rglob("*") if p.is_file()}
    output = tmp_path/"result"
    args = args_for(upstream, output, "--threshold", "both")
    report = cli.run(args)
    assert report["status"] == "TESTED_SYNTHETIC / NOT_RUN_CANONICAL"
    dirs = cli.directories(output/"synthetic_validation", "clinical_baseline")
    manifest = json.loads((dirs["qc"]/"delta3/run_manifest.json").read_text())
    assert manifest["definition_version"] == analysis.DEFINITION
    assert manifest["n_png"] > 0
    for threshold in (3, 5):
        plot_manifest = pd.read_csv(dirs["tables"]/f"delta{threshold}/plot_manifest.csv")
        assert plot_manifest.threshold_abs.eq(threshold).all()
        assert set(plot_manifest.loc[plot_manifest.variable.eq("esspri_total"), "type"]) == {"absolute", "delta"}
        assert set(plot_manifest.loc[plot_manifest.variable.eq("anti_ro_ssa"), "type"]) == {"first_available_profile"}
        assert set(plot_manifest.loc[plot_manifest.variable.eq("overlap_status"), "type"]) == {"category_distribution"}
        for row in plot_manifest.itertuples():
            path = output/"synthetic_validation"/row.path
            assert path.suffix == ".png" and analysis.sha256(path) == row.sha256
            with Image.open(path) as image:
                image.verify()
        for csv in (dirs["tables"]/f"delta{threshold}").glob("*.csv"):
            # No identifying records or individual dates in any exported table.
            data = pd.read_csv(csv)
            assert not set(analysis.KEYS+["reference_date", "variable_measurement_date"]) & set(data)
            assert not any("synthetic_000" in str(value) for value in data.to_numpy().ravel())
    assert not list(output.rglob("*.pdf"))
    for entry in manifest["artifacts"]:
        assert analysis.sha256(output/"synthetic_validation"/entry["path"]) == entry["sha256"]
    assert {p: analysis.sha256(p) for p in hashes} == hashes
    with pytest.raises(FileExistsError):
        cli.run(args)
    # Dry run validates an existing bundle without overwriting it.
    args.dry_run = True
    assert cli.run(args)["dry_run"]


def test_missing_upstream_creates_no_final_files(tmp_path):
    with pytest.raises(analysis.ContractError, match="BLOCKED_UPSTREAM"):
        cli.run(cli.parse_args(["--integrated", str(tmp_path/"missing"), "--output-root", str(tmp_path/"out"), "--dry-run"]))
    assert not (tmp_path/"out").exists()


def test_install_rollback_and_backup(tmp_path, monkeypatch):
    stage, dest = {}, {}
    for area in ("figures", "tables", "qc"):
        stage[area], dest[area] = tmp_path/"stage"/area, tmp_path/"dest"/area
        stage[area].mkdir(parents=True)
        dest[area].mkdir(parents=True)
        (stage[area]/"new").write_text("new")
        (dest[area]/"old").write_text("old")
    original = shutil.move
    def failing_move(source, target):
        if Path(target).name == "qc":
            raise OSError("simulated install failure")
        return original(source, target)
    monkeypatch.setattr(shutil, "move", failing_move)
    with pytest.raises(OSError, match="simulated"):
        cli.install_bundle(stage, dest, tmp_path/"backups")
    assert all((d/"old").read_text() == "old" and not (d/"new").exists() for d in dest.values())
    assert len(list((tmp_path/"backups").rglob("old"))) == 3


def test_sequential_thresholds_and_overwrite_preserve_other_threshold(upstream, tmp_path):
    output = tmp_path/"sequential"
    three = args_for(upstream, output, "--threshold", "3", "--variables", "esspri_total")
    five = args_for(upstream, output, "--threshold", "5", "--variables", "esspri_total")
    cli.run(three)
    dirs = cli.directories(output/"synthetic_validation", "clinical_baseline")
    prior = {p: analysis.sha256(p) for area in dirs.values() for p in (area/"delta3").rglob("*") if p.is_file()}
    cli.run(five)
    assert {p: analysis.sha256(p) for p in prior} == prior
    five_hashes = {p: analysis.sha256(p) for area in dirs.values() for p in (area/"delta5").rglob("*") if p.is_file()}
    three.overwrite = True
    three.backup_root = output/"synthetic_validation/outputs/backups"/analysis.STEM
    cli.run(three)
    assert {p: analysis.sha256(p) for p in five_hashes} == five_hashes
    assert len(list(three.backup_root.rglob("run_manifest.json"))) == 1


def test_profile_denominator_pools_first_measurements_across_dates(upstream):
    (frame, _, _, unit), spec, _ = measure(upstream, "anti_ro_ssa")
    # Two people in the same ESSDAI group first observed in different years.
    frame.loc[frame.patient_id.eq("synthetic_004"), "window"] = 0
    _, category, _ = analysis.summarize(frame, spec, 3, "clinical_baseline", unit, profile=True)
    worsening = category.loc[category.group.eq("worsening_observed")]
    assert len(worsening) == 1 and worsening.n_evaluable.iloc[0] == 2
    assert worsening.summary_type.iloc[0] == "first_available_profile"
    assert worsening.window_start.isna().all()


def test_partial_dates_and_context_conflicts_are_blocking(upstream):
    with pytest.raises(analysis.ContractError, match="partial"):
        analysis.dates(pd.Series(["2020"]), "unverified", "2025-01-01")
    context = upstream["context"].copy()
    context.loc[0, "clinical_anchor_date"] = pd.Timestamp("2021-01-01")
    with pytest.raises(analysis.ContractError, match="context conflicts"):
        analysis.attach_context(upstream["x"], context)
    with pytest.raises(analysis.ContractError, match="coverage"):
        analysis.attach_context(upstream["x"], upstream["context"].iloc[:-1])


def test_no_data_panels_and_delta_singleton_exclusion(upstream):
    x = analysis.attach_context(upstream["x"], upstream["context"])
    x.loc[x.patient_id.eq("synthetic_002"), "esspri__total"] = np.nan
    x.loc[x.patient_id.eq("synthetic_003") & ~x.is_clinical_baseline, "esspri__total"] = np.nan
    (frame, _, _, unit), spec, groups = measure(upstream, x=x)
    support = analysis.support_table(frame, groups, spec, 3)
    fig = build_figure(frame, groups, support, spec, 3, "delta", "clinical_baseline", unit, "synthetic_validation")
    assert len(fig.axes) == 4
    assert "No observed data" in [t.get_text() for t in fig.axes[2].texts]
    # The singleton remains in the absolute/profile denominator, not a zero delta.
    assert not fig.axes[3].collections
    plt.close(fig)


def test_protocol_availability_counts_each_observed_protocol_once(upstream):
    (frame, _, _, unit), spec, _ = measure(upstream)
    extra = frame.copy()
    extra["source_protocol"] = "additional_protocol"
    _, _, availability = analysis.summarize(pd.concat([frame, extra]), spec, 3, "clinical_baseline", unit)
    assert "additional_protocol" in availability.source_protocol.unique()
    assert availability.n_patients_unique.max() <= 2


def test_consumer_never_calls_classification_or_models():
    forbidden = {"build_observed_patterns_v2", "classify_sequence", "build_observed_patterns", "fit", "score_all_pros"}
    for path in [ROOT/"src/block_A/15_secondary_trajectories_by_essdai_pattern.py", *ROOT.glob("src/block_A/secondary_trajectory_*.py")]:
        tree = ast.parse(path.read_text())
        called = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert not called & forbidden
