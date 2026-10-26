"""Scientific equivalence against the pinned implementation; synthetic only."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
import numpy as np
import pandas as pd
import pytest

from test_secondary_trajectories_by_essdai_pattern import upstream, args_for, cli, measure
from src.block_A import secondary_trajectory_analysis as analysis
from src.block_A import secondary_trajectory_figures as figures
from src.block_A.secondary_trajectory_catalog import build_catalog
from step15_reference import reference


@pytest.fixture(scope="module")
def old():
    return reference()


def equal(left, right):
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                  check_dtype=False, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("threshold", [3, 5])
@pytest.mark.parametrize("axis", ["clinical_baseline", "essdai_index"])
def test_measurement_and_summary_equivalence(upstream, old, threshold, axis):
    x = analysis.attach_context(upstream["x"], upstream["context"])
    # Discordant same-day follow-up, discordant baseline, concordant same-day,
    # prebaseline, singletons, invalid scale, conflicts, and delayed references.
    duplicate = x.iloc[[0, 1, 2]].copy()
    duplicate["clinical_episode_id"] = ["e990", "e991", "e992"]
    duplicate["is_clinical_baseline"] = False
    duplicate["esspri__total"] = [9, 3, 6]
    x = pd.concat([x, duplicate], ignore_index=True)
    x.loc[x.patient_id.eq("synthetic_001"), "esspri__total"] = np.nan
    x.loc[x.patient_id.eq("synthetic_002") & ~x.is_clinical_baseline, "esspri__total"] = 99
    groups = analysis.fixed_groups(upstream["p"], threshold)
    specs, _ = build_catalog(upstream["x"])
    for spec in specs.values():
        before = old[0].observations(x, groups, spec, "2025-01-01", axis)
        after = analysis.observations(x, groups, spec, "2025-01-01", axis)
        assert before[1:] == after[1:]
        columns = list(after[0])
        equal(before[0][columns], after[0])
        equal(old[0].support_table(before[0], groups, spec, threshold), analysis.support_table(after[0], groups, spec, threshold))
        if not after[0].empty:
            for profile in (True, False):
                a = old[0].summarize(before[0], spec, threshold, axis, after[3], profile=profile)
                b = analysis.summarize(after[0], spec, threshold, axis, after[3], profile=profile)
                for aa, bb in zip(a, b):
                    equal(aa, bb)


@pytest.mark.parametrize("fault", ["index_date", "index_episode", "null_selected_index", "other_threshold", "count"])
def test_vectorized_validation_rejects_reference_conflicts(upstream, old, fault):
    x, b, p, e, s = [upstream[k].copy(deep=True) for k in ("x", "b", "p", "e", "s")]
    if fault == "null_selected_index":
        s.loc[0, "essdai_index_date"] = pd.NaT
    elif fault == "index_date":
        s.loc[0, "essdai_index_date"] = pd.Timestamp("2021-01-01")
    else:
        mask = p.patient_id.eq("synthetic_000") & p.threshold_abs.eq(5)
        emask = e.patient_id.eq("synthetic_000") & e.threshold_abs.eq(5)
        col, value = {"index_episode": ("essdai_index_episode_id", "missing"),
                      "other_threshold": ("essdai_index_date", pd.Timestamp("2022-01-01")),
                      "count": ("n_distinct_essdai_dates", 99)}[fault]
        p.loc[mask, col] = value
        e.loc[emask, col] = value
    for module in (old[0], analysis):
        with pytest.raises(analysis.ContractError if module is analysis else old[0].ContractError):
            module.validate_inputs(x, b, p, e, s, "2025-01-01")


@pytest.mark.parametrize("fault", ["duplicate", "missing", "extra", "conflict", "nulls", "shuffle"])
def test_context_alignment_equivalence(upstream, old, fault):
    x, context = upstream["x"].copy(), upstream["context"].copy()
    if fault == "duplicate":
        context = pd.concat([context, context.iloc[[0]]])
    elif fault == "missing":
        context = context.iloc[1:]
    elif fault == "extra":
        extra = context.iloc[[0]].assign(clinical_episode_id="extra")
        context = pd.concat([context, extra])
    elif fault == "conflict":
        context.loc[0, "clinical_anchor_date"] = pd.Timestamp("2021-01-01")
    elif fault == "nulls":
        context["same_missing"] = x["same_missing"] = np.nan
    else:
        context = context.sample(frac=1, random_state=7)
    if fault in ("nulls", "shuffle"):
        equal(old[0].attach_context(x, context), analysis.attach_context(x, context))
    else:
        for module in (old[0], analysis):
            with pytest.raises(module.ContractError):
                module.attach_context(x, context)


@pytest.mark.parametrize("kind", ["absolute", "delta"])
@pytest.mark.parametrize("threshold", [3, 5])
def test_batched_points_and_segments_equal_original(upstream, old, kind, threshold):
    (frame, _, _, unit), spec, groups = measure(upstream, groups=analysis.fixed_groups(upstream["p"], threshold))
    support = analysis.support_table(frame, groups, spec, threshold)
    previous = old[1].build_figure(frame, groups, support, spec, threshold, kind, "clinical_baseline", unit, "synthetic_validation")
    current = figures.build_figure(frame, groups, support, spec, threshold, kind, "clinical_baseline", unit, "synthetic_validation")
    try:
        for a, b in zip(previous.axes, current.axes):
            old_points = [c.get_offsets() for c in a.collections if isinstance(c, PathCollection)]
            new_points = [c.get_offsets() for c in b.collections if isinstance(c, PathCollection)]
            old_points = np.concatenate(old_points) if old_points else np.empty((0, 2))
            new_points = np.concatenate(new_points) if new_points else np.empty((0, 2))
            np.testing.assert_allclose(old_points, new_points)
            expected = []
            for line in a.lines:
                if line.get_color() == "0.6":
                    continue
                xy = np.column_stack([line.get_xdata(), line.get_ydata()])
                expected += [(line.get_linestyle(), pair.tolist()) for pair in np.stack([xy[:-1], xy[1:]], axis=1)]
            actual = []
            for c in b.collections:
                if isinstance(c, LineCollection):
                    style = "--" if c.get_linestyle()[0][1] else "-"
                    actual += [(style, segment.tolist()) for segment in c.get_segments()]
            assert sorted(map(str, expected)) == sorted(map(str, actual))
            assert a.get_title() == b.get_title()
            assert [t.get_text() for t in a.texts] == [t.get_text() for t in b.texts]
    finally:
        plt.close(previous)
        plt.close(current)


def test_full_bundle_tables_equal_reference(upstream, old, tmp_path):
    previous, current = tmp_path/"old", tmp_path/"new"
    old[2].run(args_for(upstream, previous, "--threshold", "both"))
    cli.run(args_for(upstream, current, "--threshold", "both"))
    before = cli.directories(previous/"synthetic_validation", "clinical_baseline")
    after = cli.directories(current/"synthetic_validation", "clinical_baseline")
    for threshold in (3, 5):
        for path in (before["tables"]/f"delta{threshold}").glob("*.csv"):
            a, b = pd.read_csv(path), pd.read_csv(after["tables"]/f"delta{threshold}"/path.name)
            if path.name == "plot_manifest.csv":
                a, b = a.drop(columns="sha256"), b.drop(columns="sha256")
            equal(a, b)
        equal(pd.read_csv(before["qc"]/f"delta{threshold}/qc_checks.csv"),
              pd.read_csv(after["qc"]/f"delta{threshold}/qc_checks.csv"))
        a = json.loads((before["qc"]/f"delta{threshold}/run_manifest.json").read_text())
        b = json.loads((after["qc"]/f"delta{threshold}/run_manifest.json").read_text())
        for key in a.keys() - {"artifacts", "created_at_utc", "source_sha256", "parameters"}:
            assert a[key] == b[key]


def test_dry_run_no_staging_and_progress(upstream, tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("dry run attempted staging or rendering")
    monkeypatch.setattr(cli.tempfile, "TemporaryDirectory", fail)
    monkeypatch.setattr(cli, "plot_variable", fail)
    output = tmp_path/"absent"
    report = cli.run(args_for(upstream, output, "--threshold", "both", "--dry-run"))
    assert not output.exists()
    assert report["runtime"]["wall_seconds"] > 0
    assert "analysis=" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["render", "hash"])
def test_failure_preserves_outputs_and_cleans_stage(upstream, tmp_path, monkeypatch, failure):
    output = tmp_path/"output"
    args = args_for(upstream, output, "--variables", "esspri_total")
    cli.run(args)
    hashes = {p: analysis.sha256(p) for p in output.rglob("*") if p.is_file()}
    original_plot = cli.plot_variable
    def fail(*args, **kwargs):
        original_plot(*args, **kwargs)
        if failure == "render":
            raise RuntimeError("render failure")
        monkeypatch.setattr(cli, "sha256", lambda path: "changed")
    monkeypatch.setattr(cli, "plot_variable", fail)
    args.overwrite = True
    args.backup_root = output/"synthetic_validation/outputs/backups"/analysis.STEM
    with pytest.raises((RuntimeError, analysis.ContractError)):
        cli.run(args)
    assert {p: analysis.sha256(p) for p in hashes} == hashes
    assert not list(output.rglob(".step15-*"))
    assert not plt.get_fignums()


@pytest.mark.parametrize("fault", ["unit", "version", "date", "producer_conflict", "prebaseline", "binary", "ordinal"])
def test_measurement_qc_edge_equivalence(upstream, old, fault):
    from dataclasses import replace
    x = analysis.attach_context(upstream["x"], upstream["context"])
    groups = analysis.fixed_groups(upstream["p"], 3)
    specs, _ = build_catalog(upstream["x"])
    spec = next(s for s in specs.values() if s.variable_id == "complement_c3")
    if fault == "unit":
        # Out-of-group bad units must never block eligible measurements.
        groups = groups.loc[~groups.patient_id.eq("synthetic_000")]
        x.loc[x.patient_id.eq("synthetic_000"), "lab__complement_c3__unit"] = "incompatible"
    elif fault == "version":
        x["lab__complement_c3__assay"] = "v1"
        x.loc[1, "lab__complement_c3__assay"] = "v2"
    elif fault == "date":
        x.loc[1, spec.measurement_date_column] = pd.NaT
    elif fault == "prebaseline":
        x.loc[1, spec.measurement_date_column] = pd.Timestamp("2019-01-01")
    elif fault == "producer_conflict":
        x["lab__complement_c3__conflict"] = False
        x.loc[1, "lab__complement_c3__conflict"] = True
    elif fault == "binary":
        spec = next(s for s in specs.values() if s.variable_id == "ocular_staining_positive")
    else:
        spec = replace(next(s for s in specs.values() if s.variable_id == "esspri_total"), type="ordinal")
        x[spec.canonical_column] = x[spec.canonical_column].astype(float)
        x.loc[1, spec.canonical_column] = 2.5
    a = old[0].observations(x, groups, spec, "2025-01-01")
    b = analysis.observations(x, groups, spec, "2025-01-01")
    assert a[1:] == b[1:]
    equal(a[0][list(b[0])], b[0])


def test_baseline_hash_tamper_blocks(upstream, tmp_path):
    altered = tmp_path/"altered_step11.parquet"
    altered.write_bytes(upstream["baseline"].read_bytes()+b"tampered")
    with pytest.raises(analysis.ContractError, match="SHA256"):
        analysis.verified_manifest(upstream["manifest"], upstream["patterns"], upstream["integrated"], altered, "synthetic_validation")


def test_partial_figure_is_closed(upstream, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        plt.subplots()
        raise RuntimeError("layout failed")
    monkeypatch.setattr(figures, "build_figure", fail)
    (frame, _, _, unit), spec, groups = measure(upstream)
    with pytest.raises(RuntimeError, match="layout"):
        figures.plot_variable(frame, groups, analysis.support_table(frame, groups, spec, 3), spec, 3, "absolute",
            tmp_path/"unused.png", time_axis="clinical_baseline", unit=unit, origin="synthetic_validation")
    assert not plt.get_fignums()


@pytest.mark.parametrize("threshold", [3, 5])
@pytest.mark.parametrize("variable,kind", [("ocular_staining_positive", "prevalence"), ("overlap_status", "category_distribution"),
    ("anti_ro_ssa", "first_available_profile"), ("esspri_total", "category_distribution")])
def test_category_figures_equal_reference(upstream, old, threshold, variable, kind):
    from dataclasses import replace
    (frame, _, _, unit), spec, groups = measure(upstream, variable, groups=analysis.fixed_groups(upstream["p"], threshold))
    if variable == "esspri_total":
        spec = replace(spec, type="ordinal")
    support = analysis.support_table(frame, groups, spec, threshold)
    args = (frame, groups, support, spec, threshold, kind, "clinical_baseline", unit, "synthetic_validation")
    previous, current = old[1].build_figure(*args), figures.build_figure(*args)
    try:
        for a, b in zip(previous.axes, current.axes):
            assert a.get_title() == b.get_title()
            assert [t.get_text() for t in a.texts] == [t.get_text() for t in b.texts]
            assert len(a.collections) == len(b.collections)
            for x, y in zip(a.collections, b.collections):
                np.testing.assert_allclose(x.get_offsets(), y.get_offsets())
                if isinstance(x, LineCollection):
                    np.testing.assert_allclose(x.get_segments(), y.get_segments())
            assert [(p.get_x(), p.get_height(), p.get_y()) for p in a.patches] == [(p.get_x(), p.get_height(), p.get_y()) for p in b.patches]
    finally:
        plt.close(previous)
        plt.close(current)


def test_direct_summary_null_protocol_equivalence(upstream, old):
    (frame, _, _, unit), spec, _ = measure(upstream)
    frame["source_protocol"] = np.nan
    for profile in (False, True):
        a = old[0].summarize(frame, spec, 3, "clinical_baseline", unit, profile=profile)
        b = analysis.summarize(frame, spec, 3, "clinical_baseline", unit, profile=profile)
        for left, right in zip(a, b):
            equal(left, right)
