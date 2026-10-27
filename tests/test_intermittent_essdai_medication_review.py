"""Synthetic-only validation of the optional private documentary review."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.block_A import intermittent_medication_analysis as a
from src.block_A import observed_patterns_v2 as v2
from src.block_A.secondary_trajectory_analysis import ContractError, sha256
from src.block_A._internal.medications import contracts
from src.block_A._internal.medications.documentation import pseudonym
from src.block_A._internal.medications.episode_linkage import match
from src.block_A._internal.medications.episode_features import build
from src.block_A._internal.medications.persistence import publish, private_json

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/block_A" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load("private_step15", "15_intermittent_essdai_medication_review.py")
step13 = load("private_step13", "13_disease_activity_progression.py")


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "synthetic_validation" / "upstream"
    root.mkdir(parents=True)
    frames = []
    for i, scores in enumerate(([2, 6, 2], [8, 3, 8], [4, 8, 1], [2, 6], [2, 8, 2], [2, 3, 2])):
        times = pd.to_datetime(["2020-01-15", "2021-06-15", "2022-10-20"][:len(scores)])
        frame = pd.DataFrame(dict(patient_id=f"00{i:03d}",
            clinical_episode_id=[f"synthetic_e{j}" for j in range(len(scores))],
            clinical_anchor_date=times, episode_start_date=times, episode_end_date=times,
            clinical_baseline_date=times[0], clinical_baseline_episode_id="synthetic_e0",
            is_clinical_baseline=[True] + [False]*(len(scores)-1), clinical_visit=True,
            clinical_visit_number=range(1, len(scores)+1),
            time_since_clinical_baseline_days=(times-times[0]).days,
            time_since_clinical_baseline_years=(times-times[0]).days/365.25,
            **{"essdai__total": scores, "esspri__total": [8., 6., 4.][:len(scores)]}))
        for domain in a.DOMAINS:
            frame[f"essdai__{domain}_ordinal_score"] = [0., 1., np.nan][:len(scores)]
        frames.append(frame)
    x = pd.concat(frames, ignore_index=True)
    for key in a.KEYS:
        x[key] = x[key].astype("string")
    integrated, baseline, spine_path = (root / n for n in ("step10.parquet", "step11.parquet", "spine.parquet"))
    b = x.loc[x.is_clinical_baseline].copy()
    spine = x[contracts.STRUCTURE].copy()
    x.to_parquet(integrated, index=False)
    b.to_parquet(baseline, index=False)
    spine.to_parquet(spine_path, index=False)
    endpoints = step13.build_patient_absolute_change_endpoints(x)
    v2.write_observed_pattern_outputs_v2(x, b, endpoints, root, input_origin="synthetic_validation",
                                        input_paths={"step10": integrated, "step11": baseline})
    dirs = v2.output_directories(root)
    audit = root / "audit"
    key_path = audit / "restricted/.pseudonym_key"
    key_path.parent.mkdir(parents=True)
    secret = b"synthetic-only-test-key-never-for-production" * 2
    key_path.write_bytes(secret)
    private_json(audit / "restricted/08_key_compatibility.json", {"scope": pseudonym(secret, "compatibility", "08-medications")})
    documents = []
    specifications = [
        ("report_before", "CTDB", "hydroxychloroquine", "2020-01-15", "", "", "synthetic_e0"),
        ("report_after", "CTDB", "prednisone", "2021-06-15", "", "", "synthetic_e1"),
        ("order", "BTRIS", "rituximab", "2021-05-25", "2021-06-20", "", ""),
        ("consent", "BTRIS", "rituximab", "", "", "2020-01-15", ""),
        ("partial", "BTRIS", "prednisone", "2021-05", "", "", ""),
        ("undated", "BTRIS", "", "", "", "", ""),
        ("future", "BTRIS", "rituximab", "2030-01-01", "", "", ""),
        ("contradictory", "BTRIS", "rituximab", "2021-05-25", "2019-01-01", "", ""),
        ("post", "BTRIS", "prednisone", "2021-07-01", "", "", ""),
    ]
    for pid in x.patient_id.unique():
        for label, source, name, start, end, consent, ep in specifications:
            documents.append(dict(documentation_record_id=f"synthetic_{pid}_{label}", patient_id=pid,
                linked_patient_pseudonymous=pseudonym(secret, "patient", pid), source_kind=source, source="CTDB" if source == "CTDB" else "11D",
                canonical_medication_name=name, medication_name=name, name_requires_review=not bool(name), qc_flags="",
                original_episode_id=ep, visit_documentation_date=start if source == "CTDB" else "",
                order_start_raw=start if source == "BTRIS" else "", order_end_raw=end, initial_consent_raw=consent,
                clinical_category_primary="systemic_sjd", classification_confidence="high_exact",
                route_canonical="oral", formulation="", dose_raw="5", uom_raw="mg"))
    catalog = pd.DataFrame(documents).drop(columns="patient_id")
    links = match(catalog, spine, secret)
    features = build(spine, links)
    output = root / "features.parquet"
    qc = {"confirmed_medication_exposure": False}
    manifest = dict(contract_version=contracts.VERSION, spine_sha256=sha256(spine_path),
        spine_signature=contracts.spine_signature(spine), keys=contracts.KEYS,
        n_rows=len(features), n_episodes=len(features), n_patients=features.patient_id.nunique(),
        temporal_policy="one_calendar_year", confirmed_medication_exposure=False,
        consent_is_not_medication_date=True, qc=qc)
    publish(features, manifest, output, {
        a.CATALOG: catalog, a.LINKS: links,
        a.REVIEW_FILES[0]: catalog.loc[~catalog.documentation_record_id.isin(links.documentation_record_id)],
        a.REVIEW_FILES[1]: links.loc[links.name_requires_review]}, qc, audit)
    return dict(root=root, integrated=integrated, baseline=baseline, spine=spine_path, x=x,
        features=output, key=key_path, audit=audit,
        patterns=dirs["analytic"] / "13_essdai_patient_observed_patterns_v2.parquet",
        manifest=dirs["qc"] / "13_essdai_observed_pattern_manifest_v2.json")


def args_for(u, output, *extra):
    return cli.parse_args(["--integrated", str(u["integrated"]), "--baseline", str(u["baseline"]),
        "--spine", str(u["spine"]), "--patterns", str(u["patterns"]), "--patterns-manifest", str(u["manifest"]),
        "--medication-features", str(u["features"]), "--medication-audit-root", str(u["audit"]),
        "--pseudonym-key", str(u["key"]), "--restricted-output-root", str(output),
        "--input-origin", "synthetic_validation", "--data-cutoff", "2025-01-01", "--dpi", "50", *extra])


def data(u, k=3):
    _, paths = a_verified(u)
    patterns = pd.read_parquet(u["patterns"])
    s = pd.read_parquet(paths["selected_dates"])
    roster, visits = a.selected_visits(u["x"], patterns, s, k, "2025-01-01")
    features, catalog, links, _, _ = a.medication_bundle(u["features"], u["audit"], u["spine"], u["x"], u["key"], "2025-01-01")
    jumps, domains, _ = a.intervals(visits, k)
    history = a.documentary_history(catalog, links, roster.patient_id, "2025-01-01")
    return roster, visits, jumps, domains, history, features


def a_verified(u):
    from src.block_A.secondary_trajectory_analysis import verified_manifest
    return verified_manifest(u["manifest"], u["patterns"], u["integrated"], u["baseline"], "synthetic_validation")


def test_fixed_labels_consecutive_intervals_and_independent_thresholds(upstream):
    roster, visits, jumps, domains, _, _ = data(upstream)
    assert set(roster.patient_id) == {"00000", "00001", "00002", "00004"}
    assert len(jumps) == 8
    assert set(data(upstream, 5)[0].patient_id) == {"00001", "00004"}
    assert jumps.days_between.gt(0).all()
    assert jumps.measurement_gap_warning.all()
    assert len(domains) == len(jumps)*12
    first = jumps.loc[jumps.patient_id.eq("00000")]
    assert first.delta_essdai_interval.tolist() == [4, -4]
    assert first.threshold_up_entered.tolist() == [True, False]
    assert first.threshold_up_exited.tolist() == [False, True]
    assert visits.patient_id.str.startswith("00").all()
    comparison = a.threshold_comparison(pd.read_parquet(upstream["patterns"]))
    assert len(comparison) == 4 and comparison.pattern_changed.sum() == 2


def test_source_dates_consent_partial_future_and_document_dedup(upstream):
    roster, _, jumps, _, history, features = data(upstream)
    enriched, evidence, contrast = a.evidence_for_intervals(jumps, history, features)
    first = evidence.loc[evidence.patient_id.eq("00000") & evidence.jump_id.eq("interval_0001")]
    assert set(first.loc[first.documentation_record_id.str.endswith("_order"), "temporal_relation"]) == {"documented_in_interval", "documented_after_to"}
    assert first.loc[first.reference_kind.eq("initial_consent"), "temporal_relation"].eq("consent_only").all()
    assert first.loc[first.documentation_record_id.str.endswith(("_partial", "_future", "_contradictory")), "temporal_relation"].eq("date_unknown_or_partial").all()
    assert history.confirmed_exposure.eq("unknown").all()
    assert history.loc[history.reference_kind.eq("order_end"), "confirmed_exposure_event"].isna().all()
    assert contrast.loc[contrast.patient_id.eq("00000") & contrast.jump_id.eq("interval_0001") & contrast.canonical_medication_name.eq("prednisone"), "report_contrast"].tolist() == ["reported_only_at_to"]
    assert not contrast.start_or_stop_inferred.any()
    assert enriched.post_jump_order_possible_reverse_causation.any()
    assert enriched.sjd_systemic_drug_documented_near_interval.eq("unknown").all()
    assert history.loc[history.patient_id.eq("00000"), "documentation_record_id"].nunique() == 9
    summary = a.cohort_summary(enriched, evidence, 3)
    assert summary.loc[summary.suppressed, "n_documents_unique"].isna().all()
    # Duplicated links never create extra references or document counts.
    _, catalog, links, _, _ = a.medication_bundle(upstream["features"], upstream["audit"], upstream["spine"], upstream["x"], upstream["key"], "2025-01-01")
    again = a.documentary_history(catalog, pd.concat([links, links]), roster.patient_id, "2025-01-01")
    assert len(again) == len(history)


def test_domains_missing_are_not_zero_or_weighted_points(upstream):
    _, _, _, domains, _, features = data(upstream)
    missing = domains.loc[domains.jump_id.eq("interval_0002")]
    assert missing.ordinal_after.isna().all()
    assert missing.direction.eq("not_comparable").all()
    assert not missing.weighted_contribution_available.any()
    assert missing.weighted_point_change.isna().all()
    features.loc[:, contracts.COLUMNS] = pd.NA
    _, _, jumps, _, history, _ = data(upstream)
    _, _, contrasts = a.evidence_for_intervals(jumps, history, features)
    assert contrasts.report_contrast.eq("not_documented_either").all()
    assert not contrasts.from_list_documented.any()
    assert contrasts.confirmed_exposure.eq("unknown").all()


@pytest.mark.parametrize("mutation", ["key", "hash", "generation", "spine", "step10_names", "exposure"])
def test_provenance_fails_closed(upstream, tmp_path, mutation):
    if mutation == "key":
        upstream["key"].write_bytes(b"different-synthetic-key" * 3)
    elif mutation == "hash":
        path = upstream["audit"] / "restricted/current" / a.CATALOG
        path.write_bytes(path.read_bytes() + b"changed")
    elif mutation == "generation":
        current = upstream["audit"] / "restricted/current"
        current.unlink()
        current.symlink_to("generations/not-the-current-generation")
    elif mutation == "spine":
        frame = pd.read_parquet(upstream["spine"])
        frame.loc[0, "patient_id"] = "wrong_synthetic"
        frame.to_parquet(upstream["spine"], index=False)
    elif mutation == "step10_names":
        upstream["x"][contracts.COLUMNS[0]] = "different-synthetic-value"
    else:
        path = upstream["features"].with_suffix(".manifest.json")
        meta = json.loads(path.read_text())
        meta["confirmed_medication_exposure"] = True
        path.write_text(json.dumps(meta))
    with pytest.raises((ValueError, FileNotFoundError)):
        a.medication_bundle(upstream["features"], upstream["audit"], upstream["spine"], upstream["x"], upstream["key"], "2025-01-01")


def rehash_restricted(u, name):
    path = u["features"].with_suffix(".manifest.json")
    meta = json.loads(path.read_text())
    meta["restricted_outputs_sha256"][name] = sha256(u["audit"] / "restricted/current" / name)
    path.write_text(json.dumps(meta))


@pytest.mark.parametrize("mutation", ["patient", "episode", "selected_date", "catalog_hmac"])
def test_semantic_identity_checked_even_with_matching_hashes(upstream, mutation):
    name = a.CATALOG if mutation == "catalog_hmac" else a.LINKS
    path = upstream["audit"] / "restricted/current" / name
    frame = pd.read_parquet(path)
    col, value = {"patient": ("patient_id", "wrong_synthetic"), "episode": ("clinical_episode_id", "wrong_synthetic"),
                  "selected_date": ("selected_date", pd.Timestamp("2024-01-01")),
                  "catalog_hmac": ("linked_patient_pseudonymous", "unknown_synthetic_scope")}[mutation]
    frame.loc[0, col] = value
    frame.to_parquet(path, index=False)
    rehash_restricted(upstream, name)
    with pytest.raises(ValueError):
        a.medication_bundle(upstream["features"], upstream["audit"], upstream["spine"], upstream["x"], upstream["key"], "2025-01-01")


def test_order_support_and_two_points_cannot_be_overridden(upstream):
    _, paths = a_verified(upstream)
    sequence = pd.read_parquet(paths["selected_dates"])
    patterns = pd.read_parquet(upstream["patterns"])
    with pytest.raises(ContractError, match="chronological"):
        a.selected_visits(upstream["x"], patterns, sequence.iloc[::-1], 3, "2025-01-01")
    patterns.loc[patterns.patient_id.eq("00003"), "observed_pattern"] = "intermittent_observed"
    with pytest.raises(ContractError, match="three dates"):
        a.selected_visits(upstream["x"], patterns, sequence, 3, "2025-01-01")


def test_dry_run_is_read_only_and_missing_upstream_is_blocked(upstream, tmp_path, capsys):
    before = {p: sha256(p) for p in upstream["root"].rglob("*") if p.is_file()}
    output = tmp_path / "restricted"
    args = args_for(upstream, output, "--dry-run")
    assert cli.run(args)["status"] == "validated_dry_run_no_publication"
    assert not output.exists()
    assert all(sha256(p) == value for p, value in before.items())
    upstream["integrated"].unlink()
    assert cli.main(["--restricted-output-root", str(output), "--integrated", str(upstream["integrated"]), "--dry-run"]) == 2
    assert "BLOCKED_UPSTREAM" in capsys.readouterr().err
    assert not output.exists()


def test_output_policy_and_no_console_identifiers(upstream, tmp_path, capsys):
    with pytest.raises(ContractError):
        cli.output_destination(ROOT / "restricted", "canonical")
    public = tmp_path / "restricted"
    public.mkdir(mode=0o755)
    os.chmod(public, 0o755)
    with pytest.raises(ContractError):
        cli.output_destination(public, "canonical")
    link = tmp_path / "alias"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContractError):
        cli.output_destination(link / "restricted", "canonical")
    assert cli.main(["--restricted-output-root", str(public), "--dry-run"]) == 2
    logs = capsys.readouterr()
    assert "rituximab" not in logs.err and "00000" not in logs.err


def test_complete_private_publication_figures_and_unchanged_sources(upstream, tmp_path):
    before = {p: sha256(p) for p in upstream["root"].rglob("*") if p.is_file()}
    output = tmp_path / "restricted"
    args = args_for(upstream, output)
    assert cli.run(args)["status"] == "published_restricted"
    destination = cli.output_destination(output, "synthetic_validation")
    for k, n in ((3, 4), (5, 2)):
        folder = destination / "clinical_baseline" / f"delta{k}"
        roster = pd.read_parquet(folder / "intermittent_patient_roster.parquet")
        assert len(roster) == n
        assert len(list((folder / "figures").glob("patient_*.png"))) == n
        assert len(list((folder / "figures").glob("jump_*.png"))) == n * 2
        metadata = json.loads((folder / "run_manifest.json").read_text())
        assert metadata["input_origin"] == "synthetic_validation"
        assert metadata["threshold_abs"] == k
        for name, digest in metadata["outputs_sha256"].items():
            assert sha256(destination / name) == digest
    for path in output.rglob("*"):
        assert path.stat().st_mode & 0o777 == (0o700 if path.is_dir() else 0o600)
        if path.suffix == ".png":
            with Image.open(path) as image:
                image.verify()
    assert all(sha256(p) == value for p, value in before.items())
    with pytest.raises(ContractError, match="already exists"):
        cli.run(args)
    assert not list(output.glob(".step15-review-*"))


def test_empty_intermittent_cohort_and_no_false_treatment_absence(upstream):
    _, paths = a_verified(upstream)
    patterns = pd.read_parquet(upstream["patterns"])
    patterns["observed_pattern"] = "stable_observed"
    roster, visits = a.selected_visits(upstream["x"], patterns, pd.read_parquet(paths["selected_dates"]), 3, "2025-01-01")
    jumps, _, _ = a.intervals(visits, 3)
    history = a.documentary_history(pd.DataFrame(columns=["patient_id"]), pd.DataFrame(columns=["documentation_record_id"]), [], "2025-01-01")
    jumps, evidence, _ = a.evidence_for_intervals(jumps, history, pd.read_parquet(upstream["features"]))
    assert a.cohort_summary(jumps, evidence, 3).n_intervals_total.eq(0).all()
    assert roster.empty


def test_small_interval_can_explain_threshold_loss(upstream):
    _, visits, _, _, _, _ = data(upstream)
    visits = visits.loc[visits.patient_id.eq("00000")].copy()
    visits["essdai__total"] = [2., 5., 4.]
    jumps, _, _ = a.intervals(visits, 3)
    assert jumps.delta_essdai_interval.tolist() == [3., -1.]
    assert jumps.iloc[1].threshold_up_exited
    assert jumps.iloc[1].explains_observed_pattern
    assert not jumps.iloc[1].abs_change_ge_k_in_interval


def test_empty_documentation_is_unknown_and_explicit_in_denominators(upstream):
    _, _, jumps, _, history, features = data(upstream)
    enriched, evidence, _ = a.evidence_for_intervals(jumps, history.iloc[:0], features)
    assert enriched.treatment_documentation_in_interval.eq("unknown").all()
    assert enriched.temporal_status_unresolved.all()
    summary = a.cohort_summary(enriched, evidence, 3, minimum_cell=1)
    absent = summary.loc[summary.temporal_relation.eq("no_source_documentation_in_review_window") & summary.n_intervals_total.gt(0)]
    assert absent.proportion_intervals.eq(1).all()
    assert absent.n_documents_unique.eq(0).all()


def test_failed_render_rolls_back_entire_private_run(upstream, tmp_path, monkeypatch):
    from src.block_A import intermittent_medication_figures as figures
    def broken(*args, **kwargs):
        raise RuntimeError("synthetic rendering failure")
    monkeypatch.setattr(figures, "timeline", broken)
    output = tmp_path / "restricted"
    args = args_for(upstream, output, "--threshold", "5")
    with pytest.raises(RuntimeError, match="synthetic rendering failure"):
        cli.run(args)
    assert not cli.output_destination(output, "synthetic_validation").exists()
    assert not list(output.rglob("*.parquet"))
    assert not list(output.rglob(".step15-review-*"))
