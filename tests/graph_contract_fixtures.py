"""Synthetic-only approvals and artifact bundles for Graph contract tests."""
import json

import pandas as pd
import yaml

from src.studies.graph import _lab_contract as LAB


def candidate(lab="lab__glucose", status="include_numeric", **changes):
    numeric = status != "include_categorical"
    source = lab + ("__value" if numeric else "__text")
    row = dict(lab=lab, graph_candidate_status=status,
               graph_encoding_hint="review" if status == "review_before_graph" else "continuous" if numeric else "binary",
               primary_result_source="value_numeric" if numeric else "text_categorical",
               value_column=lab + "__value" if numeric else "", text_column="" if numeric else lab + "__text",
               reference_status_column="", source_column=source,
               profile_contract_version=LAB.PROFILE_VERSION,
               clinical_review_required=status == "review_before_graph",
               clinically_approved="pending" if status == "review_before_graph" else "approved",
               unit_assessment_status="not_assessed", eligibility_reason="SYNTHETIC_TEST_ONLY")
    row.update(changes)
    return row


def empty_candidates():
    return pd.DataFrame(columns=sorted(LAB.REQUIRED_COLUMNS))


def write_bundle(integrated, candidates_path, *, approved=True, profile=None, drift=None):
    """Make a reviewable synthetic release; never used by production code."""
    folder = candidates_path.parent
    candidates = pd.read_csv(candidates_path)
    paths = {"integrated": integrated, "pharma_candidates": candidates_path,
             "pharma_summary": folder / "00_pharma_profile_summary.json",
             "pharma_profile": folder / "00_pharma_lab_profile.csv",
             "pharma_expected_audit": folder / "00_pharma_expected_lab_audit.csv",
             "pharma_impact_audit": folder / "00_pharma_downstream_impact.csv"}
    if profile is None:
        profile = candidates
    profile.to_csv(paths["pharma_profile"], index=False)
    for role in ("pharma_expected_audit", "pharma_impact_audit"):
        pd.DataFrame({"lab": candidates.lab, "review": "SYNTHETIC_TEST_ONLY"}).to_csv(paths[role], index=False)
    summary = dict(profile_contract_version=LAB.PROFILE_VERSION,
                   input_hashes=[dict(role="integrated", sha256=LAB.sha256(integrated)),
                                 dict(role="registry", sha256="0" * 64)],
                   context_mode="integrated_only_explicit", schema_drift=drift or {},
                   release_recommendation="hold_for_clinical_diff_and_temporal_review_no_downstream_reruns_authorized")
    paths["pharma_summary"].write_text(json.dumps(summary))
    release = dict(release_id="SYNTHETIC_TEST_ONLY", status="approved_for_analysis" if approved else "pending_review",
                   reviewer="synthetic_test", review_date="2026-10-16", review_evidence="Synthetic fixture; no real clinical approval",
                   analysis_scope="graph01_graph02_baseline", acknowledged_release_recommendation=summary["release_recommendation"],
                   reviewed_expected_lab_audit=True, reviewed_downstream_impact=True, schema_drift_reviewed=True,
                   approved_lab_stems=candidates.loc[candidates.graph_candidate_status.isin({"include_numeric", "include_categorical"}), "lab"].tolist())
    for role in ("integrated", "pharma_candidates", "pharma_summary", "pharma_expected_audit", "pharma_impact_audit"):
        release[role + "_sha256"] = LAB.sha256(paths[role])
    release_path = folder / "synthetic_graph_release.yaml"
    release_path.write_text(yaml.safe_dump(release))
    return release_path, paths, summary, release
