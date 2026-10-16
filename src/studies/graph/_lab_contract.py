"""Shared, fail-closed Pharma 00 ingestion and scientific release governance.

Only aggregate metadata leave preflight. This module never writes producer inputs
and never grants assay approval; that decision belongs exclusively to Pharma 00.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from src.studies._shared import STUDY_CONTRACT_VERSION, validate_integrated_dataset

PROFILE_VERSION = "pharma_lab_profile_v2"
FEATURE_POLICIES = ("legacy_with_overlap", "without_overlap")
REQUIRED_COLUMNS = {
    "lab", "graph_candidate_status", "graph_encoding_hint", "primary_result_source",
    "value_column", "text_column", "reference_status_column", "source_column",
    "profile_contract_version", "clinical_review_required", "clinically_approved",
    "unit_assessment_status", "eligibility_reason",
}
BINARY_PAIRS = ({"positive", "negative"}, {"pos", "neg"},
                {"reactive", "nonreactive"}, {"detected", "not detected"},
                {"present", "absent"}, {"high", "normal"}, {"low", "normal"})
UNKNOWN_CATEGORIES = {"", "unknown", "not reported", "not_reported", "n/a", "na",
                      "uninterpretable", "indeterminate", "not tested", "not_tested"}
POSITIVE = {"positive", "pos", "reactive", "detected", "present", "high", "low"}


class LabContractError(ValueError):
    """A fixed input/release error, never bootstrap instability."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise LabContractError("CONTRACT_MISMATCH", f"Missing required input {path}; regenerate Pharma 00")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_fingerprint(raw: pd.DataFrame) -> str:
    """Hash column order, values and missingness without exporting patient rows."""
    digest = hashlib.sha256(json.dumps(list(raw.columns)).encode())
    digest.update(pd.util.hash_pandas_object(raw, index=True).values.tobytes())
    return digest.hexdigest()


def extend_ingestion_audit(audit, baseline, profile, policy, release):
    """Add profiled but unexported assays as aggregate audit rows only."""
    rows = []
    known = set(audit.lab) if "lab" in audit else set()
    for row in profile.itertuples(index=False):
        if row.lab in known:
            continue
        sources = [row.lab + suffix for suffix in ("__value", "__text") if row.lab + suffix in baseline]
        observed = baseline[sources].notna().any(axis=1) if sources else pd.Series(False, index=baseline.index)
        rows.append(dict(lab=row.lab, graph_candidate_status=row.graph_candidate_status,
                         graph_encoding_hint="none", source_column=" | ".join(sources),
                         source_column_found=bool(sources), baseline_nonmissing=int(observed.sum()),
                         baseline_coverage=float(observed.mean()) if len(observed) else 0.,
                         included_in_graph=False, final_graph_feature_names="",
                         exclusion_reason="profiled_but_not_exported_as_graph_candidate" if sources else "not_observed_in_step10",
                         pharma_contract_version=PROFILE_VERSION, clinically_approved=getattr(row, "clinically_approved", ""),
                         feature_policy=policy, graph_release_id=release.get("release_id") if release else None))
    return pd.concat([audit, pd.DataFrame(rows)], ignore_index=True) if rows else audit


def strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value in {"True", "False", "true", "false"}:
        return value.lower() == "true"
    raise LabContractError("CONTRACT_MISMATCH", "clinical_review_required must be an explicit boolean")


def normalize_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    normalized = re.sub(r"\s+", " ", str(value).strip().lower()).replace("non-reactive", "nonreactive")
    return None if normalized in UNKNOWN_CATEGORIES else normalized


def validate_candidate_rows(integrated: pd.DataFrame, candidates: pd.DataFrame,
                            path: object = "Pharma 00 candidates") -> None:
    missing = REQUIRED_COLUMNS - set(candidates)
    if missing:
        raise LabContractError("CONTRACT_MISMATCH", f"{path}: missing v2 columns {sorted(missing)}; regenerate Pharma 00")
    if not candidates.profile_contract_version.eq(PROFILE_VERSION).all():
        raise LabContractError("CONTRACT_MISMATCH", f"{path}: expected {PROFILE_VERSION}")
    stems = candidates.lab
    if (not stems.map(lambda x: isinstance(x, str) and bool(re.fullmatch(r"lab__[a-z0-9]+(?:_[a-z0-9]+)*(?:__[a-z0-9]+(?:_[a-z0-9]+)*)*", x))).all()
            or stems.duplicated().any()):
        raise LabContractError("CONTRACT_MISMATCH", f"{path}: lab stems must be unique canonical lab__ names")
    for row in candidates.itertuples(index=False):
        status = row.graph_candidate_status
        if status not in {"include_numeric", "include_categorical", "review_before_graph"}:
            raise LabContractError("CONTRACT_MISMATCH", f"{path}: invalid candidate status for {row.lab}")
        review = strict_bool(row.clinical_review_required)
        if row.clinically_approved not in {"approved", "pending"}:
            raise LabContractError("CONTRACT_MISMATCH", f"{path}: invalid approval state for {row.lab}")
        if not isinstance(row.eligibility_reason, str) or not row.eligibility_reason.strip():
            raise LabContractError("CONTRACT_MISMATCH", f"{path}: missing eligibility_reason for {row.lab}")
        if pd.isna(row.unit_assessment_status) or not str(row.unit_assessment_status).strip():
            raise LabContractError("CONTRACT_MISMATCH", f"{path}: missing unit assessment for {row.lab}")
        if status == "review_before_graph":
            if row.graph_encoding_hint != "review":
                raise LabContractError("CONTRACT_MISMATCH", f"{path}: review-only hint for {row.lab}")
            continue
        if row.clinically_approved != "approved" or review:
            raise LabContractError("CLINICAL_APPROVAL_PENDING", f"{path}: {row.lab} is not approved without further review")
        numeric = status == "include_numeric"
        suffix, primary, hint = ("__value", "value_numeric", "continuous") if numeric else ("__text", "text_categorical", "binary")
        expected = row.lab + suffix
        field = row.value_column if numeric else row.text_column
        if (row.source_column != expected or field != expected or row.primary_result_source != primary
                or row.graph_encoding_hint != hint or any(token in expected for token in ("ever_positive", "through_episode", "patient_consensus", "future", "next_"))):
            raise LabContractError("CONTRACT_MISMATCH", f"{path}: unauthorized source/encoding for {row.lab}; expected {expected}")
        if expected not in integrated:
            raise LabContractError("MISSING_SOURCE_COLUMN", f"{path}: {expected} absent from integrated Parquet")
        raw = integrated[expected]
        if numeric:
            cast = pd.to_numeric(raw, errors="coerce")
            if (raw.notna() & cast.isna()).any() or np.isinf(cast.to_numpy(dtype=float, na_value=np.nan)).any():
                raise LabContractError("CONTRACT_MISMATCH", f"{path}: {expected} contains nonnumeric or infinite observations; no silent cast")
        else:
            categories = set(raw.map(normalize_category).dropna().unique())
            if categories and not any(categories <= pair for pair in BINARY_PAIRS):
                raise LabContractError("CONTRACT_MISMATCH", f"{path}: {expected} is incompatible with approved binary encoding (values redacted)")


def validate_graph_lab_contract(integrated, candidates, pharma_summary, *,
                                integrated_path, candidates_path):
    dataset = validate_integrated_dataset(integrated)
    validate_candidate_rows(integrated, candidates, candidates_path)
    if pharma_summary.get("profile_contract_version") != PROFILE_VERSION:
        raise LabContractError("CONTRACT_MISMATCH", "Pharma summary must declare pharma_lab_profile_v2, including for empty CSVs")
    hashes = pharma_summary.get("input_hashes")
    if (not isinstance(hashes, list) or not all(isinstance(h, dict) for h in hashes)
            or any(not isinstance(h.get("role"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(h.get("sha256", ""))) for h in hashes)):
        raise LabContractError("CONTRACT_MISMATCH", "Pharma summary lacks input_hashes list")
    if sum(h["role"] == "registry" for h in hashes) != 1:
        raise LabContractError("CONTRACT_MISMATCH", "Pharma summary must retain the clinical registry SHA-256")
    integrated_hashes = [h.get("sha256") for h in hashes if isinstance(h, dict) and h.get("role") == "integrated"]
    if integrated_hashes != [sha256(integrated_path)]:
        raise LabContractError("STALE_PHARMA00_OUTPUT", f"{candidates_path}: summary integrated hash differs from {integrated_path}")
    if pharma_summary.get("context_mode") not in {
            "integrated_only_explicit", "explicit_context", "adjacent_context", "integrated_only_no_companion"}:
        raise LabContractError("CONTRACT_MISMATCH", "Pharma summary must identify the context mode")
    approved = candidates.graph_candidate_status.isin({"include_numeric", "include_categorical"})
    report = {"dataset_contract": dataset, "pharma_contract_version": PROFILE_VERSION,
              "n_approved_labs": int(approved.sum()),
              "n_numeric_approved": int(candidates.graph_candidate_status.eq("include_numeric").sum()),
              "n_categorical_approved": int(candidates.graph_candidate_status.eq("include_categorical").sum()),
              "n_review_only": int(candidates.graph_candidate_status.eq("review_before_graph").sum()),
              "schema_drift": pharma_summary.get("schema_drift", {}),
              "context_mode": pharma_summary.get("context_mode"),
              "release_recommendation": pharma_summary.get("release_recommendation"),
              "registry_input_hashes": [h for h in hashes if h.get("role") == "registry"]}
    return candidates.copy(), report


def validate_release(release: dict | None, candidates: pd.DataFrame, summary: dict,
                     paths: dict[str, Path]) -> None:
    eligible = set(candidates.loc[candidates.graph_candidate_status.isin(
        {"include_numeric", "include_categorical"}), "lab"])
    if not eligible:
        raise LabContractError("NO_APPROVED_LABS", f"No approved assays in {paths['pharma_candidates']}")
    required = ("release_id", "reviewer", "review_date", "review_evidence", "analysis_scope")
    if (not isinstance(release, dict) or release.get("status") != "approved_for_analysis"
            or any(not isinstance(release.get(key), (str, date)) or not str(release[key]).strip() for key in required)):
        raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "Clinical release is pending or lacks reviewer/date/evidence/scope")
    try:
        date.fromisoformat(str(release["review_date"]))
    except ValueError as exc:
        raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "review_date must be an ISO date") from exc
    if release["analysis_scope"] != "graph01_graph02_baseline":
        raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "analysis_scope must cover graph01_graph02_baseline")
    stems = release.get("approved_lab_stems")
    if (not isinstance(stems, list) or not all(isinstance(stem, str) for stem in stems)
            or len(stems) != len(set(stems)) or set(stems) != eligible):
        raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "Release stems must exactly match Pharma 00 approved assays; cannot promote pending assays")
    for role in ("integrated", "pharma_candidates", "pharma_summary", "pharma_expected_audit", "pharma_impact_audit"):
        if release.get(role + "_sha256") != sha256(paths[role]):
            raise LabContractError("INCOMPLETE_RELEASE_REVIEW", f"Release hash mismatch: {paths[role]}")
    if (release.get("acknowledged_release_recommendation") != summary.get("release_recommendation")
            or not summary.get("release_recommendation")
            or release.get("reviewed_expected_lab_audit") is not True
            or release.get("reviewed_downstream_impact") is not True):
        raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "Release must acknowledge original Pharma HOLD and both reviewed audits")
    if summary.get("schema_drift") and release.get("schema_drift_reviewed") is not True:
        raise LabContractError("SCHEMA_DRIFT_REVIEW", "Document review of schema drift; reference column counts are not a universal invariant")


def add_arguments(parser) -> None:
    for flag in ("pharma-profile-summary", "pharma-lab-profile", "pharma-impact-audit", "pharma-expected-audit"):
        parser.add_argument("--" + flag, type=Path, default=None,
                            help="Default: companion file beside --graph-lab-candidates")
    parser.add_argument("--graph-release", type=Path, default=Path(__file__).with_name("graph_lab_release.yaml"))
    parser.add_argument("--preflight-only", action="store_true", help="Write aggregate audits and stop before fitting")
    parser.add_argument("--feature-policy", choices=FEATURE_POLICIES, default=None)
    parser.add_argument("--run-id", default=None, help="Unique output directory ID; existing runs are never overwritten")


def configure_run(args, config: dict, study: str, create_dirs):
    if config.get("contract_version") != STUDY_CONTRACT_VERSION:
        raise LabContractError("CONTRACT_MISMATCH", f"{args.config}: expected {STUDY_CONTRACT_VERSION}")
    policy = args.feature_policy or config.get("feature_policy", "legacy_with_overlap")
    if policy not in FEATURE_POLICIES:
        raise LabContractError("CONTRACT_MISMATCH", "Unknown feature policy")
    args.feature_policy = policy
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise LabContractError("CONTRACT_MISMATCH", "run-id must contain only letters, digits, underscores or hyphens")
    dirs = create_dirs(f"graph/{study}/{policy}/{run_id}")
    # Exclusive marker also protects repeated explicit IDs and concurrent launches.
    marker = dirs["tables"] / ".graph_run"
    try:
        with marker.open("x") as handle:
            handle.write(f"{policy}/{run_id}\n")
    except FileExistsError as exc:
        raise LabContractError("CONTRACT_MISMATCH", f"Output run already exists: {marker.parent}") from exc
    return dirs, {"run_id": run_id, "feature_policy": policy, "input_policy_version": "graph_lab_v2_symmetric_coverage"}


def input_paths(args) -> dict[str, Path]:
    folder = args.graph_lab_candidates.parent
    defaults = {"pharma_profile_summary": "00_pharma_profile_summary.json",
                "pharma_lab_profile": "00_pharma_lab_profile.csv",
                "pharma_impact_audit": "00_pharma_downstream_impact.csv",
                "pharma_expected_audit": "00_pharma_expected_lab_audit.csv"}
    for attr, name in defaults.items():
        if getattr(args, attr, None) is None:
            setattr(args, attr, folder / name)
    paths = {"integrated": args.integrated, "pharma_candidates": args.graph_lab_candidates,
             "pharma_summary": args.pharma_profile_summary, "pharma_profile": args.pharma_lab_profile,
             "pharma_impact_audit": args.pharma_impact_audit, "pharma_expected_audit": args.pharma_expected_audit,
             "baseline_metrics": args.baseline_metrics, "config": args.config}
    for attr, role in (("graph_release", "release"), ("canonical_config", "canonical_config"), ("lab_group_map", "s4_map")):
        path = getattr(args, attr, None)
        if path is not None and (attr != "graph_release" or path.exists()):
            paths[role] = path
    return paths


def read_bundle(args):
    paths = input_paths(args)
    provenance = {role: {"path": str(path.resolve()), "sha256": sha256(path)} for role, path in paths.items()}
    summary = json.loads(paths["pharma_summary"].read_text())
    if not isinstance(summary, dict):
        raise LabContractError("CONTRACT_MISMATCH", f"Expected summary mapping: {paths['pharma_summary']}")
    release = yaml.safe_load(args.graph_release.read_text()) if args.graph_release.is_file() else None
    profile = pd.read_csv(paths["pharma_profile"])
    # Profile is an audit source only. Its projection must agree with candidates;
    # it can never introduce a predictor that Pharma did not export.
    candidates = pd.read_csv(paths["pharma_candidates"])
    if not {"lab", "graph_candidate_status"} <= set(profile):
        raise LabContractError("CONTRACT_MISMATCH", f"Invalid audit profile {paths['pharma_profile']}")
    if profile.lab.duplicated().any():
        raise LabContractError("CONTRACT_MISMATCH", "Duplicate lab in full audit profile")
    projected = profile.loc[profile.graph_candidate_status.isin(
        {"include_numeric", "include_categorical", "review_before_graph"})]
    shared = sorted(set(candidates) & set(profile))
    try:
        pd.testing.assert_frame_equal(projected[shared].sort_values("lab").reset_index(drop=True).fillna("").astype(str),
                                      candidates[shared].sort_values("lab").reset_index(drop=True).fillna("").astype(str))
    except AssertionError as exc:
        raise LabContractError("STALE_PHARMA00_OUTPUT", "Candidates and full profile disagree; regenerate the same Pharma 00 run") from exc
    for role in ("pharma_expected_audit", "pharma_impact_audit"):
        pd.read_csv(paths[role])  # Document must be a readable table, including an empty table with headers.
    return paths, provenance, summary, release, profile


def finish_preflight(args, dirs, prefix, report, provenance, release, candidates,
                     summary, paths, manifest, *, extra_blockers=()):
    report.update(n_excluded_by_graph_coverage=int(manifest.loc[
        manifest.family.eq("lab") & manifest.exclusion_reason.eq("below_minimum_baseline_coverage"), "lab"].nunique()),
        n_lab_features_after_encoding=int((manifest.family.eq("lab") & manifest.included).sum()),
        n_lab_features_selected=int((manifest.family.eq("lab") & manifest.included).sum()),
        graph_release_id=release.get("release_id") if isinstance(release, dict) else None,
        age_dx_temporal_lineage="temporal_lineage_unverified",
        overlap_lineage="BlockA06/overlap_flags: glandular and extraglandular ESSDAI domains; conceptual dependence on Pop requires sensitivity")
    report.update(n_approved_numeric=report["n_numeric_approved"],
                  n_approved_categorical=report["n_categorical_approved"],
                  n_lab_features_baseline_ge70=int((manifest.family.eq("lab") & manifest.pct_nonmissing_baseline.ge(.70)
                      & manifest.n_unique_baseline.ge(2) & manifest.graph_candidate_status.isin({"include_numeric", "include_categorical"})).sum()),
                  canonical_minimum_baseline_coverage=float(yaml.safe_load(paths.get("canonical_config", paths["config"]).read_text())["feature_selection"]["minimum_baseline_coverage"]),
                  n_baseline_patients=report["dataset_contract"]["n_patients_with_baseline"])
    blockers = list(extra_blockers)
    try:
        validate_release(release, candidates, summary, paths)
    except LabContractError as exc:
        blockers.append({"code": exc.code, "reason": str(exc)})
    if not report["n_lab_features_after_encoding"]:
        blockers.append({"code": "NO_LABS_AFTER_COVERAGE", "reason": "No approved, variable LAB survives canonical baseline selection"})
    report.update(status="BLOCKED" if blockers else "READY_FOR_SCIENTIFIC_RUN", blockers=blockers)
    report["input_hashes"] = provenance
    (dirs["tables"] / f"{prefix}_preflight.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (dirs["tables"] / f"{prefix}_input_provenance.json").write_text(json.dumps({**report, "inputs": provenance}, indent=2, default=str) + "\n")
    print(f"{prefix}: {report['status']}; aggregate preflight: {dirs['tables']}")
    if blockers and not args.preflight_only:
        raise LabContractError(blockers[0]["code"], blockers[0]["reason"])
    return report


def write_failure(dirs, prefix, report, exc, args=None):
    code = getattr(exc, "code", "CONTRACT_MISMATCH")
    payload = {**report, "status": "BLOCKED", "blockers": [{"code": code, "reason": str(exc)}]}
    if args is not None:
        payload["inputs"] = {role: {"path": str(path.resolve()),
                                     "sha256": sha256(path) if path.is_file() else None}
                             for role, path in input_paths(args).items()}
    (dirs["tables"] / f"{prefix}_preflight.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    (dirs["tables"] / f"{prefix}_input_provenance.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


# Baseline exploratory contract: clinical gates above are retained verbatim.
from ._feature_catalog import CONTRACT_VERSION, file_hash


def resolve_candidates(path: Path) -> Path:
    # A historical unversioned candidate file must never shadow a fresh run.
    runs = path.parent / "runs"
    found = sorted(runs.glob("*/" + path.name)) if runs.exists() else []
    return found[-1] if found else path


def validate_provenance(integrated: Path, candidates: Path, analysis_mode: str) -> dict:
    summary_path = candidates.with_name("00_pharma_profile_summary.json")
    if not summary_path.is_file():
        raise ValueError("MISSING_PROFILE_PROVENANCE: rerun Pharma 00 with the current contract")
    summary = json.loads(summary_path.read_text())
    if summary.get("feature_contract_version") != CONTRACT_VERSION:
        raise ValueError("STALE_FEATURE_CONTRACT: rerun Pharma 00")
    expected = next((r["sha256"] for r in summary["input_hashes"] if r["role"] == "integrated"), None)
    if expected != file_hash(integrated):
        raise ValueError("PROFILE_SOURCE_HASH_MISMATCH: candidates are from a different master")
    for name in (candidates.name, "00_pharma_graph_lab_selection_audit.csv", summary.get("profile_output_name", "00_pharma_lab_profile.csv")):
        path = candidates.with_name(name)
        if not path.is_file() or summary.get("output_hashes", {}).get(name) != file_hash(path):
            raise ValueError("PROFILE_OUTPUT_HASH_MISMATCH: " + name)
    return {"analysis_mode": analysis_mode, "clinical_validation": "not_established",
            "input_hash": expected, "feature_contract_version": CONTRACT_VERSION,
            "profile_summary_hash": file_hash(summary_path)}


def enforce_lab_mode(manifest: pd.DataFrame, *, analysis_mode: str, allow_without_lab: bool = False,
                     release_path: Path | None = None, integrated_path: Path | None = None,
                     candidates_path: Path | None = None) -> dict:
    selected = manifest.loc[manifest.included & manifest.family.eq("lab")]
    if analysis_mode == "clinical_release":
        release = yaml.safe_load(release_path.read_text()) if release_path and release_path.is_file() else None
        if integrated_path is None or candidates_path is None:
            raise LabContractError("INCOMPLETE_RELEASE_REVIEW", "Clinical release requires the complete input bundle")
        candidates_path = resolve_candidates(candidates_path)
        folder = candidates_path.parent
        summary = json.loads((folder / "00_pharma_profile_summary.json").read_text())
        profile = pd.read_csv(folder / summary.get("profile_output_name", "00_pharma_lab_profile.csv"))
        approved = profile.loc[profile.graph_candidate_status.isin({"include_numeric", "include_categorical"})]
        validate_candidate_rows(pd.read_parquet(integrated_path), approved)
        validate_release(release, approved, summary, {
            "integrated": integrated_path, "pharma_candidates": candidates_path,
            "pharma_summary": folder / "00_pharma_profile_summary.json",
            "pharma_expected_audit": folder / "00_pharma_expected_lab_audit.csv",
            "pharma_impact_audit": folder / "00_pharma_downstream_impact.csv"})
        if selected.empty or not selected.clinical_approval_status.eq("approved").all():
            raise LabContractError("NO_APPROVED_LABS", "No clinically released labs survive selection")
    if selected.empty and not allow_without_lab:
        raise ValueError("NO_TECHNICALLY_ELIGIBLE_LABS: review the selection audit; use --allow-without-lab for an explicit no-LAB sensitivity")
    return {"analysis_mode": analysis_mode,
            "clinical_validation": "released" if analysis_mode == "clinical_release" else "not_established",
            "lab_modality": "present" if len(selected) else "EXPLICIT_NO_LAB_SENSITIVITY",
            "n_selected_lab_features": len(selected)}
