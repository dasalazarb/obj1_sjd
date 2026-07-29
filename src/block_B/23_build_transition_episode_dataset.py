#!/usr/bin/env python3
"""Build the canonical consecutive-visit transition episode dataset."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa: E402

REQUIRED = ["patient_id", "visit_id", "visit_date", "visit_number", "is_observed_baseline",
            "time_since_observed_baseline_days", "time_since_observed_baseline_years"]
MEASURES = {"essdai": "essdai_total", "esspri": "esspri_total", "pcs": "sf36_pcs",
            "mcs": "sf36_mcs", "profad": "profad_total", "mdafs": "mdafs_global",
            "n_extraglandular_domains": "n_extraglandular_domains_active"}
DOMAINS = [f"eg_{name}_active" for name in ("constitutional", "lymphadenopathy", "articular",
           "cutaneous", "pulmonary", "renal", "muscular", "pns", "cns", "hematologic", "biological")]
OBSERVATION_FIELDS = [f"{x}_observation_state" for x in ("essdai", "esspri", "sf36", "profad", "mdafs", "overlap", "pop")]
OBSERVATION_FLAGS = ["avail_essdai", "avail_esspri_complete", "avail_sf36_complete",
                     "avail_profad_complete", "avail_mdafs_complete", "avail_overlap_evaluable",
                     "avail_pop_classifiable", "pop_core_complete", "pro_core_complete",
                     "systemic_overlap_core_complete", "full_multidomain_complete",
                     "extended_multidomain_complete"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    parser.add_argument("--observation-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "20_observation_process_patient_visit.parquet")
    parser.add_argument("--baseline-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "21_multidimensional_baseline_patient.parquet")
    parser.add_argument("--legacy-interval-input", type=Path, default=common.INTERMEDIATE_DATA_DIR / "10_pop_transition_intervals.parquet")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--overwrite", action="store_true", default=True)
    group.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    return parser.parse_args(argv)


def load_inputs(input_path: Path, observation_path: Path, baseline_path: Path):
    required = [(input_path, "10_build_integrated_longitudinal_dataset.py"),
                (observation_path, "20_missingness_and_observation_process.py"),
                (baseline_path, "21_multidimensional_baseline_phenotypes.py")]
    for path, script in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {path}; run {script} first.")
    return pd.read_parquet(input_path), pd.read_parquet(observation_path), pd.read_parquet(baseline_path)


def validate_visit_data(visits: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in visits]
    if missing:
        raise ValueError(f"Integrated visit data missing required columns: {missing}")
    out = visits.copy()
    if out.duplicated(["patient_id", "visit_id"]).any():
        raise ValueError("Duplicate patient_id + visit_id keys in integrated visit data")
    if out.visit_id.isna().any() or out.visit_id.duplicated().any():
        raise ValueError("visit_id must be present and globally unique")
    parsed = pd.to_datetime(out.visit_date, errors="coerce")
    if parsed.isna().any():
        raise ValueError("visit_date contains missing or unparseable values")
    out["visit_date"] = parsed
    baseline_counts = out.assign(_baseline=out.is_observed_baseline.fillna(False).astype(bool)).groupby("patient_id")._baseline.sum()
    if not baseline_counts.eq(1).all():
        raise ValueError("Every patient must have exactly one observed baseline")
    original = out.sort_values(["patient_id", "visit_number"], kind="stable")
    if original.groupby("patient_id").visit_date.diff().dropna().dt.days.lt(0).any():
        raise ValueError("Patient visit order is not chronological")
    out = out.sort_values(["patient_id", "visit_date", "visit_number"], kind="stable").reset_index(drop=True)
    if pd.to_numeric(out.visit_number, errors="coerce").isna().any():
        raise ValueError("visit_number contains nonnumeric values")
    if out.groupby("patient_id").visit_number.diff().dropna().lt(0).any():
        raise ValueError("visit numbers decrease within chronological patient order")
    intervals = out.groupby("patient_id").visit_date.diff().dt.days.dropna()
    if intervals.le(0).any():
        raise ValueError("Consecutive visits must have positive intervals")
    return out


def merge_visit_completeness(visits: pd.DataFrame, observation: pd.DataFrame) -> pd.DataFrame:
    for key in ("patient_id", "visit_id"):
        if key not in observation:
            raise ValueError(f"Observation-process input missing {key}")
    if observation.duplicated(["patient_id", "visit_id"]).any():
        raise ValueError("Observation-process input has duplicate patient-visit keys")
    keep = ["patient_id", "visit_id"] + [c for c in OBSERVATION_FIELDS + OBSERVATION_FLAGS if c in observation and c not in visits]
    merged = visits.merge(observation[keep], on=["patient_id", "visit_id"], how="left", validate="one_to_one")
    if len(merged) != len(visits):
        raise ValueError("Observation merge changed visit count")
    return merged


def _direction(delta: pd.Series) -> pd.Series:
    return pd.Series(np.select([delta.gt(0), delta.lt(0), delta.eq(0)],
                               ["increased", "decreased", "unchanged"], default="not_evaluable"),
                     index=delta.index, dtype="string")


def build_transition_episodes(visits: pd.DataFrame) -> pd.DataFrame:
    visits = validate_visit_data(visits)
    rows = []
    carry = [c for c in ["pop_status", "overlap_status", *MEASURES.values(), *DOMAINS,
                         *OBSERVATION_FIELDS, *OBSERVATION_FLAGS] if c in visits]
    for patient_id, group in visits.groupby("patient_id", sort=False):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            left, right = group.iloc[i], group.iloc[i + 1]
            row = {"patient_id": patient_id, "from_visit_id": left.visit_id, "to_visit_id": right.visit_id,
                   "from_visit_number": left.visit_number, "to_visit_number": right.visit_number,
                   "from_date": left.visit_date, "to_date": right.visit_date,
                   "from_time_since_baseline_days": left.time_since_observed_baseline_days,
                   "to_time_since_baseline_days": right.time_since_observed_baseline_days,
                   "from_time_since_baseline_years": left.time_since_observed_baseline_years,
                   "to_time_since_baseline_years": right.time_since_observed_baseline_years}
            for column in carry:
                row[f"from_{column}"] = left[column]
                row[f"to_{column}"] = right[column]
            rows.append(row)
    episodes = pd.DataFrame(rows)
    expected = int(visits.groupby("patient_id").size().sub(1).clip(lower=0).sum())
    if len(episodes) != expected:
        raise ValueError(f"Observed episode count {len(episodes)} differs from expected {expected}")
    if episodes.empty:
        columns = ["patient_id", "from_visit_id", "to_visit_id", "from_visit_number", "to_visit_number",
                   "from_date", "to_date", "from_time_since_baseline_days", "to_time_since_baseline_days",
                   "from_time_since_baseline_years", "to_time_since_baseline_years", "episode_id",
                   "interval_days", "interval_years", "episode_midpoint_date",
                   "episode_midpoint_time_since_baseline_years", "episode_number", "n_episodes_patient",
                   "is_first_episode", "is_last_episode"]
        return add_domain_transitions(pd.DataFrame(columns=columns))
    episodes["episode_id"] = (episodes.patient_id.astype(str) + "__" + episodes.from_visit_id.astype(str) + "__" + episodes.to_visit_id.astype(str))
    if episodes.episode_id.duplicated().any():
        raise ValueError("Duplicate episode_id values")
    episodes["interval_days"] = (episodes.to_date - episodes.from_date).dt.days
    if episodes.interval_days.le(0).any():
        raise ValueError("Consecutive visits must have positive intervals")
    episodes["interval_years"] = episodes.interval_days / 365.25
    episodes["episode_midpoint_date"] = episodes.from_date + (episodes.to_date - episodes.from_date) / 2
    episodes["episode_midpoint_time_since_baseline_years"] = (pd.to_numeric(episodes.from_time_since_baseline_years, errors="coerce") + pd.to_numeric(episodes.to_time_since_baseline_years, errors="coerce")) / 2
    episodes["episode_number"] = episodes.groupby("patient_id").cumcount() + 1
    episodes["n_episodes_patient"] = episodes.groupby("patient_id").episode_id.transform("size")
    episodes["is_first_episode"] = episodes.episode_number.eq(1)
    episodes["is_last_episode"] = episodes.episode_number.eq(episodes.n_episodes_patient)
    if "from_pop_status" in episodes:
        episodes = episodes.rename(columns={"from_pop_status": "from_pop", "to_pop_status": "to_pop"})
        valid = episodes.from_pop.isin(["Pop1", "Pop2", "Pop3"]) & episodes.to_pop.isin(["Pop1", "Pop2", "Pop3"])
        episodes["pop_transition_pair"] = episodes.from_pop.astype("string").fillna("Missing") + " -> " + episodes.to_pop.astype("string").fillna("Missing")
        episodes["pop_transition_evaluable"] = valid
        episodes["pop_changed"] = valid & episodes.from_pop.ne(episodes.to_pop)
        episodes["pop_stable"] = valid & episodes.from_pop.eq(episodes.to_pop)
        episodes["transition_involves_unclassifiable"] = ~valid
        episodes["at_risk_for_pop_change"] = valid
        episodes["event_pop_change"] = episodes.pop_changed
        episodes["at_risk_for_transition_to_pop1"] = episodes.from_pop.isin(["Pop2", "Pop3"]) & episodes.to_pop.isin(["Pop1", "Pop2", "Pop3"])
        episodes["event_transition_to_pop1"] = episodes.at_risk_for_transition_to_pop1 & episodes.to_pop.eq("Pop1")
    if "from_overlap_status" in episodes:
        bad = "insufficient|unknown|unclassifiable"
        evaluable = (episodes.from_overlap_status.notna() & episodes.to_overlap_status.notna() &
                     ~episodes.from_overlap_status.astype(str).str.contains(bad, case=False) &
                     ~episodes.to_overlap_status.astype(str).str.contains(bad, case=False))
        episodes["overlap_transition_pair"] = episodes.from_overlap_status.astype("string").fillna("Missing") + " -> " + episodes.to_overlap_status.astype("string").fillna("Missing")
        episodes["overlap_transition_evaluable"] = evaluable
        episodes["overlap_changed"] = evaluable & episodes.from_overlap_status.ne(episodes.to_overlap_status)
    for short, source in MEASURES.items():
        old_from, old_to = f"from_{source}", f"to_{source}"
        if old_from not in episodes:
            continue
        episodes = episodes.rename(columns={old_from: f"from_{short}", old_to: f"to_{short}"})
        episodes[f"delta_{short}"] = pd.to_numeric(episodes[f"to_{short}"], errors="coerce") - pd.to_numeric(episodes[f"from_{short}"], errors="coerce")
        if short != "n_extraglandular_domains":
            episodes[f"{short}_change_direction"] = _direction(episodes[f"delta_{short}"])
    return add_domain_transitions(episodes)


def _bool_value(value) -> Optional[bool]:
    if pd.isna(value): return None
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, float)) and value in (0, 1): return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}: return True
    if text in {"false", "no", "0"}: return False
    return None


def add_domain_transitions(episodes: pd.DataFrame) -> pd.DataFrame:
    out = episodes.copy(); available = [d for d in DOMAINS if f"from_{d}" in out]
    for domain in available:
        transitions = []
        for left, right in zip(out[f"from_{domain}"], out[f"to_{domain}"]):
            a, b = _bool_value(left), _bool_value(right)
            transitions.append("not_evaluable" if a is None or b is None else
                               "incident" if not a and b else "resolved" if a and not b else
                               "persistent_active" if a and b else "persistent_inactive")
        out[f"{domain}_transition"] = pd.Series(transitions, index=out.index, dtype="string")
    transition_cols = {d: f"{d}_transition" for d in available}
    for label in ("incident", "resolved", "persistent_active"):
        out[f"n_domains_{label}"] = sum((out[c].eq(label) for c in transition_cols.values()), start=pd.Series(0, index=out.index))
    out["any_new_extraglandular_domain"] = out.get("n_domains_incident", pd.Series(0, index=out.index)).gt(0)
    out["any_resolved_extraglandular_domain"] = out.get("n_domains_resolved", pd.Series(0, index=out.index)).gt(0)
    for label, column in (("incident", "incident_domain_names"), ("resolved", "resolved_domain_names")):
        out[column] = ["|".join(d.removeprefix("eg_").removesuffix("_active") for d, c in transition_cols.items() if out.at[i, c] == label) for i in out.index]
    domain_eval = pd.Series(False, index=out.index)
    if transition_cols:
        domain_eval = pd.concat([out[c].ne("not_evaluable") for c in transition_cols.values()], axis=1).all(axis=1)
    out["episode_pop_evaluable"] = out.get("pop_transition_evaluable", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["episode_overlap_evaluable"] = out.get("overlap_transition_evaluable", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    pairs = {"essdai_change": "avail_essdai", "esspri_change": "avail_esspri_complete", "sf36_change": "avail_sf36_complete"}
    for name, flag in pairs.items():
        out[f"episode_{name}_evaluable"] = out.get(f"from_{flag}", pd.Series(False, index=out.index)).eq(True) & out.get(f"to_{flag}", pd.Series(False, index=out.index)).eq(True)
    profad = out.get("from_avail_profad_complete", pd.Series(False, index=out.index)).eq(True) & out.get("to_avail_profad_complete", pd.Series(False, index=out.index)).eq(True)
    mdafs = out.get("from_avail_mdafs_complete", pd.Series(False, index=out.index)).eq(True) & out.get("to_avail_mdafs_complete", pd.Series(False, index=out.index)).eq(True)
    out["episode_fatigue_change_evaluable"] = profad | mdafs
    out["episode_domain_change_evaluable"] = domain_eval
    out["episode_full_multidomain_complete"] = out.get("from_full_multidomain_complete", pd.Series(False, index=out.index)).eq(True) & out.get("to_full_multidomain_complete", pd.Series(False, index=out.index)).eq(True)
    return out


def merge_baseline_characteristics(episodes: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in baseline or baseline.patient_id.duplicated().any():
        raise ValueError("Baseline input must have one unique row per patient_id")
    aliases = {"pop_status": "baseline_pop", "overlap_status": "baseline_overlap_status",
               "time_since_diagnosis_years": "baseline_time_since_diagnosis_years"}
    wanted = ["baseline_pop", "baseline_overlap_status", "age_baseline", "sex", "race", "ethnicity",
              "protocol_group", "baseline_time_since_diagnosis_years", "anti_ro_pos", "low_c4",
              "fibromyalgia", "depression", "anxiety", "any_comorbidity", "n_prespecified_comorbidities"]
    source = baseline.rename(columns={k: v for k, v in aliases.items() if v not in baseline}).copy()
    keep = ["patient_id"] + [c for c in wanted if c in source and c not in episodes]
    result = episodes.merge(source[keep], on="patient_id", how="left", validate="many_to_one")
    if len(result) != len(episodes): raise ValueError("Baseline merge changed episode count")
    return result


def build_summary_tables(episodes: pd.DataFrame, visits: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    n = len(episodes); q = episodes.interval_days.quantile([.25, .5, .75]) if n else pd.Series(dtype=float)
    count = lambda c: int(episodes.get(c, pd.Series(False, index=episodes.index)).eq(True).sum())
    summary = pd.DataFrame([{"n_patients_in_integrated_dataset": visits.patient_id.nunique(), "n_patients_with_episodes": episodes.patient_id.nunique(), "n_episodes": n,
        "n_pop_evaluable_episodes": count("episode_pop_evaluable"), "n_overlap_evaluable_episodes": count("episode_overlap_evaluable"),
        "n_essdai_change_evaluable": count("episode_essdai_change_evaluable"), "n_esspri_change_evaluable": count("episode_esspri_change_evaluable"),
        "n_sf36_change_evaluable": count("episode_sf36_change_evaluable"), "n_domain_change_evaluable": count("episode_domain_change_evaluable"),
        "n_full_multidomain_complete_episodes": count("episode_full_multidomain_complete"), "median_interval_days": q.get(.5, np.nan), "q1_interval_days": q.get(.25, np.nan), "q3_interval_days": q.get(.75, np.nan)}])
    pair_rows = []
    if {"from_pop", "to_pop"}.issubset(episodes):
        for (a, b), group in episodes.groupby(["from_pop", "to_pop"], dropna=False):
            denom = int(episodes.loc[episodes.from_pop.eq(a), "episode_pop_evaluable"].sum())
            ne = int(group.episode_pop_evaluable.sum())
            pair_rows.append({"from_pop": a, "to_pop": b, "transition_pair": f"{a} -> {b}", "n_episodes": len(group), "n_unique_patients": group.patient_id.nunique(), "median_interval_days": group.interval_days.median(), "q1_interval_days": group.interval_days.quantile(.25), "q3_interval_days": group.interval_days.quantile(.75), "n_evaluable": ne, "pct_of_evaluable_origin_state_episodes": 100 * ne / denom if denom else np.nan})
    domain_rows = []
    for domain in DOMAINS:
        c = f"{domain}_transition"
        if c not in episodes: continue
        ev = episodes[c].ne("not_evaluable"); denom = int(ev.sum())
        domain_rows.append({"domain": domain.removeprefix("eg_").removesuffix("_active"), "n_evaluable_episodes": denom,
            "n_incident": int(episodes[c].eq("incident").sum()), "pct_incident": 100 * episodes[c].eq("incident").sum() / denom if denom else np.nan,
            "n_resolved": int(episodes[c].eq("resolved").sum()), "pct_resolved": 100 * episodes[c].eq("resolved").sum() / denom if denom else np.nan,
            "n_persistent_active": int(episodes[c].eq("persistent_active").sum()), "n_persistent_inactive": int(episodes[c].eq("persistent_inactive").sum())})
    specs = [("Pop transition", "episode_pop_evaluable", "Both Pop states are Pop1, Pop2, or Pop3"), ("Overlap transition", "episode_overlap_evaluable", "Both upstream overlap states are evaluable"), ("ESSDAI change", "episode_essdai_change_evaluable", "ESSDAI observed at both visits"), ("ESSPRI change", "episode_esspri_change_evaluable", "Complete ESSPRI at both visits"), ("SF-36 change", "episode_sf36_change_evaluable", "Valid SF-36 at both visits"), ("Fatigue change", "episode_fatigue_change_evaluable", "Valid PROFAD or MDAFS at both visits"), ("Domain change", "episode_domain_change_evaluable", "All available domain states evaluable at both visits"), ("Full multidomain change", "episode_full_multidomain_complete", "Full multidomain completeness at both visits")]
    eligibility = pd.DataFrame([{"analysis": label, "eligibility_definition": definition, "n_episodes": int(episodes.get(flag, pd.Series(False, index=episodes.index)).eq(True).sum()), "n_unique_patients": episodes.loc[episodes.get(flag, pd.Series(False, index=episodes.index)).eq(True), "patient_id"].nunique(), "pct_all_episodes": 100 * episodes.get(flag, pd.Series(False, index=episodes.index)).eq(True).sum() / n if n else np.nan} for label, flag, definition in specs])
    return {"summary": summary, "pairs": pd.DataFrame(pair_rows), "domains": pd.DataFrame(domain_rows), "eligibility": eligibility}


def build_dictionary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in episodes:
        group = "identity/timing" if column in {"patient_id", "episode_id"} or "date" in column or "interval" in column or "visit" in column else "eligibility" if column.startswith("episode_") else "domain transition" if column.startswith("eg_") or "domain" in column else "Pop transition" if "pop" in column else "overlap transition" if "overlap" in column else "baseline" if column.startswith("baseline_") or column in {"sex", "race", "ethnicity", "protocol_group"} else "clinical change"
        source = "integrated patient-visit dataset" if not group.startswith("baseline") and group != "eligibility" else "baseline phenotype dataset" if group == "baseline" else "derived from upstream observation flags"
        rows.append({"variable": column, "variable_group": group, "description": column.replace("_", " "), "source": source, "derivation": "Copied at origin/destination or derived as documented by variable name", "allowed_values_or_units": "days/years for timing; boolean for flags; upstream values otherwise"})
    return pd.DataFrame(rows)


def reconcile_legacy_intervals(episodes: pd.DataFrame, path: Path) -> Tuple[Optional[pd.DataFrame], Dict[str, object]]:
    if not path.exists(): return None, {"legacy_reconciliation_performed": False, "legacy_episode_count": None, "legacy_count_difference": None}
    legacy = pd.read_parquet(path); aliases = {"visit_date_clean": "from_date", "transition_pair": "pop_transition_pair"}
    legacy = legacy.rename(columns=aliases)
    keys = [c for c in ("patient_id", "from_visit_number", "to_visit_number") if c in legacy and c in episodes]
    mismatches = []
    if len(keys) == 3:
        compare = episodes.merge(legacy, on=keys, how="outer", suffixes=("_new", "_legacy"), indicator=True)
        fields = ["from_date", "to_date", "from_pop", "to_pop", "interval_days"]
        for _, row in compare.iterrows():
            differences = [f for f in fields if f"{f}_new" in compare and f"{f}_legacy" in compare and not (pd.isna(row[f"{f}_new"]) and pd.isna(row[f"{f}_legacy"])) and str(row[f"{f}_new"]) != str(row[f"{f}_legacy"])]
            if row["_merge"] != "both" or differences: mismatches.append({**{k: row[k] for k in keys}, "match_status": row["_merge"], "mismatched_fields": "|".join(differences)})
    else:
        mismatches.append({"match_status": "legacy_missing_reconciliation_keys", "mismatched_fields": "|".join(keys)})
    return pd.DataFrame(mismatches), {"legacy_reconciliation_performed": True, "legacy_episode_count": len(legacy), "legacy_count_difference": len(episodes) - len(legacy)}


def main(argv=None):
    args = parse_args(argv)
    tables = common.BLOCKB_TABLES_DIR; qc_dir = common.BLOCKB_QC_DIR
    outputs = [common.TRANSITION_EPISODE_PARQUET, common.TRANSITION_EPISODE_CSV, tables / "23_transition_episode_summary.csv", tables / "23_transition_pair_summary.csv", tables / "23_domain_change_summary.csv", tables / "23_episode_analysis_eligibility.csv", tables / "23_transition_episode_dictionary.csv", qc_dir / "23_transition_episode_qc.json", common.BLOCKB_LOGS_DIR / "23_build_transition_episode_dataset.log"]
    if not args.overwrite and any(p.exists() for p in outputs): raise FileExistsError("A planned output exists; use --overwrite")
    common.ensure_output_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(outputs[-1], mode="w"), logging.StreamHandler()])
    raw, observation, baseline = load_inputs(args.input, args.observation_input, args.baseline_input)
    visits = validate_visit_data(raw); warnings = []
    recommended = ["pop_status", "overlap_status", *MEASURES.values(), *DOMAINS]
    warnings.extend(f"Integrated input lacks optional column {c}; corresponding episode fields omitted" for c in recommended if c not in visits)
    visits = merge_visit_completeness(visits, observation)
    episodes = merge_baseline_characteristics(build_transition_episodes(visits), baseline).sort_values(["patient_id", "from_date", "to_date"], kind="stable")
    summaries = build_summary_tables(episodes, visits); reconciliation, legacy_qc = reconcile_legacy_intervals(episodes, args.legacy_interval_input)
    expected = int(visits.groupby("patient_id").size().sub(1).clip(lower=0).sum())
    qc = {"input_path": str(args.input), "observation_input_path": str(args.observation_input), "baseline_input_path": str(args.baseline_input), "n_input_visits": len(visits), "n_input_patients": visits.patient_id.nunique(), "n_patients_with_one_visit": int(visits.groupby("patient_id").size().eq(1).sum()), "n_patients_with_two_or_more_visits": int(visits.groupby("patient_id").size().ge(2).sum()), "expected_episode_count": expected, "observed_episode_count": len(episodes), "n_unique_episode_ids": episodes.episode_id.nunique(), "n_duplicate_episode_ids": int(episodes.episode_id.duplicated().sum()), "n_nonpositive_intervals": int(episodes.interval_days.le(0).sum()), "n_nonconsecutive_visit_number_pairs": int(pd.to_numeric(episodes.to_visit_number).sub(pd.to_numeric(episodes.from_visit_number)).ne(1).sum()), "n_missing_from_visit_ids": int(episodes.from_visit_id.isna().sum()), "n_missing_to_visit_ids": int(episodes.to_visit_id.isna().sum()), "n_pop_evaluable_episodes": int(episodes.episode_pop_evaluable.sum()), "n_pop_changed_episodes": int(episodes.get("pop_changed", pd.Series(False, index=episodes.index)).sum()), "n_transitions_involving_unclassifiable": int(episodes.get("transition_involves_unclassifiable", pd.Series(False, index=episodes.index)).sum()), "n_overlap_evaluable_episodes": int(episodes.episode_overlap_evaluable.sum()), "n_domain_evaluable_episodes": int(episodes.episode_domain_change_evaluable.sum()), **legacy_qc, "warnings": warnings}
    episodes.to_parquet(outputs[0], index=False); episodes.to_csv(outputs[1], index=False)
    for key, filename in (("summary", outputs[2]), ("pairs", outputs[3]), ("domains", outputs[4]), ("eligibility", outputs[5])): summaries[key].to_csv(filename, index=False)
    build_dictionary(episodes).to_csv(outputs[6], index=False); outputs[7].write_text(json.dumps(qc, indent=2, default=str) + "\n")
    if reconciliation is not None: reconciliation.to_csv(qc_dir / "23_legacy_interval_reconciliation.csv", index=False)
    logging.info("Loaded %d visits for %d patients; expected/observed episodes=%d/%d", len(visits), visits.patient_id.nunique(), expected, len(episodes))
    logging.info("Pop-evaluable episodes=%d; domain incidents=%d resolutions=%d", qc["n_pop_evaluable_episodes"], int(episodes.n_domains_incident.sum()), int(episodes.n_domains_resolved.sum()))
    logging.info("Legacy reconciliation performed=%s; outputs written", legacy_qc["legacy_reconciliation_performed"])
    for warning in warnings: logging.warning(warning)


if __name__ == "__main__":
    main()
