"""Versioned descriptive shapes of observed ESSDAI, separate from ever endpoints.

Only Step 13 supplies scores/endpoints. Never use these retrospective labels as
FROM predictors. Aggregate outputs still require institutional disclosure review.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS, ESSDAI_MAX_VALID

DEFINITION_VERSION = "observed-essdai-pattern-v1"
PATTERNS = ("insufficient_observations", "within_threshold_observed",
            "intermittent_observed", "worsening_threshold_observed_no_reversion",
            "improvement_threshold_observed_no_reversion")
POLICY = "last_canonical_episode_per_date"
PATIENT_COLUMNS = ["patient_id", "threshold_abs", "definition_version", "date_policy",
    "observed_pattern", "baseline_state", "baseline_essdai", "n_points", "n_dates",
    "n_followup_dates", "n_discordant_followup_dates", "followup_years",
    "ever_up", "ever_down", "both_excursions", "up_reversion", "down_reversion",
    "first_up_date", "first_down_date", "n_up_crossings", "n_down_crossings",
    "n_threshold_crossings", "n_direction_changes", "max_delta", "min_delta",
    "last_delta", "last_distance_to_baseline", "last_up", "last_down",
    "directionality_at_last", "can_improve", "can_worsen"]


def build_observed_patterns(df, endpoints, exclude_discordant_dates=False):
    """One row per Level-1 eligible patient × threshold, including insufficient.

    Official baseline is immutable. Only follow-up date ties are reduced. For
    sensitivity discard entire discordant follow-up dates, retaining the original
    patient universe (possibly leaving baseline alone). Zero slopes are ignored
    when counting changes of direction. Crossings count each entry/exit across
    each boundary; a direct +k to -k transition crosses two boundaries.
    """
    ordered = df.sort_values(["patient_id", "clinical_anchor_date", "clinical_episode_id"], kind="stable")
    groups = dict(tuple(ordered.groupby("patient_id", sort=False)))
    rows = []
    for e in endpoints.itertuples():
        g = groups[e.patient_id]
        after = g.loc[g.clinical_anchor_date.gt(e.baseline_date) & g["essdai__total"].notna()]
        discordant = after.groupby("clinical_anchor_date")["essdai__total"].nunique().gt(1)
        if exclude_discordant_dates:
            after = after.loc[~after.clinical_anchor_date.isin(discordant.index[discordant])]
        selected = after.drop_duplicates("clinical_anchor_date", keep="last")
        delta = selected["essdai__total"].to_numpy(dtype=float) - e.baseline_essdai
        steps = np.sign(np.diff(np.r_[0., delta]))
        steps = steps[steps != 0]
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            up, down = delta >= k, delta <= -k
            # A later distinct date must document that the threshold is lost.
            ur = bool((np.maximum.accumulate(up) & ~up).any()) if len(up) else False
            dr = bool((np.maximum.accumulate(down) & ~down).any()) if len(down) else False
            pattern = (PATTERNS[0] if len(delta) < 2 else PATTERNS[1] if not (up.any() or down.any())
                       else PATTERNS[2] if (up.any() and down.any()) or ur or dr
                       else PATTERNS[3] if up.any() else PATTERNS[4])
            u_cross = int(np.count_nonzero(np.diff(np.r_[False, up])))
            d_cross = int(np.count_nonzero(np.diff(np.r_[False, down])))
            last = delta[-1] if len(delta) else np.nan
            dates = selected.clinical_anchor_date
            rows.append(dict(zip(PATIENT_COLUMNS, [
                e.patient_id, k, DEFINITION_VERSION,
                "exclude_discordant_followup_dates" if exclude_discordant_dates else POLICY,
                pattern, e.baseline_state, e.baseline_essdai,
                1 + len(after), 1 + len(selected), len(selected), int(discordant.sum()),
                (dates.iloc[-1] - e.baseline_date).days / 365.25 if len(selected) else 0.,
                bool(up.any()), bool(down.any()), bool(up.any() and down.any()), ur, dr,
                dates.loc[up].iloc[0] if up.any() else pd.NaT,
                dates.loc[down].iloc[0] if down.any() else pd.NaT,
                u_cross, d_cross, u_cross + d_cross,
                int(np.count_nonzero(np.diff(steps))),
                delta.max() if len(delta) else np.nan, delta.min() if len(delta) else np.nan,
                last, abs(last), bool(up[-1]) if len(up) else False,
                bool(down[-1]) if len(down) else False,
                "not_evaluable" if not len(delta) else "up" if up[-1] else "down" if down[-1] else "within_band",
                e.baseline_essdai >= k, e.baseline_essdai + k <= ESSDAI_MAX_VALID])))
    out = pd.DataFrame(rows, columns=PATIENT_COLUMNS)
    for col in ("first_up_date", "first_down_date"):
        out[col] = pd.to_datetime(out[col])
    return out


def verified_baseline_pop(baseline):
    """Consume published Pop only; verify classified rows, never fill unknowns."""
    if "pop__status" not in baseline:
        return None
    if baseline.patient_id.duplicated().any() or not baseline.is_clinical_baseline.eq(True).all():
        raise ValueError("Pop requires unique official Step 11 baselines")
    pop = baseline["pop__status"].astype("string")
    essdai = pd.to_numeric(baseline["essdai__total"], errors="coerce")
    esspri = pd.to_numeric(baseline.get("esspri__total", pd.Series(np.nan, index=baseline.index)), errors="coerce")
    valid = ((pop.eq("Pop1") & essdai.ge(5)) |
             (pop.eq("Pop2") & essdai.lt(5) & esspri.ge(5)) |
             (pop.eq("Pop3") & essdai.lt(5) & esspri.lt(5))).fillna(False)
    if (pop.isin(["Pop1", "Pop2", "Pop3"]) & ~valid).any():
        raise ValueError("Published baseline Pop conflicts with canonical ESSDAI/ESSPRI")
    return pd.Series(pop.where(valid, "Unclassifiable").to_numpy(), index=baseline.patient_id.astype("string"))


def pattern_summary(patterns, stratifier="overall", pop=None):
    """All labels including empty classes; denominators include insufficient."""
    if stratifier == "baseline_state":
        strata = [(s, patterns.baseline_state.eq(s)) for s in ("Low", "Moderate", "High")]
    elif stratifier == "baseline_pop":
        strata = [(s, patterns.patient_id.map(pop).fillna("Unclassifiable").eq(s))
                  for s in ("Pop1", "Pop2", "Pop3", "Unclassifiable")]
    else:
        strata = [("Overall", pd.Series(True, index=patterns.index))]
    rows = []
    for group, mask in strata:
        for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
            g = patterns.loc[mask & patterns.threshold_abs.eq(k)]
            for label in PATTERNS:
                n = int(g.observed_pattern.eq(label).sum())
                rows.append({"definition_version": DEFINITION_VERSION, "stratifier": stratifier,
                    "stratum": group, "threshold_abs": k, "observed_pattern": label,
                    "n_patients": n, "denominator": len(g),
                    "n_pattern_eligible": int(g.n_dates.ge(3).sum()),
                    "pct_patients": 100*n/len(g) if len(g) else np.nan,
                    "support_status": "not_estimable" if not n else "sparse" if n < 10 else "descriptive_only"})
    return pd.DataFrame(rows)


def followup_sensitivity(patterns, excluded):
    rows = []
    duration = pd.cut(patterns.followup_years, [0, 2, 5, np.inf], right=False,
                      labels=("<2y", "2-<5y", ">=5y"))
    subsets = [("dates_ge3", patterns.loc[patterns.n_dates.ge(3)]),
               ("dates_ge4", patterns.loc[patterns.n_dates.ge(4)]),
               ("exclude_discordant_dates", excluded)]
    subsets += [(str(band), patterns.loc[duration.eq(band)]) for band in duration.cat.categories]
    for name, data in subsets:
        for stratifier in ("overall", "baseline_state"):
            table = pattern_summary(data, stratifier)
            table.insert(0, "sensitivity", name)
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def observed_pattern_qc(df, endpoints, patterns, excluded):
    rows = []
    def check(name, failures):
        rows.append({"check": name, "status": "fail" if failures else "pass", "n_failures": int(failures)})
    for label, data, exclusion in (("primary", patterns, False), ("sensitivity", excluded, True)):
        check(label + "_unique_patient_threshold", data.duplicated(["patient_id", "threshold_abs"]).sum())
        check(label + "_universe", abs(len(data) - len(endpoints)*len(ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS)))
        check(label + "_labels", (~data.observed_pattern.isin(PATTERNS)).sum())
        check(label + "_date_eligibility", (data.observed_pattern.eq(PATTERNS[0]) != data.n_dates.lt(3)).sum())
        check(label + "_rebuild", not data.equals(build_observed_patterns(df, endpoints, exclusion)))
        check(label + "_floor", (data.ever_down & ~data.can_improve).sum())
        check(label + "_ceiling", (data.ever_up & ~data.can_worsen).sum())
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        p = patterns.loc[patterns.threshold_abs.eq(k)].set_index("patient_id")
        e = endpoints.set_index("patient_id").reindex(p.index)
        for label, old in (("up", "worsened"), ("down", "improved")):
            difference = p["ever_"+label].ne(e[f"ever_{old}_ge{k}"])
            check(f"ever_{label}_{k}_unexplained_disagreement", (difference & p.n_discordant_followup_dates.eq(0)).sum())
            rows.append({"check": f"ever_{label}_{k}_audited_tie_disagreement", "status": "info",
                         "n_failures": 0, "n_patients": int(difference.sum())})
            check(f"last_{label}_{k}_agreement", p["last_"+label].ne(e[f"last_{old}_ge{k}"]).sum())
    small = patterns.loc[patterns.threshold_abs.eq(3)].set_index("patient_id")
    large = patterns.loc[patterns.threshold_abs.eq(5)].set_index("patient_id")
    for direction in ("up", "down"):
        check("nested_excursions_"+direction, (large["ever_"+direction] & ~small["ever_"+direction]).sum())
    summary = pattern_summary(patterns)
    for k in ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS:
        g = summary.loc[summary.threshold_abs.eq(k)]
        check(f"pattern_partition_{k}", abs(int(g.n_patients.sum()) - len(endpoints)))
    rows.append({"check": "discordant_followup_dates", "status": "info", "n_failures": 0,
                 "n_dates": int(patterns.loc[patterns.threshold_abs.eq(3), "n_discordant_followup_dates"].sum())})
    return pd.DataFrame(rows)


def write_manifest(paths, destination, metadata):
    """Hashes include controlled artifacts, but manifest reveals no row values."""
    base = Path(destination).parents[4]
    artifacts = [{"path": str(Path(p).relative_to(base)),
                  "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]
    destination.write_text(json.dumps({**metadata, "artifacts": artifacts}, indent=2), encoding="utf-8")


def write_observed_pattern_outputs(df, baseline, endpoints, tables, figures, qc, analytic,
                                   input_origin="canonical"):
    patterns = build_observed_patterns(df, endpoints)
    excluded = build_observed_patterns(df, endpoints, True)
    pop = verified_baseline_pop(baseline)
    checks = observed_pattern_qc(df, endpoints, patterns, excluded)
    paths = [qc/"13_essdai_observed_pattern_qc.csv"]
    checks.to_csv(paths[0], index=False)
    if checks.status.eq("fail").any():
        raise ValueError("Observed pattern integrity failure; inspect aggregate QC")
    patient_path = analytic/"13_essdai_patient_observed_patterns.parquet"
    patterns.to_parquet(patient_path, index=False)
    paths.append(patient_path)
    summary = pattern_summary(patterns)
    products = [("observed_pattern_summary", summary),
                ("pattern_by_baseline_state", pattern_summary(patterns, "baseline_state")),
                ("pattern_followup_sensitivity", followup_sensitivity(patterns, excluded))]
    if pop is not None:
        products.append(("pattern_by_pop", pattern_summary(patterns, "baseline_pop", pop)))
    for name, data in products:
        path = tables/f"13_essdai_{name}.csv"
        data.to_csv(path, index=False)
        paths.append(path)
    # Prevent a stale Pop table from a previous run when Pop is unavailable now.
    if pop is None:
        (tables/"13_essdai_pattern_by_pop.csv").unlink(missing_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    labels = ["Insufficient dates", "Within threshold", "Intermittent", "Worsening, no observed reversion", "Improvement, no observed reversion"]
    for ax, k in zip(axes, ESSDAI_ABSOLUTE_CHANGE_THRESHOLDS):
        g = summary.loc[summary.threshold_abs.eq(k)]
        bars = ax.barh(labels, g.n_patients)
        ax.bar_label(bars, padding=3)
        ax.set(title=f"Observed ESSDAI patterns: ±{k}", xlabel="Patients (including insufficient dates)")
        ax.margins(x=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Observed intermittent assessments; no claim of persistence")
    fig.tight_layout()
    path = figures/"13_essdai_observed_pattern_aggregate.pdf"
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    paths.append(path)
    root = Path(__file__).resolve().parents[2]
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write_manifest(paths, qc/"13_essdai_observed_pattern_manifest.json", {
        "definition_version": DEFINITION_VERSION, "input_origin": input_origin,
        "code_commit": code_sha,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "step13_sha256": hashlib.sha256(Path(__file__).with_name("13_disease_activity_progression.py").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((root/"config.py").read_bytes()).hexdigest(),
        "input_frame_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest(),
        "baseline_frame_sha256": hashlib.sha256(pd.util.hash_pandas_object(baseline, index=True).values.tobytes()).hexdigest(),
        "pop_status": "verified" if pop is not None else "unavailable",
        "clinical_review": "PENDING_CLINICAL_REVIEW", "privacy_review": "required_before_disclosure"})
