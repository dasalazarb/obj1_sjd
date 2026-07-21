"""Reusable visit-level scoring for patient-reported outcomes (PROs).

The formulas, item recoding, completeness rules, normative values, and PCS/MCS
coefficients are migrated unchanged from ``09_pros_baseline.py``.  Scores are
computed independently for every input row; callers control visit selection.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

ESSPRI_COMPONENTS = {
    "dryness": "esspri_questionnaire__dryness",
    "fatigue": "esspri_questionnaire__fatigue",
    "pain": "esspri_questionnaire__pain",
}

SF36_ITEMS = [f"sf-36_health_survey__sf36_q{i}" for i in range(1, 26)] + [
    "sf-36_health_survey__sf_q26"
] + [f"sf-36_health_survey__sf36_q{i}" for i in range(27, 37)]
SF36_MEASURES = {
    "sf36_physical_functioning": "Physical Functioning",
    "sf36_role_physical": "Role Physical",
    "sf36_bodily_pain": "Bodily Pain",
    "sf36_general_health": "General Health",
    "sf36_vitality": "Vitality",
    "sf36_social_functioning": "Social Functioning",
    "sf36_role_emotional": "Role Emotional",
    "sf36_mental_health": "Mental Health",
    "sf36_pcs": "Physical Component Summary",
    "sf36_mcs": "Mental Component Summary",
}

SF36_DOMAIN_ITEMS = {
    "sf36_physical_functioning": [f"sf-36_health_survey__sf36_q{i}" for i in range(3, 13)],
    "sf36_role_physical": [f"sf-36_health_survey__sf36_q{i}" for i in range(13, 17)],
    "sf36_bodily_pain": ["sf-36_health_survey__sf36_q21", "sf-36_health_survey__sf36_q22"],
    "sf36_general_health": ["sf-36_health_survey__sf36_q1", "sf-36_health_survey__sf36_q33", "sf-36_health_survey__sf36_q34", "sf-36_health_survey__sf36_q35", "sf-36_health_survey__sf36_q36"],
    "sf36_vitality": ["sf-36_health_survey__sf36_q23", "sf-36_health_survey__sf36_q27", "sf-36_health_survey__sf36_q29", "sf-36_health_survey__sf36_q31"],
    "sf36_social_functioning": ["sf-36_health_survey__sf36_q20", "sf-36_health_survey__sf36_q32"],
    "sf36_role_emotional": [f"sf-36_health_survey__sf36_q{i}" for i in range(17, 20)],
    "sf36_mental_health": ["sf-36_health_survey__sf36_q24", "sf-36_health_survey__sf_q26", "sf-36_health_survey__sf36_q25", "sf-36_health_survey__sf36_q28", "sf-36_health_survey__sf36_q30"],
}
SF36_NORM_MEANS = {"sf36_physical_functioning": 84.52404, "sf36_role_physical": 81.19907, "sf36_bodily_pain": 75.49196, "sf36_general_health": 72.21316, "sf36_vitality": 61.05453, "sf36_social_functioning": 83.59753, "sf36_role_emotional": 81.29467, "sf36_mental_health": 74.84212}
SF36_NORM_SDS = {"sf36_physical_functioning": 22.89490, "sf36_role_physical": 33.79729, "sf36_bodily_pain": 23.55879, "sf36_general_health": 20.16964, "sf36_vitality": 20.86942, "sf36_social_functioning": 22.37642, "sf36_role_emotional": 33.02717, "sf36_mental_health": 18.01189}
SF36_PCS_COEFF = {"sf36_physical_functioning": 0.42402, "sf36_role_physical": 0.35119, "sf36_bodily_pain": 0.31754, "sf36_general_health": 0.24954, "sf36_vitality": 0.02877, "sf36_social_functioning": -0.00753, "sf36_role_emotional": -0.19206, "sf36_mental_health": -0.22069}
SF36_MCS_COEFF = {"sf36_physical_functioning": -0.22999, "sf36_role_physical": -0.12329, "sf36_bodily_pain": -0.09731, "sf36_general_health": -0.01571, "sf36_vitality": 0.23534, "sf36_social_functioning": 0.26876, "sf36_role_emotional": 0.43407, "sf36_mental_health": 0.48581}

PROFAD_ITEMS = [
    "profile_of_fatigue_and_discomfort__profad_need_rest",
    "profile_of_fatigue_and_discomfort__profad_get_going",
    "profile_of_fatigue_and_discomfort__profad_keep_going",
    "profile_of_fatigue_and_discomfort__profad_weak",
    "profile_of_fatigue_and_discomfort__profad_think_clear",
    "profile_of_fatigue_and_discomfort__profad_forget_things",
    "profile_of_fatigue_and_discomfort__profad_limb_discomfort",
    "profile_of_fatigue_and_discomfort__profad_finger_wrist_discomfort",
    "profile_of_fatigue_and_discomfort__profad_cold_hands",
    "profile_of_fatigue_and_discomfort__profad_skin_dry_itchy",
    "profile_of_fatigue_and_discomfort__profad_vaginal_dry",
    "profile_of_fatigue_and_discomfort__profad_eyes_sore",
    "profile_of_fatigue_and_discomfort__profad_eye_irritation",
    "profile_of_fatigue_and_discomfort__profad_vision_poor",
    "profile_of_fatigue_and_discomfort__profad_eating_diff",
    "profile_of_fatigue_and_discomfort__profad_throat_nose_dry",
    "profile_of_fatigue_and_discomfort__profad_breath_bad",
    "profile_of_fatigue_and_discomfort__profad_mouth_fluid_wet",
    "profile_of_fatigue_and_discomfort__profad_mouth_prob_othr",
]
MDAFS_ITEMS = [f"multidimensional_assessment_of_fatigue_scale__fat_q{i}" for i in range(1, 17)]
MDAFS_ACTIVITY_FLAGS = [
    "multidimensional_assessment_of_fatigue_scale__fat_q4_dont_do_activity",
    "multidimensional_assessment_of_fatigue_scale__fat_q5_dont_do_activity",
] + [f"multidimensional_assessment_of_fatigue_scale__fat_q{i}_no_actvty" for i in range(6, 15)]

MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "null", "unknown", "not available", "missing"}
NOT_VALIDATED = "scoring_algorithm_not_validated"


def _is_missing(value: Any) -> bool:
    return pd.isna(value) or str(value).strip().lower() in MISSING_STRINGS


def _numeric_in_range(df: pd.DataFrame, columns: list[str], minimum: float, maximum: float, instrument: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    numeric = pd.DataFrame(index=df.index)
    violations: list[dict[str, Any]] = []
    for col in columns:
        raw = df[col] if col in df else pd.Series(np.nan, index=df.index)
        values = pd.to_numeric(raw, errors="coerce")
        bad = raw.notna() & (~raw.map(_is_missing)) & (values.isna() | values.lt(minimum) | values.gt(maximum))
        for idx in df.index[bad]:
            violations.append({"patient_id": df.at[idx, "patient_id"] if "patient_id" in df else np.nan,
                               "visit_date": df.at[idx, "visit_date"] if "visit_date" in df else pd.NaT,
                               "instrument": instrument, "variable": col, "raw_value": raw.loc[idx],
                               "expected_min": minimum, "expected_max": maximum, "action_taken": "set_missing"})
        numeric[col] = values.where(values.between(minimum, maximum))
    return numeric, violations


def _with_violations(df: pd.DataFrame, violations: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    previous = list(out.attrs.get("pro_range_violations", []))
    out.attrs["pro_range_violations"] = previous + violations
    return out


def score_esspri_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Score observed ESSPRI components; total requires all three observed items."""
    out = df.copy(); violations: list[dict[str, Any]] = []
    component_scores = []
    for name, column in ESSPRI_COMPONENTS.items():
        numeric, found = _numeric_in_range(out, [column], 0, 10, "ESSPRI")
        violations.extend(found)
        target = f"esspri_{name}"; out[target] = numeric[column]; component_scores.append(target)
    out["esspri_n_components"] = out[component_scores].notna().sum(axis=1)
    out["esspri_n_components_available"] = out["esspri_n_components"]
    out["esspri_total"] = out[component_scores].mean(axis=1).where(out["esspri_n_components"].eq(3))
    # Retained baseline sensitivity only; never used to impute the primary total.
    out["esspri_partial_mean"] = out[component_scores].mean(axis=1).where(out["esspri_n_components"].ge(2))
    out["esspri_scoring_valid"] = out["esspri_total"].notna()
    out["esspri_scoring_version"] = "observed_components_only"
    return _with_violations(out, violations)


def _map_sf36_item(col: str, value: Any) -> float:
    if pd.isna(value): return np.nan
    try: value = int(float(value))
    except (TypeError, ValueError): return np.nan
    positive5={1:100,2:75,3:50,4:25,5:0}; negative5={1:0,2:25,3:50,4:75,5:100}
    positive6={1:100,2:80,3:60,4:40,5:20,6:0}; negative6={1:0,2:20,3:40,4:60,5:80,6:100}
    if col in [f"sf-36_health_survey__sf36_q{i}" for i in range(3,13)]: return {1:0,2:50,3:100}.get(value,np.nan)
    if col in [f"sf-36_health_survey__sf36_q{i}" for i in range(13,20)]: return {1:0,2:100}.get(value,np.nan)
    if col in {"sf-36_health_survey__sf36_q1","sf-36_health_survey__sf36_q20","sf-36_health_survey__sf36_q22","sf-36_health_survey__sf36_q34","sf-36_health_survey__sf36_q36"}: return positive5.get(value,np.nan)
    if col in {"sf-36_health_survey__sf36_q21","sf-36_health_survey__sf36_q23","sf-36_health_survey__sf_q26","sf-36_health_survey__sf36_q27","sf-36_health_survey__sf36_q30"}: return positive6.get(value,np.nan)
    if col in {"sf-36_health_survey__sf36_q24","sf-36_health_survey__sf36_q25","sf-36_health_survey__sf36_q28","sf-36_health_survey__sf36_q29","sf-36_health_survey__sf36_q31"}: return negative6.get(value,np.nan)
    if col in {"sf-36_health_survey__sf36_q32","sf-36_health_survey__sf36_q33","sf-36_health_survey__sf36_q35"}: return negative5.get(value,np.nan)
    return np.nan


def score_sf36_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Score current-repository SF-36 domains and norm-based PCS/MCS unchanged."""
    out=df.copy(); expected={c:(1,6) for c in SF36_ITEMS}
    for c in [f"sf-36_health_survey__sf36_q{i}" for i in range(3,20)]: expected[c]=(1,3) if int(c.rsplit("q",1)[1]) < 13 else (1,2)
    for c in ["sf-36_health_survey__sf36_q1","sf-36_health_survey__sf36_q2","sf-36_health_survey__sf36_q20","sf-36_health_survey__sf36_q22","sf-36_health_survey__sf36_q32","sf-36_health_survey__sf36_q33","sf-36_health_survey__sf36_q34","sf-36_health_survey__sf36_q35","sf-36_health_survey__sf36_q36"]: expected[c]=(1,5)
    scored=pd.DataFrame(index=out.index); violations=[]
    for col in SF36_ITEMS:
        values, found=_numeric_in_range(out,[col],*expected[col],"SF-36"); violations.extend(found)
        scored[col]=values[col].map(lambda v, c=col: _map_sf36_item(c,v))
    for measure, items in SF36_DOMAIN_ITEMS.items():
        answered=scored[items].notna().sum(axis=1); out[measure]=scored[items].mean(axis=1).where(answered.ge(int(np.ceil(len(items)/2))))
        out[f"{measure}_n_items_expected"]=len(items); out[f"{measure}_n_items_answered"]=answered
        out[f"{measure}_scoring_status"]=np.where(out[measure].notna(),"scored_validated","insufficient_items")
    complete=out[list(SF36_DOMAIN_ITEMS)].notna().all(axis=1)
    z={m:(out[m]-SF36_NORM_MEANS[m])/SF36_NORM_SDS[m] for m in SF36_DOMAIN_ITEMS}
    out["sf36_pcs"]=(50+10*sum(z[m]*SF36_PCS_COEFF[m] for m in SF36_DOMAIN_ITEMS)).where(complete)
    out["sf36_mcs"]=(50+10*sum(z[m]*SF36_MCS_COEFF[m] for m in SF36_DOMAIN_ITEMS)).where(complete)
    out["sf36_pcs_scoring_status"]=np.where(out.sf36_pcs.notna(),"scored_validated","insufficient_items")
    out["sf36_mcs_scoring_status"]=np.where(out.sf36_mcs.notna(),"scored_validated","insufficient_items")
    out["sf36_n_items_available"]=scored.notna().sum(axis=1); out["sf36_n_domains_available"]=out[list(SF36_DOMAIN_ITEMS)].notna().sum(axis=1)
    out["sf36_scoring_valid"]=out[["sf36_pcs","sf36_mcs"]].notna().all(axis=1)
    out["sf36_scoring_version"]="SF36_v1_current_repo_algorithm"; out["sf36_norm_reference"]="1998 US norm means, SDs, and current repo coefficients"
    return _with_violations(out, violations)


def score_profad_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Score the existing PROFAD total only; no unvalidated subscales are created."""
    out=df.copy(); numeric, violations=_numeric_in_range(out,PROFAD_ITEMS,0,7,"PROFAD"); answered=numeric.notna().sum(axis=1)
    out["profad_total"]=numeric.mean(axis=1).where(answered.ge(int(np.ceil(len(PROFAD_ITEMS)/2))))
    out["profad_n_items_answered"]=answered; out["profad_n_items_available"]=answered
    out["profad_scoring_status"]=np.where(out.profad_total.notna(),"scored_validated","insufficient_items")
    out["profad_scoring_valid"]=out.profad_total.notna(); out["profad_scoring_version"]="current_repo_19_item_mean"
    return _with_violations(out, violations)


def score_mdafs_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Score the existing MAF/MDAFS global fatigue index without new activity rules."""
    out=df.copy(); q1_14=[f"multidimensional_assessment_of_fatigue_scale__fat_q{i}" for i in range(1,15)]; q15_16=[f"multidimensional_assessment_of_fatigue_scale__fat_q{i}" for i in range(15,17)]
    first,v1=_numeric_in_range(out,q1_14,1,10,"MDAFS"); last,v2=_numeric_in_range(out,q15_16,0,4,"MDAFS")
    activity=[f"multidimensional_assessment_of_fatigue_scale__fat_q{i}" for i in range(4,15)]; answered=first[activity].notna().sum(axis=1); q15=last[q15_16[0]]*2.5
    required=first[[f"multidimensional_assessment_of_fatigue_scale__fat_q{i}" for i in range(1,4)]].notna().all(axis=1)&q15.notna()&answered.ge(6)
    out["mdafs_global"]=(first[q1_14[0]]+first[q1_14[1]]+first[q1_14[2]]+first[activity].mean(axis=1)+q15).where(required)
    out["mdafs_n_activity_items_answered"]=answered; out["mdafs_n_items_available"]=pd.concat([first,last],axis=1).notna().sum(axis=1)
    out["mdafs_scoring_status"]=np.where(out.mdafs_global.notna(),"scored_validated","insufficient_items"); out["mdafs_scoring_valid"]=out.mdafs_global.notna(); out["mdafs_scoring_version"]="current_repo_maf_global_index"
    return _with_violations(out, v1+v2)


def score_all_pros(df: pd.DataFrame) -> pd.DataFrame:
    """Add all supported visit-level PRO scores in the established order."""
    for scorer in (score_esspri_visit, score_sf36_visit, score_profad_visit, score_mdafs_visit): df=scorer(df)
    return df
