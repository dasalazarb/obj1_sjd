"""Patient-series refits and set-based matching, separate from temporal flow."""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from src.studies.longitudinal_graph.validation import KEYS


def patient_draw(raw, rng):
    patients = np.array(sorted(raw.patient_id.unique(), key=str)); draw = rng.choice(patients, len(patients), replace=True)
    chunks = []
    for i, patient in enumerate(draw):
        chunk = raw.loc[raw.patient_id.eq(patient)].copy()
        chunk["original_patient_id"] = chunk.patient_id
        chunk["original_episode_id"] = chunk.clinical_episode_id
        # Copies are independent sample rows, but support still counts ORIGINAL patients.
        chunk["clinical_episode_id"] = [f"bootstrap_copy_{i}_visit_{j}" for j in range(len(chunk))]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True), set(draw)


def structure_sets(scenario, state, kind, common):
    if kind == "component":
        structures = {i: set().union(*(scenario["graph"]["nodes"][n] for n in c))
                      for i, c in enumerate(sorted(nx.connected_components(scenario["raw_nerve"]), key=lambda x: (-len(x), min(x))))}
    else:
        structures = {}
        for node, macro in scenario["node_to_macrostate"].items():
            structures.setdefault(macro, set()).update(scenario["graph"]["nodes"][node])
    sets = {}
    for identity, indices in structures.items():
        rows = state.iloc[sorted(indices)]
        patients = set(rows.patient_id) & common; rows = rows.loc[rows.patient_id.isin(common)]
        episode = "original_episode_id" if "original_episode_id" in rows else "clinical_episode_id"
        sets[identity] = (patients, set(zip(rows.patient_id, rows[episode])))
    return sets


def match_structures(left, right, jaccard):
    rows = []
    pairs = [(a, b, jaccard(lp, rp), jaccard(lv, rv)) for a, (lp, lv) in left.items()
             for b, (rp, rv) in right.items() if lp & rp]
    for a, (lp, lv) in left.items():
        eligible = [p for p in pairs if p[0] == a]
        best = max(eligible, key=lambda p: (p[2], p[3], -p[1])) if eligible else (a, None, 0., 0.)
        rows.append({"primary_structure_id": a, "matched_structure_id": best[1], "patient_jaccard": best[2],
                     "visit_jaccard": best[3], "n_primary_common_patients": len(lp), "n_primary_common_visits": len(lv),
                     "match_status": "matched" if best[1] is not None else "unmatched",
                     "split_candidate": len(eligible)>1,
                     "merge_candidate": sum(p[1] == best[1] for p in pairs)>1 if best[1] is not None else False})
    return rows


def bootstrap_mapper(raw, primary, config, features, fit, prune, jaccard):
    settings = config.get("mapper_validation", {}); reps = int(settings.get("bootstrap_replicates", 0))
    columns = ["replicate", "status", "reason", "n_patient_draws", "n_unique_patients_drawn", "n_visits", "n_nodes",
               "n_connected_components", "n_macrostates", "n_supported_macrostates", "pct_mapper_covered_visits", "n_retained_features", "preprocessing"]
    columns += ["n_common_hard_visits","n_common_hard_patients","n_primary_labels","n_refit_labels","ari","ari_status"]
    match_columns = ["replicate", "structure_kind", "primary_structure_id", "matched_structure_id", "patient_jaccard", "visit_jaccard",
                     "n_primary_common_patients", "n_primary_common_visits", "match_status", "split_candidate", "merge_candidate"]
    if raw is None or any(f not in raw for f in features) or reps == 0:
        reason = "original_feature_observations_unavailable" if raw is None or any(f not in raw for f in features) else "disabled_by_config"
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=match_columns), {"status":"not_applicable", "reason":reason, "replicates_requested":reps}
    rng = np.random.default_rng(config["random_seed"]); results = []; matches = []
    for replicate in range(reps):
        sampled, common = patient_draw(raw, rng)
        row = {"replicate":replicate, "n_patient_draws":raw.patient_id.nunique(), "n_unique_patients_drawn":len(common),
               "n_visits":len(sampled), "preprocessing":"sample_only_median_imputation_pruning_robust_scaler_family_balance_pca_cover_dbscan_support_modularity"}
        try:
            numeric = sampled[features].replace([np.inf,-np.inf],np.nan)
            medians = numeric.median(); usable = medians.index[medians.notna()].tolist()
            filled = numeric[usable].fillna(medians[usable])
            retained, _ = prune(filled, config["representation"]["redundancy_spearman_abs_threshold"])
            state = pd.concat([sampled[KEYS], filled[retained]], axis=1)
            if len(retained)<2: raise ValueError("fewer_than_two_fit_features")
            scenario = fit(state, retained, config, family_balance=True, scenario_name=f"bootstrap-{replicate}")
            row.update({k:scenario["summary"][k] for k in ("n_nodes","n_connected_components","n_macrostates","n_supported_macrostates","pct_mapper_covered_visits")})
            row["status"] = "completed" if scenario["graph"]["nodes"] else "no_nodes"
            row["n_retained_features"] = len(retained)
            ref=primary["visit_membership"]
            sampled_labels=scenario["visit_membership"].copy()
            sampled_labels["clinical_episode_id"]=sampled.original_episode_id.to_numpy()
            groups=sampled_labels.groupby(KEYS,sort=False).hard_macrostate
            consensus=groups.agg(lambda x:x.iloc[0] if x.notna().all() and x.nunique()==1 else pd.NA).reset_index()
            common_hard=ref.loc[ref.hard_macrostate.notna(),KEYS+["hard_macrostate"]].merge(
                consensus.dropna(subset=["hard_macrostate"]),on=KEYS,validate="one_to_one",suffixes=("_reference","_refit"))
            nl=common_hard.hard_macrostate_reference.nunique();nr=common_hard.hard_macrostate_refit.nunique()
            row.update(n_common_hard_visits=len(common_hard),n_common_hard_patients=common_hard.patient_id.nunique(),
                       n_primary_labels=nl,n_refit_labels=nr,
                       ari=float(adjusted_rand_score(common_hard.hard_macrostate_reference.astype(int),common_hard.hard_macrostate_refit.astype(int))) if len(common_hard)>1 else None,
                       ari_status="requires_comparability_review" if min(nl,nr)>1 else "degenerate_or_insufficient_not_stability_evidence")
            for kind in ("component","macrostate"):
                left = structure_sets(primary, raw, kind, common)
                right = structure_sets(scenario, sampled, kind, common)
                matches.extend({"replicate":replicate,"structure_kind":kind,**m} for m in match_structures(left,right,jaccard))
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update(status="failed",reason=f"{type(exc).__name__}:{exc}")
            # Failed refits count as failures, including unmatched reference structures.
            for kind in ("component","macrostate"):
                left = structure_sets(primary, raw, kind, common)
                matches.extend({"replicate":replicate,"structure_kind":kind,**m} for m in match_structures(left,{},jaccard))
        results.append(row)
    table = pd.DataFrame(results,columns=columns); matching = pd.DataFrame(matches,columns=match_columns)
    summary = {"status":"completed", "replicates_requested":reps, "replicates_completed":int(table.status.eq("completed").sum()),
               "replicates_failed":int(table.status.eq("failed").sum()), "replicates_no_nodes":int(table.status.eq("no_nodes").sum()),
               "pct_without_viable_structure":100*float((~table.status.eq("completed")).mean()),
               "resampling_unit":"patient_id_all_visits", "duplicate_draw_support":"unique_original_patients",
               "matching_universe":"reference_and_refit_restricted_to_sampled_original_patients",
               "held_out_projection":"not_run; in_sample_structure_refits_only", "branch_definition":"not_defined; connected_components_are_separate_QC", "stability":[]}
    threshold = float(settings.get("bootstrap_match_jaccard", .5)); frequency = float(settings.get("bootstrap_stability_frequency", .8))
    for (kind, identity), group in matching.groupby(["structure_kind","primary_structure_id"]):
        eligible = group.loc[group.n_primary_common_patients.gt(0)]
        summary["stability"].append({"kind":kind,"primary_structure_id":int(identity),"n_replicates_eligible":len(eligible),
                                     "n_replicates_total":reps,"patient_jaccard_min":eligible.patient_jaccard.min() if len(eligible) else None,
                                     "patient_jaccard_median":eligible.patient_jaccard.median() if len(eligible) else None,
                                     "patient_jaccard_max":eligible.patient_jaccard.max() if len(eligible) else None,
                                     "frequency_jaccard_ge_threshold":float(eligible.patient_jaccard.ge(threshold).mean()) if len(eligible) else None,
                                     "match_threshold":threshold,"required_frequency":frequency,
                                     "stable_by_prespecified_rule":bool(len(eligible) and eligible.patient_jaccard.ge(threshold).mean()>=frequency)})
    return table,matching,summary
