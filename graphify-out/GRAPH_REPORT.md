# Graph Report - .  (2026-07-29)

## Corpus Check
- Corpus is ~48,966 words - fits in a single context window. You may not need a graph.

## Summary
- 789 nodes · 1844 edges · 40 communities (22 shown, 18 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.86)
- Token cost: 0 input · 511,618 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Population Distribution Classification|Population Distribution Classification]]
- [[_COMMUNITY_Glandular Overlap Followup + Shared Derivation Logic|Glandular Overlap Followup + Shared Derivation Logic]]
- [[_COMMUNITY_Missingness & Observation Process (Block B)|Missingness & Observation Process (Block B)]]
- [[_COMMUNITY_Comorbidity Prevalence & Survival Analysis|Comorbidity Prevalence & Survival Analysis]]
- [[_COMMUNITY_Serological Profile Analysis|Serological Profile Analysis]]
- [[_COMMUNITY_Visit Spine & Timing|Visit Spine & Timing]]
- [[_COMMUNITY_Baseline Table1 Cohort|Baseline Table1 Cohort]]
- [[_COMMUNITY_Longitudinal Multidomain Models (Block B)|Longitudinal Multidomain Models (Block B)]]
- [[_COMMUNITY_PRO Baseline Scoring|PRO Baseline Scoring]]
- [[_COMMUNITY_Transition Mechanisms (Block B)|Transition Mechanisms (Block B)]]
- [[_COMMUNITY_Integrated Longitudinal Dataset|Integrated Longitudinal Dataset]]
- [[_COMMUNITY_PRO Scoring Engine|PRO Scoring Engine]]
- [[_COMMUNITY_Glandular Overlap Baseline|Glandular Overlap Baseline]]
- [[_COMMUNITY_ESSDAIESSPRI Activity Scoring|ESSDAI/ESSPRI Activity Scoring]]
- [[_COMMUNITY_Transition Episode Dataset (Block B)|Transition Episode Dataset (Block B)]]
- [[_COMMUNITY_Input Data Merging|Input Data Merging]]
- [[_COMMUNITY_Multidimensional Baseline Phenotypes (Block B)|Multidimensional Baseline Phenotypes (Block B)]]
- [[_COMMUNITY_Integrated Baseline Profile|Integrated Baseline Profile]]
- [[_COMMUNITY_Population Transition Analysis|Population Transition Analysis]]
- [[_COMMUNITY_Cross-Domain Correlations (Block B)|Cross-Domain Correlations (Block B)]]
- [[_COMMUNITY_Block B Pipeline Overview|Block B Pipeline Overview]]
- [[_COMMUNITY_Derivations Package Init|Derivations Package Init]]
- [[_COMMUNITY_Overlap-Flag Dedup Rationale|Overlap-Flag Dedup Rationale]]
- [[_COMMUNITY_Codebook Path|Codebook Path]]
- [[_COMMUNITY_ESSDAI Domain Weights|ESSDAI Domain Weights]]
- [[_COMMUNITY_PRO Scoring Module Node|PRO Scoring Module Node]]
- [[_COMMUNITY_Visit Dates Module Node|Visit Dates Module Node]]
- [[_COMMUNITY_Test Module Comorbidities|Test Module: Comorbidities]]
- [[_COMMUNITY_Test Module Cross-Domain Correlations|Test Module: Cross-Domain Correlations]]
- [[_COMMUNITY_Test Module Input Data|Test Module: Input Data]]
- [[_COMMUNITY_Test Module Integrated Longitudinal Dataset|Test Module: Integrated Longitudinal Dataset]]
- [[_COMMUNITY_Test Module Longitudinal Multidomain Models|Test Module: Longitudinal Multidomain Models]]
- [[_COMMUNITY_Test Module Missingness & Observation Process|Test Module: Missingness & Observation Process]]
- [[_COMMUNITY_Test Module Multidimensional Baseline Phenotypes|Test Module: Multidimensional Baseline Phenotypes]]
- [[_COMMUNITY_Test Module PRO Scoring|Test Module: PRO Scoring]]
- [[_COMMUNITY_Test Module Transition Episode Dataset|Test Module: Transition Episode Dataset]]
- [[_COMMUNITY_Test Module Transition Mechanisms|Test Module: Transition Mechanisms]]
- [[_COMMUNITY_Test Module Visit Dates|Test Module: Visit Dates]]
- [[_COMMUNITY_Test Module Visit Spine Integration|Test Module: Visit Spine Integration]]

## God Nodes (most connected - your core abstractions)
1. `DataFrame` - 38 edges
2. `main()` - 28 edges
3. `main()` - 24 edges
4. `DataFrame` - 24 edges
5. `DataFrame` - 24 edges
6. `Series` - 22 edges
7. `add_parsed_visit_dates()` - 19 edges
8. `main()` - 18 edges
9. `main()` - 18 edges
10. `main()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `SjD Analysis Project overview` --conceptually_related_to--> `20_missingness_and_observation_process.py (module)`  [INFERRED]
  README.md → src/block_B/20_missingness_and_observation_process.py
- `load_filter_export()` --shares_data_with--> `DEFAULT_ANALYTIC_DATASET path constant`  [INFERRED]
  src/00_input data.py → common.py
- `load_filter_export()` --shares_data_with--> `VISITS_FILE path constant`  [INFERRED]
  src/00_input data.py → config.py
- `main()` --references--> `ESSDAI_SEVERE threshold (5)`  [EXTRACTED]
  src/block_A/07_comorbidities.py → config.py
- `main()` --shares_data_with--> `PROS_LONGITUDINAL_PARQUET path constant`  [EXTRACTED]
  src/block_A/10_build_integrated_longitudinal_dataset.py → common.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pop1/Pop2/Pop3 longitudinal classification pipeline** — common_pop_longitudinal_parquet, block_a_01_pop_distribution_build_longitudinal_pop_dataset, block_a_10_pop_transitions_build_intervals, block_a_10_build_integrated_longitudinal_dataset_build_integrated, block_a_11_integrated_baseline_profile_select_baseline [EXTRACTED 0.95]
- **Three parallel glandular/extraglandular overlap baseline-anchoring strategies (baseline-only, dx-anchored, 1st-visit)** — block_a_06_overlap_glandular_build_baseline, block_a_06_overlap_glandular_followup_select_observed_baseline, block_a_06_overlap_glandular_followup_base_1st_visit_build_patient_summary [INFERRED 0.85]
- **Canonical upstream artifact dependency chain for integration and comorbidity analysis** — common_visit_spine_parquet, common_pop_longitudinal_parquet, common_overlap_longitudinal_parquet, common_pros_longitudinal_parquet, block_a_10_build_integrated_longitudinal_dataset_build_integrated, block_a_07_comorbidities_check_upstream_artifacts [EXTRACTED 0.95]
- **Block B regression models with eligibility-gated graceful degradation** — block_b_20_missingness_and_observation_process_fit_observation_models, block_b_21_multidimensional_baseline_phenotypes_fit_adjusted_models, block_b_22_cross_domain_correlations_fit_ols_model, block_b_24_transition_mechanisms_fit_profile_models, block_b_25_longitudinal_multidomain_models_fit_longitudinal_model [INFERRED 0.85]
- **Canonical Block B script pipeline (20 -> 21 -> 22/23 -> 24, plus 20+21 -> 25)** — block_b_20_missingness_and_observation_process_module, block_b_21_multidimensional_baseline_phenotypes_module, block_b_22_cross_domain_correlations_module, block_b_23_build_transition_episode_dataset_module, block_b_24_transition_mechanisms_module, block_b_25_longitudinal_multidomain_models_module [INFERRED 0.85]
- **PRO instrument scoring/completeness contract shared by producer and consumer** — derivations_pro_scoring_score_esspri_visit, derivations_pro_scoring_score_sf36_visit, derivations_pro_scoring_score_profad_visit, derivations_pro_scoring_score_mdafs_visit, block_b_20_missingness_and_observation_process_derive_observation_states [INFERRED 0.85]

## Communities (40 total, 18 thin omitted)

### Community 0 - "Population Distribution Classification"
Cohesion: 0.08
Nodes (70): add_esspri_scenarios(), build_baseline_dataset(), build_by_visit_qc(), build_candidate_ranking(), build_coverage_and_rescue(), build_hierarchical_component(), build_longitudinal_pop_dataset(), build_pop2_pop3_reclassification() (+62 more)

### Community 1 - "Glandular Overlap Followup + Shared Derivation Logic"
Cohesion: 0.08
Nodes (68): add_time_references(), assign_plot_group(), _bh(), build_patient_summary(), _fisher_exact_p(), main(), make_domain_incident_table(), make_domain_timeline_plot() (+60 more)

### Community 2 - "Missingness & Observation Process (Block B)"
Cohesion: 0.08
Nodes (57): add_time_groups(), build_analytic_cohort_counts(), build_analytic_cohort_membership(), build_availability_heatmap_data(), build_joint_completeness_tables(), build_patient_level_outputs(), classify_binary_sequence(), derive_completeness_indicators() (+49 more)

### Community 3 - "Comorbidity Prevalence & Survival Analysis"
Cohesion: 0.08
Nodes (61): apply_fdr(), build_baseline_comorbidity_dataset(), build_longitudinal_essdai_dataset(), build_new_domain_survival_dataset(), build_severe5_survival_dataset(), calculate_or_and_fisher(), calculate_wilson_ci(), check_upstream_artifacts() (+53 more)

### Community 4 - "Serological Profile Analysis"
Cohesion: 0.09
Nodes (51): _add_reference_lines(), add_table_block(), build_patient_level(), _coalesced_datetime(), _continuous_markers(), _continuous_timeline_point(), _detect_col(), _existing_date_candidates() (+43 more)

### Community 5 - "Visit Spine & Timing"
Cohesion: 0.09
Nodes (35): add_spine_timing(), availability(), load(), main(), parse_args(), Use canonical spine when present, otherwise derive compatible timing., write(), PROS_LONGITUDINAL_PARQUET path constant (+27 more)

### Community 6 - "Baseline Table1 Cohort"
Cohesion: 0.12
Nodes (38): is_missing(), is_not_applicable(), normalize_sex(), add_metric_audit_flags(), apply_eligibility(), build_baseline_patient_table(), build_outputs(), coalesce_same_date() (+30 more)

### Community 7 - "Longitudinal Multidomain Models (Block B)"
Cohesion: 0.13
Nodes (34): add_eligibility_flags(), _attempt_model(), Baseline-Pop circularity / regression-to-the-mean caveat, _bh(), build_outcome_support(), calculate_linear_contrast(), extract_trajectory_estimates(), fit_longitudinal_model() (+26 more)

### Community 8 - "PRO Baseline Scoring"
Cohesion: 0.13
Nodes (36): build_baseline_availability_table(), build_baseline_summary_table(), build_manuscript_numbers(), build_missingness(), build_scoring_status(), collapse_patient_visit_duplicates(), derive_parent_protocol(), fmt_num() (+28 more)

### Community 9 - "Transition Mechanisms (Block B)"
Cohesion: 0.14
Nodes (30): _bool(), build_first_to_last(), classify_transition_family(), compare_transition_scales(), derive_candidate_drivers(), derive_dominant_component(), _domain_column(), _domain_endpoint_column() (+22 more)

### Community 10 - "Integrated Longitudinal Dataset"
Cohesion: 0.13
Nodes (32): assert_unique_keys(), baseline_value(), build_integrated(), canonicalize_patient_id(), compare_shared_column(), coverage(), derive_longitudinal(), _draw_spans() (+24 more)

### Community 11 - "PRO Scoring Engine"
Cohesion: 0.11
Nodes (29): config.py — Configuración global para Objetivo Primario 2 =====================, _is_missing(), _map_sf36_item(), PRO scoring migration-unchanged rationale, _numeric_in_range(), Reusable visit-level scoring for patient-reported outcomes (PROs).  The formul, Score observed ESSPRI components; total requires all three observed items., Score current-repository SF-36 domains and norm-based PCS/MCS unchanged. (+21 more)

### Community 12 - "Glandular Overlap Baseline"
Cohesion: 0.16
Nodes (29): add_extraglandular_rollup(), aggregate_binary(), aggregate_composite(), assign_overlap_category(), build_baseline(), build_output_table(), classify_domain_for_group(), essdai_to_binary() (+21 more)

### Community 13 - "ESSDAI/ESSPRI Activity Scoring"
Cohesion: 0.15
Nodes (26): coalesce_numeric(), _date_fragments(), derive_domain_activity(), derive_essdai_total(), derive_esspri_total(), ensure_dirs(), extract_visit_year(), _fmt() (+18 more)

### Community 14 - "Transition Episode Dataset (Block B)"
Cohesion: 0.18
Nodes (24): add_domain_transitions(), _bool_value(), build_dictionary(), build_summary_tables(), build_transition_episodes(), _direction(), load_inputs(), main() (+16 more)

### Community 15 - "Input Data Merging"
Cohesion: 0.12
Nodes (19): DEFAULT_ANALYTIC_DATASET path constant, VISITS_FILE path constant, has_included_sjogrens_class(), _is_non_natural_history_visit(), _is_present(), load_filter_export(), merge_matching_visits(), _normalized_patient_id() (+11 more)

### Community 16 - "Multidimensional Baseline Phenotypes (Block B)"
Cohesion: 0.16
Nodes (17): build_denominator_table(), build_profile_table(), _denom_text(), fit_adjusted_models(), load_inputs(), main(), make_heatmap(), _mask() (+9 more)

### Community 17 - "Integrated Baseline Profile"
Cohesion: 0.26
Nodes (20): correlations(), grouped_summary(), main(), make_figures(), missingness(), normalize_integrated_dtypes(), normalize_status(), overlap_by_pop() (+12 more)

### Community 18 - "Population Transition Analysis"
Cohesion: 0.20
Nodes (16): build_intervals(), load_classification(), main(), pct(), plot_diagram(), plot_heatmap(), plot_sankey(), poisson_ci() (+8 more)

### Community 19 - "Cross-Domain Correlations (Block B)"
Cohesion: 0.20
Nodes (16): _bh(), build_correlations(), define_ssa_group(), fit_ols_model(), load_baseline(), main(), make_correlation_heatmap(), make_forestplot() (+8 more)

### Community 20 - "Block B Pipeline Overview"
Cohesion: 0.38
Nodes (7): 20_missingness_and_observation_process.py (module), 21_multidimensional_baseline_phenotypes.py (module), 22_cross_domain_correlations.py (module), 23_build_transition_episode_dataset.py (module), 24_transition_mechanisms.py (module), 25_longitudinal_multidomain_models.py (module), SjD Analysis Project overview

## Ambiguous Edges - Review These
- `select_baseline()` → `build_patient_summary()`  [AMBIGUOUS]
  src/block_A/01_essdai_esspri.py · relation: conceptually_related_to
- `select_baseline()` → `select_global_baseline()`  [AMBIGUOUS]
  src/block_A/01_essdai_esspri.py · relation: conceptually_related_to

## Knowledge Gaps
- **49 isolated node(s):** `Path`, `Timestamp`, `Path`, `Namespace`, `callable` (+44 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `select_baseline()` and `build_patient_summary()`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `select_baseline()` and `select_global_baseline()`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `add_parsed_visit_dates()` connect `Visit Spine & Timing` to `Population Distribution Classification`, `Glandular Overlap Followup + Shared Derivation Logic`, `Comorbidity Prevalence & Survival Analysis`, `PRO Baseline Scoring`, `Glandular Overlap Baseline`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `score_all_pros()` connect `PRO Scoring Engine` to `PRO Baseline Scoring`, `Visit Spine & Timing`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `Centralized repository paths for Sjögren's disease analyses.  All analysis scr`, `Create standard output directories used by analysis scripts.`, `config.py — Configuración global para Objetivo Primario 2 =====================` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Population Distribution Classification` be split into smaller, more focused modules?**
  _Cohesion score 0.08249496981891348 - nodes in this community are weakly interconnected._
- **Should `Glandular Overlap Followup + Shared Derivation Logic` be split into smaller, more focused modules?**
  _Cohesion score 0.0772635814889336 - nodes in this community are weakly interconnected._