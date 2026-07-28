# Graph Report - .  (2026-07-28)

## Corpus Check
- 12 files · ~37,751 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 590 nodes · 1413 edges · 27 communities (17 shown, 10 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 33 edges (avg confidence: 0.82)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Population Distribution Classification|Population Distribution Classification]]
- [[_COMMUNITY_Comorbidity Prevalence & Survival Analysis|Comorbidity Prevalence & Survival Analysis]]
- [[_COMMUNITY_Serological Profile Analysis|Serological Profile Analysis]]
- [[_COMMUNITY_Visit Spine & Timing|Visit Spine & Timing]]
- [[_COMMUNITY_Glandular Overlap Followup (Visit-anchored)|Glandular Overlap Followup (Visit-anchored)]]
- [[_COMMUNITY_Glandular Overlap Followup (Dx-anchored)|Glandular Overlap Followup (Dx-anchored)]]
- [[_COMMUNITY_PRO Baseline Scoring|PRO Baseline Scoring]]
- [[_COMMUNITY_Integrated Longitudinal Dataset|Integrated Longitudinal Dataset]]
- [[_COMMUNITY_Baseline Table1 Cohort|Baseline Table1 Cohort]]
- [[_COMMUNITY_Glandular Overlap Baseline|Glandular Overlap Baseline]]
- [[_COMMUNITY_ESSDAIESSPRI Activity Scoring|ESSDAI/ESSPRI Activity Scoring]]
- [[_COMMUNITY_PRO Scoring Engine|PRO Scoring Engine]]
- [[_COMMUNITY_Input Data Merging|Input Data Merging]]
- [[_COMMUNITY_Integrated Baseline Profile|Integrated Baseline Profile]]
- [[_COMMUNITY_Shared Overlap-Flag Derivation Logic|Shared Overlap-Flag Derivation Logic]]
- [[_COMMUNITY_Population Transition Analysis|Population Transition Analysis]]
- [[_COMMUNITY_ESSDAI Domain Weights|ESSDAI Domain Weights]]
- [[_COMMUNITY_Derivations Package Init|Derivations Package Init]]
- [[_COMMUNITY_Global Configuration|Global Configuration]]
- [[_COMMUNITY_Analytic Dataset Path|Analytic Dataset Path]]
- [[_COMMUNITY_Integrated Longitudinal Parquet Path|Integrated Longitudinal Parquet Path]]
- [[_COMMUNITY_Overlap Longitudinal Parquet Path|Overlap Longitudinal Parquet Path]]
- [[_COMMUNITY_Pop Longitudinal Parquet Path|Pop Longitudinal Parquet Path]]
- [[_COMMUNITY_PRO Longitudinal Parquet Path|PRO Longitudinal Parquet Path]]
- [[_COMMUNITY_Visit Spine Parquet Path|Visit Spine Parquet Path]]
- [[_COMMUNITY_Population Label Constants|Population Label Constants]]
- [[_COMMUNITY_Project README|Project README]]

## God Nodes (most connected - your core abstractions)
1. `DataFrame` - 39 edges
2. `main()` - 26 edges
3. `main()` - 24 edges
4. `DataFrame` - 24 edges
5. `Series` - 23 edges
6. `main()` - 21 edges
7. `main()` - 18 edges
8. `build_longitudinal_pop_dataset()` - 17 edges
9. `DataFrame` - 17 edges
10. `main()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `ESSDAI_SEVERE=5 clinical threshold` --semantically_similar_to--> `classify_pop()`  [INFERRED] [semantically similar]
  config.py → .graphify/repos/dasalazarb/obj1_sjd/src/block_A/01_pop_distribution.py
- `ESSPRI_THRESHOLD=5 clinical threshold` --semantically_similar_to--> `classify_pop()`  [INFERRED] [semantically similar]
  config.py → .graphify/repos/dasalazarb/obj1_sjd/src/block_A/01_pop_distribution.py
- `ESSDAI_DOMAIN_WEIGHTS (ACR/EULAR 2010)` --semantically_similar_to--> `ESSDAI_DOMAIN_VARS mapping`  [INFERRED] [semantically similar]
  config.py → src/block_A/01_essdai_esspri.py
- `ESSDAI_DOMAIN_WEIGHTS (ACR/EULAR 2010)` --semantically_similar_to--> `DOMAINS glandular/extraglandular mapping`  [INFERRED] [semantically similar]
  config.py → src/block_A/06_overlap_glandular.py
- `add_visit_timing()` --shares_data_with--> `Canonical Patient Visit Spine Artifact`  [INFERRED]
  .graphify/repos/dasalazarb/obj1_sjd/src/derivations/visit_dates.py → tests/test_visit_spine_integration.py

## Import Cycles
- None detected.

## Communities (27 total, 10 thin omitted)

### Community 0 - "Population Distribution Classification"
Cohesion: 0.08
Nodes (74): add_esspri_scenarios(), build_baseline_dataset(), build_by_visit_qc(), build_candidate_ranking(), build_coverage_and_rescue(), build_hierarchical_component(), build_longitudinal_pop_dataset(), build_pop2_pop3_reclassification() (+66 more)

### Community 1 - "Comorbidity Prevalence & Survival Analysis"
Cohesion: 0.10
Nodes (52): apply_fdr(), build_baseline_comorbidity_dataset(), build_new_domain_survival_dataset(), build_severe5_survival_dataset(), calculate_or_and_fisher(), calculate_wilson_ci(), check_upstream_artifacts(), collapse_same_patient_date() (+44 more)

### Community 2 - "Serological Profile Analysis"
Cohesion: 0.09
Nodes (51): _add_reference_lines(), add_table_block(), build_patient_level(), _coalesced_datetime(), _continuous_markers(), _continuous_timeline_point(), _detect_col(), _existing_date_candidates() (+43 more)

### Community 3 - "Visit Spine & Timing"
Cohesion: 0.07
Nodes (39): build_longitudinal_essdai_dataset(), add_spine_timing(), availability(), load(), main(), parse_args(), Use canonical spine when present, otherwise derive compatible timing., write() (+31 more)

### Community 4 - "Glandular Overlap Followup (Visit-anchored)"
Cohesion: 0.13
Nodes (39): Axes, add_time_references(), _any_active(), assign_plot_group(), _bh(), build_patient_summary(), derive_domain_active(), derive_extraglandular_flags() (+31 more)

### Community 5 - "Glandular Overlap Followup (Dx-anchored)"
Cohesion: 0.14
Nodes (39): _any_active(), _bh(), _bool_any(), _bool_evaluable_any(), build_dx_temporal_patient_summary(), collapse_baseline_visits(), derive_domain_active(), derive_extraglandular_flags() (+31 more)

### Community 6 - "PRO Baseline Scoring"
Cohesion: 0.12
Nodes (38): build_baseline_availability_table(), build_baseline_summary_table(), build_manuscript_numbers(), build_missingness(), build_scoring_status(), collapse_patient_visit_duplicates(), derive_parent_protocol(), fmt_num() (+30 more)

### Community 7 - "Integrated Longitudinal Dataset"
Cohesion: 0.11
Nodes (36): assert_unique_keys(), baseline_value(), build_integrated(), canonicalize_patient_id(), compare_shared_column(), coverage(), derive_longitudinal(), _draw_spans() (+28 more)

### Community 8 - "Baseline Table1 Cohort"
Cohesion: 0.13
Nodes (36): add_metric_audit_flags(), apply_eligibility(), build_baseline_patient_table(), build_outputs(), class_is_target_sjd(), coalesce_same_date(), earliest_nonmissing_date(), filter_to_target_sjogren_class_patients() (+28 more)

### Community 9 - "Glandular Overlap Baseline"
Cohesion: 0.17
Nodes (29): add_extraglandular_rollup(), aggregate_binary(), aggregate_composite(), any_positive_composite(), assign_overlap_category(), build_baseline(), build_output_table(), classify_domain_for_group() (+21 more)

### Community 10 - "ESSDAI/ESSPRI Activity Scoring"
Cohesion: 0.15
Nodes (26): coalesce_numeric(), _date_fragments(), derive_domain_activity(), derive_essdai_total(), derive_esspri_total(), ensure_dirs(), extract_visit_year(), _fmt() (+18 more)

### Community 11 - "PRO Scoring Engine"
Cohesion: 0.15
Nodes (24): score_all_pros (baseline), PRO_COLUMNS, _is_missing(), _map_sf36_item(), _numeric_in_range(), Reusable visit-level scoring for patient-reported outcomes (PROs).  The formul, Score observed ESSPRI components; total requires all three observed items., Score current-repository SF-36 domains and norm-based PCS/MCS unchanged. (+16 more)

### Community 12 - "Input Data Merging"
Cohesion: 0.14
Nodes (19): DataFrame, Timestamp, has_included_sjogrens_class(), _is_non_natural_history_visit(), _is_present(), load_filter_export(), merge_matching_visits(), _normalized_patient_id() (+11 more)

### Community 13 - "Integrated Baseline Profile"
Cohesion: 0.28
Nodes (19): correlations(), grouped_summary(), main(), make_figures(), missingness(), normalize_integrated_dtypes(), normalize_status(), overlap_by_pop() (+11 more)

### Community 14 - "Shared Overlap-Flag Derivation Logic"
Cohesion: 0.32
Nodes (17): _any_active(), derive_domain_active(), derive_extraglandular_flags(), derive_glandular_flags(), derive_overlap_flags(), essdai_numeric_to_active(), essdai_ordinal_score(), essdai_string_to_active() (+9 more)

### Community 15 - "Population Transition Analysis"
Cohesion: 0.29
Nodes (13): build_intervals(), load_classification(), main(), pct(), plot_diagram(), plot_heatmap(), plot_sankey(), poisson_ci() (+5 more)

### Community 16 - "ESSDAI Domain Weights"
Cohesion: 0.67
Nodes (3): ESSDAI_DOMAIN_VARS mapping, DOMAINS glandular/extraglandular mapping, ESSDAI_DOMAIN_WEIGHTS (ACR/EULAR 2010)

## Knowledge Gaps
- **32 isolated node(s):** `Path`, `Timestamp`, `Path`, `Namespace`, `callable` (+27 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `add_parsed_visit_dates()` connect `Visit Spine & Timing` to `Population Distribution Classification`, `Comorbidity Prevalence & Survival Analysis`, `Glandular Overlap Followup (Dx-anchored)`, `PRO Baseline Scoring`, `Glandular Overlap Baseline`?**
  _High betweenness centrality (0.294) - this node is a cross-community bridge._
- **Why does `main()` connect `Glandular Overlap Followup (Dx-anchored)` to `Visit Spine & Timing`, `Shared Overlap-Flag Derivation Logic`, `Integrated Longitudinal Dataset`?**
  _High betweenness centrality (0.271) - this node is a cross-community bridge._
- **Why does `main()` connect `Population Distribution Classification` to `Visit Spine & Timing`, `Input Data Merging`?**
  _High betweenness centrality (0.250) - this node is a cross-community bridge._
- **What connects `Centralized repository paths for Sjögren's disease analyses.  All analysis scr`, `Create standard output directories used by analysis scripts.`, `config.py — Configuración global para Objetivo Primario 2 =====================` to the rest of the system?**
  _121 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Population Distribution Classification` be split into smaller, more focused modules?**
  _Cohesion score 0.07747747747747748 - nodes in this community are weakly interconnected._
- **Should `Comorbidity Prevalence & Survival Analysis` be split into smaller, more focused modules?**
  _Cohesion score 0.10304789550072568 - nodes in this community are weakly interconnected._
- **Should `Serological Profile Analysis` be split into smaller, more focused modules?**
  _Cohesion score 0.09276018099547512 - nodes in this community are weakly interconnected._