# Migración del contrato público de Step 10 (v2)

Step 10 es la única frontera que cambia nombres. Los productos upstream conservan
su esquema. `clinical_episode_curated_v2` publica namespaces clínicos estables y
registra por separado `source`, `producer_script`, `original_variable`,
`clinical_domain`, `temporal_scope` y `public_variable`.

Ejemplos de migración: `ids__interval_name` → `spine__interval_name`,
`ids__age_at_visit` → `demo__age_at_visit`, `age_baseline` →
`demo__age_at_baseline`, `sex` → `demo__sex`, `pop_status` → `pop__status`,
`essdai_total` → `essdai__total`, `esspri_total` → `esspri__total`,
`sf36_pcs` → `pro__sf36_pcs`, `crp__value` → `lab__crp__value` y
`anti_ro_ssa__ever_positive_through_episode` →
`sero__anti_ro_ssa__ever_positive_through_episode`.

`10_variable_name_mapping.csv` es el mapeo autoritativo generado desde el mismo
registry que construye los Parquet. Los consumidores legacy usan temporalmente
`src.integrated_schema.add_legacy_aliases`: los aliases existen sólo en memoria y
se retirarán cuando deje de admitirse `clinical_episode_curated_v1`; nunca se
escriben en el master ni en context.

La auditoría `10_semantic_redundancy_audit.csv` contiene sólo conteos agregados.
No equipara edad de visita con edad basal, texto con valor de laboratorio,
serología basal con historia acumulada, ni HLA as-of con consenso retrospectivo.
La ausencia de `ids__interval_name` se registra y no se reconstruye desde fechas.
