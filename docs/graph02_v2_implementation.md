# Graph02 v2 — implementación y verificación

Base: `e5f475efdd3d35b2393aeb85548390fcf1b69526` de `dasalazarb/obj1_sjd`.

## Cambios por archivo

| Archivo bajo src/studies/graph/ | Cambio |
| --- | --- |
| configs/clinical_domains_02.yaml | 16 fuentes exactas nuevas, release v2 y linaje ocular, salival y SF-36 respaldado por código; aprobaciones pendientes conservadas. |
| configs/experiments_02.yaml | Presupuestos primario/secundario/bootstrap 200/60/100; caché v3; controles de observación y 199 permutaciones. |
| stage02/s02_feature_registry.py | Join por fuente original, auditoría de catálogo y separación de elegibilidad técnica frente a aprobación humana; fechas basales verificadas cuando existe metadato exacto. |
| stage02/s02_representations.py | MFA por constructo/subdominio/dominio, pesos singulares auditables y mínimo de dos dominios activos; R0 usa funciones históricas. |
| stage02/s02_pipeline.py | A5 y ablaciones con refits completos, nuevos controles por defecto, comparación de universos y manifiesto aditivo. |
| stage02/s02_missingness.py | Máscaras deduplicadas, N2a–N2d, M1, clinical_core/dominios, N5/N6, regresión de cobertura con CV y preprocesamiento dentro de cada fold; aliases administrativos exactos. |
| stage02/s02_mapper_engine.py | Auditoría de los cubos y nodos emitidos sin alterar Mapper histórico. |
| stage02/s02_coverage_diagnostics.py | Ruido DBSCAN, soporte mínimo, vecinos, bordes y efectos de completitud/densidad con bootstrap por paciente. |
| stage02/s02_reporting.py | Figuras PNG/PDF, inventario de no estimables, resumen con cifras de ejecución y restricción de exportación por defecto. |
| stage02/s02_contracts.py | Verificación de coherencia de denominadores, evidencia clínica e inventario SHA256; compatibilidad aditiva v1. |
| README.md | Sección Graph02 actualizada. |

Pruebas: se adapta `tests/test_graph_pipeline.py` y se agregan `tests/test_graph_clinical_domain_mapping_v2.py`, `tests/test_graph_missingness_v2.py` y `tests/test_graph_coverage_diagnostics_v2.py`.

Graph01, Pharma00, Step10 y todos los archivos de Graph03 permanecen sin cambios.
El comando público continúa siendo `python src/studies/graph/02_graph_representation_sensitivity.py`.

## Validación

```bash
pytest -q tests/test_graph_pipeline.py tests/test_graph_representation_sensitivity.py tests/test_graph_feature_catalog.py tests/test_graph_bootstrap_qc.py tests/test_graph_clinical_domain_mapping_v2.py tests/test_graph_missingness_v2.py tests/test_graph_coverage_diagnostics_v2.py
```

94 pruebas aprobadas. Se prueban las tres entradas sin flags con sintéticos, lectura de Graph03 sin refit, rechazo por alteración de hashes, conservación del puntero ante fallos, invariancia de R0 y denominadores 69/159. Los presupuestos de pruebas son reducidos; no representan estabilidad final del estudio.

## Límite de datos reales

**No se ejecutó la corrida NIH real**: el entorno no contiene el dataset Step10, el bundle Pharma00 original ni el ZIP del release previo. Estado de esa validación: `NOT_ESTIMABLE / SOURCE_UNAVAILABLE`; n real no evaluado. No se regeneraron fuentes upstream y no se presentan cifras clínicas nuevas ni una comparación before/after real.

Los 159 pacientes, 118 cubiertos y 41 no cubiertos del snapshot histórico provienen de la especificación suministrada; no son resultados esperados fijados en el código.

## Evidencia exclusivamente sintética

Run de prueba: `20261029T192428079984Z_4886a6fc`. Cohorte n=36, columnas R0=22, cubiertos=31. R2: `EXPLORATORY_PROVISIONAL`; fuentes R0 seleccionadas sin mapear=0.

### Experimentos

| experiment_id | status | reason | n_patients | n_common |
| --- | --- | --- | --- | --- |
| S0 | COMPLETE | NA | 36 | NA |
| S0/repeated_subsampling | COMPLETE | NA | 36 | NA |
| S1 | COMPLETE | NA | 36 | NA |
| S1/repeated_subsampling | COMPLETE | NA | 36 | NA |
| S2 | COMPLETE | NA | 36 | NA |
| S2/repeated_subsampling | COMPLETE | NA | 36 | NA |
| S3 | COMPLETE | NA | 36 | NA |
| S4 | COMPLETE | NA | 36 | NA |
| R0 | COMPLETE | NA | 36 | NA |
| R0/repeated_subsampling | COMPLETE | NA | 36 | NA |
| R0/classic_bootstrap | COMPLETE | NA | 36 | NA |
| R1 | COMPLETE | NA | 36 | NA |
| R1/repeated_subsampling | COMPLETE | NA | 36 | NA |
| R2 | COMPLETE | NA | 36 | NA |
| R2/repeated_subsampling | COMPLETE | NA | 36 | NA |
| R2/classic_bootstrap | COMPLETE | NA | 36 | NA |
| R3 | COMPLETE | NA | 36 | NA |
| R3/repeated_subsampling | COMPLETE | NA | 36 | NA |
| R4 | COMPLETE | NA | 36 | NA |
| R4/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A5/context_excluded | COMPLETE | NA | 36 | 30.0 |
| A5/context_excluded/repeated_subsampling | COMPLETE | NA | 36 | NA |
| D1 | COMPLETE | NA | 36 | 31.0 |
| D2 | COMPLETE | NA | 36 | 31.0 |
| D3 | COMPLETE | NA | 36 | 31.0 |
| D4 | COMPLETE | NA | 36 | 31.0 |
| D5 | NOT_ESTIMABLE | OPTIONAL_UMAP_DEPENDENCY_UNAVAILABLE | 36 | NA |
| D6/components_2 | COMPLETE | NA | 36 | 31.0 |
| D6/components_10 | COMPLETE | NA | 36 | 31.0 |
| T/R0/c2/o0.4 | COMPLETE | NA | 36 | 31.0 |
| T/R0/c2/o0.4/repeated_subsampling | COMPLETE | NA | 36 | NA |
| T/R0/c2/o0.5 | COMPLETE | NA | 36 | 31.0 |
| T/R0/c2/o0.5/repeated_subsampling | COMPLETE | NA | 36 | NA |
| T/R0/c3/o0.4 | COMPLETE | NA | 36 | 20.0 |
| T/R0/c3/o0.4/repeated_subsampling | COMPLETE | NA | 36 | NA |
| T/R0/c3/o0.5 | COMPLETE | NA | 36 | 23.0 |
| T/R0/c3/o0.5/repeated_subsampling | COMPLETE | NA | 36 | NA |
| T/local_0 | COMPLETE | NA | 36 | 26.0 |
| T/local_1 | COMPLETE | NA | 36 | 31.0 |
| A2/LAB | COMPLETE | NA | 36 | 30.0 |
| A2/LAB/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A2/SERO | NOT_ESTIMABLE | FAMILY_ABSENT | 36 | NA |
| A2/PRO | COMPLETE | NA | 36 | 30.0 |
| A2/PRO/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A2/EXT | COMPLETE | NA | 36 | 30.0 |
| A2/EXT/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A2/CLINICAL | COMPLETE | NA | 36 | 31.0 |
| A2/CLINICAL/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A1/D01 | COMPLETE | NA | 36 | 30.0 |
| A1/D01/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A1/D02 | COMPLETE | NA | 36 | 31.0 |
| A1/D02/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A1/D03 | NOT_ESTIMABLE | DOMAIN_ABSENT_OR_UNMAPPED | 36 | NA |
| A1/D04 | NOT_ESTIMABLE | DOMAIN_ABSENT_OR_UNMAPPED | 36 | NA |
| A1/D05 | COMPLETE | NA | 36 | 31.0 |
| A1/D05/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A1/D06 | COMPLETE | NA | 36 | 30.0 |
| A1/D06/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A1/D07 | NOT_ESTIMABLE | NO_ELIGIBLE_CLINICAL_DOMAINS | 36 | NA |
| A1/D08 | NOT_ESTIMABLE | CONTEXT_DOMAIN_AUDITED_IN_A5 | 36 | NA |
| A1/D09 | NOT_ESTIMABLE | CONTEXT_DOMAIN_AUDITED_IN_A5 | 36 | NA |
| A6/unstimulated_whole_saliva | COMPLETE | NA | 36 | 30.0 |
| A6/unstimulated_whole_saliva/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A6/ocular_schirmer | COMPLETE | NA | 36 | 30.0 |
| A6/ocular_schirmer/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A6/SF36_INSTRUMENT | COMPLETE | NA | 36 | 30.0 |
| A6/SF36_INSTRUMENT/repeated_subsampling | COMPLETE | NA | 36 | NA |
| A3/unit_valid_complete_panel | NOT_ESTIMABLE | S3 has fewer than two features after selection | 36 | NA |
| A4/diagnostic_features | NOT_ESTIMABLE | NO_APPROVED_DIAGNOSTIC_FEATURE_ROLE_MAP | 36 | NA |
| N1/permutation_0 | COMPLETE | NA | 36 | 27.0 |
| N1/permutation_1 | COMPLETE | NA | 36 | 30.0 |
| N2/missingness_protocol_era | COMPLETE | NA | 36 | 30.0 |
| N3/demo__protocol/stratum_3 | COMPLETE | NA | 36 | NA |
| N3/demo__protocol/stratum_4 | COMPLETE | NA | 36 | NA |
| N3/demo__age_at_baseline/stratum_5 | COMPLETE | NA | 36 | NA |
| N3/demo__age_at_baseline/stratum_6 | COMPLETE | NA | 36 | NA |
| N3/demo__age_at_baseline/stratum_7 | COMPLETE | NA | 36 | NA |
| N3/demo__age_at_baseline/stratum_8 | COMPLETE | NA | 36 | NA |
| N3/demo__sex/stratum_9 | COMPLETE | NA | 36 | NA |
| N3/demo__sex/stratum_10 | COMPLETE | NA | 36 | NA |
| N4/selection_knowledge_date | NOT_ESTIMABLE | NO_APPROVED_BASELINE_SELECTION_KNOWLEDGE_DATE_CONTRAST | 36 | NA |
| N2a/missingness_only | COMPLETE | NA | 36 | 28.0 |
| N2a/missingness_only/repeated_subsampling | COMPLETE | NA | 36 | NA |
| N2b/administrative_only | COMPLETE | NA | 36 | 30.0 |
| N2b/administrative_only/repeated_subsampling | COMPLETE | NA | 36 | NA |
| N2c/missingness_plus_admin | COMPLETE | NA | 36 | 31.0 |
| N2c/missingness_plus_admin/repeated_subsampling | COMPLETE | NA | 36 | NA |
| N2d/domain_missingness_only | COMPLETE | NA | 36 | 26.0 |
| N2d/domain_missingness_only/repeated_subsampling | COMPLETE | NA | 36 | NA |
| M1/value_plus_mask_sensitivity | COMPLETE | NA | 36 | 30.0 |
| M1/value_plus_mask_sensitivity/repeated_subsampling | COMPLETE | NA | 36 | NA |
| N2a/panel_clinical_core | COMPLETE | NA | 36 | 28.0 |
| N2a/panel_D01 | COMPLETE | NA | 36 | 25.0 |
| N2a/panel_D02 | COMPLETE | NA | 36 | 31.0 |
| N2a/panel_D05 | NOT_ESTIMABLE | NO_OBSERVATION_VARIABILITY | 36 | NA |
| N2a/panel_D06 | COMPLETE | NA | 36 | 29.0 |
| N5/mask_row_permutation | COMPLETE | NA | 36 | NA |
| N6/administrative_stratified_permutation | NOT_ESTIMABLE | INSUFFICIENT_LABEL_SUPPORT_OR_MASK_VARIANCE | 36 | NA |
| C/kmedoids/k2 | COMPLETE | NA | 36 | 31.0 |
| C/kmedoids/k3 | COMPLETE | NA | 36 | 31.0 |
| C/agglomerative/k2 | COMPLETE | NA | 36 | 31.0 |
| C/agglomerative/k3 | COMPLETE | NA | 36 | 31.0 |

### Catálogo por dominio

| clinical_domain | n_original_sources | n_encoded_features | n_unique_constructs | n_approved_sources | n_technical_eligible_sources | n_selected_R0 | n_selected_R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D01 | 4 | 4 | 4 | 0 | 4 | 4 | 4 |
| D02 | 4 | 4 | 3 | 0 | 4 | 4 | 4 |
| D05 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| D06 | 11 | 11 | 2 | 0 | 10 | 10 | 10 |
| D07 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| D08 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| D09 | 3 | 4 | 3 | 0 | 0 | 2 | 0 |
| UNMAPPED | 8 | 8 | 0 | 0 | 0 | 0 | 0 |

D02 aparece con cuatro fuentes originales en este fixture; esto verifica el mapeo y no estima su disponibilidad en la cohorte real. Las filas UNMAPPED son fuentes administrativas/metadatos no seleccionadas, auditadas como exclusiones.

### Comparaciones R0/R2/A5 y controles de observación

| experiment_id | n_total | n_covered_reference | n_covered_control | n_common | ari_primary | nmi_primary | comparison_status | comparison_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A5/context_excluded | 36 | 31 | 32 | 30 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| N2a/missingness_only | 36 | 31 | 33 | 28 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| N2b/administrative_only | 36 | 31 | 35 | 30 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| N2c/missingness_plus_admin | 36 | 31 | 36 | 31 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| N2d/domain_missingness_only | 36 | 31 | 33 | 26 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| M1/value_plus_mask_sensitivity | 36 | 31 | 35 | 30 | NA | NA | NOT_ESTIMABLE | INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES |
| R2 | 36 | 31 | 32 | 30 | NA | NA | NA | NA |

ARI/NMI quedan ausentes si solo existe una clase soportada; no se fabrican ceros ni se usa la ausencia de cobertura como clase.

### Estabilidad con presupuesto sintético de dos réplicas

| representation_id | community_id | method | n_requested | n_success | n_evaluable | stability | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 0 | repeated_subsampling | 2 | 2 | 2 | 1.0 | 0.342380227506653 | 1.0 |
| R0 | 0 | classic_bootstrap | 2 | 2 | 2 | 1.0 | 0.342380227506653 | 1.0 |
| R2 | 0 | repeated_subsampling | 2 | 2 | 2 | 1.0 | 0.342380227506653 | 1.0 |
| R2 | 0 | classic_bootstrap | 2 | 2 | 2 | 1.0 | 0.342380227506653 | 1.0 |
| A5/context_excluded | 0 | repeated_subsampling | 2 | 2 | 2 | 1.0 | 0.342380227506653 | 1.0 |

### Missingness global por unidades de observación

| panel | group_type | group | n_patients | n_observation_units | n_cells | n_missing | missing_fraction | n_any_observed | unit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_fixed | global | R0_fixed | 36 | 17 | 612 | 72 | 0.1176470588235294 | 36 | original_source_or_documented_construct | COMPLETE |

### Missingness por familia/dominio/instrumento

| panel | group_type | group | n_patients | n_observation_units | n_cells | n_missing | missing_fraction | n_any_observed | unit | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_fixed | family | EXT | 36 | 7 | 252 | 32 | 0.1269841269841269 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | family | LAB | 36 | 2 | 72 | 0 | 0.0 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | family | PRO | 36 | 8 | 288 | 40 | 0.1388888888888889 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | clinical_domain | D01 | 36 | 4 | 144 | 20 | 0.1388888888888889 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | clinical_domain | D02 | 36 | 3 | 108 | 12 | 0.1111111111111111 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | clinical_domain | D05 | 36 | 1 | 36 | 0 | 0.0 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | clinical_domain | D06 | 36 | 8 | 288 | 40 | 0.1388888888888889 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | clinical_domain | D08 | 36 | 1 | 36 | 0 | 0.0 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | CBC_AND_DIFFERENTIAL | 36 | 1 | 36 | 0 | 0.0 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | LIVER_CHEMISTRY | 36 | 1 | 36 | 0 | 0.0 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | SF36 | 36 | 8 | 288 | 40 | 0.1388888888888889 | 36 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | ocular_oxford | 36 | 1 | 36 | 5 | 0.1388888888888889 | 31 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | ocular_schirmer | 36 | 1 | 36 | 5 | 0.1388888888888889 | 31 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | ocular_tbut | 36 | 1 | 36 | 5 | 0.1388888888888889 | 31 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | ocular_van_bijsterveld | 36 | 1 | 36 | 5 | 0.1388888888888889 | 31 | original_source_or_documented_construct | COMPLETE |
| R0_fixed | instrument_id | salivary_flow | 36 | 3 | 108 | 12 | 0.1111111111111111 | 36 | original_source_or_documented_construct | COMPLETE |

### Razones de cobertura

| coverage_reason | n_patients | n_total | fraction_of_total | status |
| --- | --- | --- | --- | --- |
| covered | 31 | 36 | 0.8611111111111112 | COMPLETE |
| dbscan_noise_in_all_eligible_cubes | 5 | 36 | 0.1388888888888889 | COMPLETE |

### Completitud y soporte geométrico

| metric | status | n_covered | n_uncovered | effect | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| nearest_neighbor_distance | COMPLETE | 31 | 5 | 0.2361005885802128 | 0.2034047668323517 | 0.2743138434050207 |
| knn_mean_distance | COMPLETE | 31 | 5 | 0.2077298732608987 | 0.1801218314317307 | 0.2393176359264929 |
| observed_fraction | COMPLETE | 31 | 5 | 0.06831119544592 | 0.0455407969639468 | 0.0828273244781782 |

## Rutas de resultados y Graph03

En el entorno NIH, el comando sin flags crea automáticamente `outputs/tables/studies/graph/02_graph_representation_sensitivity/runs/<new_run_id>/`, con tablas, `figures/*.png`, `figures/*.pdf`, logs, manifiesto e inventarios. Las rutas exactas de cada corrida constan en su manifiesto y consola. La prueba de integración genera y verifica las mismas salidas en su directorio temporal; no se versionan archivos por paciente.

La compatibilidad técnica del release congelado está probada con sintéticos. Sigue pendiente ejecutar Graph02 con las fuentes NIH originales, revisar sus limitaciones y obtener revisión clínica humana de mapa, temporalidad, unidades y anticircularidad. La estabilidad no equivale a fenotipo validado, progresión temporal, etapa ni etiología. Missingness asociado estadísticamente no identifica MCAR/MAR/MNAR. Los artefactos quedan restringidos por defecto; no se preparó un paquete público de pacientes. Graph03 puede interpretar exploratoriamente un release real solo tras verificar sus hashes, QC, cobertura y roles de descubrimiento/validación.
