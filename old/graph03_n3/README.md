# Archivo de Graph03 N3

Las fases 03c–03g se retiraron del flujo activo el 24 de noviembre de 2026.
Investigaban si mediciones independientes podían aportar evidencia sobre las
comunidades congeladas por Graph02. El flujo activo mantiene la caracterización
descriptiva N1/N2, el contexto Pop y los controles de calidad de Graph03.

La última auditoría institucional de 03g concluyó
`AUDIT_COMPLETE_WITH_LIMITATIONS`, con `estimators_run: 0` y
`human_review_required: true`. La evidencia disponible carece de una base
defendible para estimar inferencia N3 bajo el contrato vigente. Sigue abierta
la pregunta por la asociación. La interpretación de R0, R1, R3 y R4 conserva
`PROVISIONAL_COMMUNITIES`. R2 conserva `R2_EXPLORATORY_UNSTABLE`.

## Recuperación

El tag `graph03-n3-prearchive-2026-11-24` apunta al commit
`9516e0e44598358aac00476eb0e6db0df89bcb84`, la última versión con los
scripts en su ubicación original. Para reproducir el flujo histórico, abra
ese tag en un checkout separado dentro del entorno institucional autorizado.
Las copias en [entrypoints/](entrypoints/) conservan el código. Su cálculo de
la raíz y sus imports dependen de la ubicación original.

| Script archivado | Propósito | Directorio histórico bajo `outputs/tables/studies/graph/` |
|---|---|---|
| `03c_graph_n3_clinical_adjudication.py` | Revisión documental de linaje e independencia | `03_graph_n3_clinical_adjudication/` |
| `03d_graph_n3_resolution_preflight.py` | Preflight documental de elegibilidad y soporte | `03_graph_n3_resolution_preflight/` |
| `03e_graph_n3_controlled_inference.py` | Evaluación exploratoria y ruta de inferencia controlada | `03_graph_n3_baseline_0_to_6m/` |
| `03f_build_n3_measurement_source.py` | Auditoría de procedencia y preparación de evidencia de medición | `03_graph_n3_baseline_0_to_6m/` para la auditoría automática |
| `03g_graph_n3_model_readiness_audit.py` | Auditoría de preparación y revisión humana | `03_graph_n3_model_readiness_audit/` |

Los resultados históricos y expedientes firmados permanecen en sus
ubicaciones institucionales bajo las reglas de acceso y retención vigentes.
Este archivo contiene código y documentación.

Guías históricas por fase.
[03c](../../docs/graph03_phasec_endpoint_review_protocol.md) ·
[03d](../../docs/graph03_phase_d_resolution.md) ·
[03e](../../docs/graph03_controlled_inference.md) ·
[03f](../../docs/graph03_n3_measurement_provenance.md) ·
[03g](../../docs/graph03g_model_readiness_audit.md) ·
[día calendario](../../docs/graph03_n3_calendar_day.md).

## Bibliotecas en `stage03/`

- `s03_n3_validation.py`, `s03_n3_candidate_audit.py` y
  `s03_n3_incremental.py` conservan el inventario de fuentes, las reglas de
  independencia y las tablas de compatibilidad con estimación basal bloqueada.
- `s03_n3_exploratory.py` comparte selección de mediciones, huellas del release,
  perfiles de ventana y supresión con Graph03. Sus dependencias incluyen
  `s03_n3_exploratory_source.py`, `s03_n3_exploratory_reporting.py`,
  `s03_n3_exploratory_models.py`, `s03_n3_auto.py`,
  `s03_n3_measurement_source.py` y `s03_n3_calendar_day.py`. Graph03 usa el
  modo descriptivo y conserva las auditorías de soporte y solapamiento.
- `s03_run_resolution.py` mantiene rutas compartidas y resolución histórica.
  Los módulos de adjudicación, resolución, inferencia controlada y auditoría
  de preparación conservan su código y las referencias originales usadas
  para verificar el histórico.

## Reapertura

Cualquier reapertura requiere una nueva rama y evidencia institucional por
representación y desenlace. Los requisitos pendientes incluyen revisión de
independencia respecto de N1/N2, procedencia e identidad de medición, unidad,
fecha específica del instrumento, estabilidad y soporte estadístico, revisión
de divulgación y prespecificación firmada. La semántica de Pop requiere revisión
si se usa como comparador. Las etiquetas inválidas conservan su estado.
Las fechas longitudinales sin verificación mantienen sus bloqueos.

## Regresión

En el flujo activo se conservan las pruebas sintéticas de release, roles,
denominadores, Pop, privacidad, figuras y elegibilidad longitudinal. Las pruebas
que ejecutan los CLI archivados, la inferencia controlada, el flujo firmado
automático o la auditoría 03g indican el tag de recuperación con `pytest.mark.skip`. Las pruebas de
bibliotecas que ejercitan contratos compartidos siguen en el árbol activo.
Los comandos originales de esas pruebas se recuperan en el tag junto a sus
rutas de código y configuraciones.

Ejecución con información institucional: `NOT_VERIFIED_WITH_REAL_DATA`.
