# Inventario y migración de Graph03

| Entrada activa | Fuentes | Producto |
|---|---|---|
| `03_graph_community_characterization.py` | Graph02 publicado, Step10, Context y fuentes originales | Snapshot descriptivo N1/N2, contexto Pop, QC, ventana basal descriptiva y auditoría longitudinal |
| `run_graph_study.py` | Graph02 y Graph03 | Resumen integrado del release congelado y su caracterización |

Graph03 conserva el inventario documental y los artefactos de compatibilidad de las fases A/B.
La ventana de 0–6 meses conserva la caracterización descriptiva y el linaje.
Las fases 03c–03g, sus salidas históricas y las instrucciones para recuperar
su versión ejecutable se registran en el [archivo N3](../old/graph03_n3/README.md).

## Módulos

| Módulos en `stage03/` | Cambio |
|---|---|
| `s03_baseline_window.py` | Nuevo contrato, calendario, bins, hash y elección de fecha con conflicto. |
| `s03_n3_exploratory.py`, `s03_n3_exploratory_source.py`, `s03_n3_exploratory_reporting.py`, `s03_n3_exploratory_models.py` | Motor compartido de identidad, procedencia, selección, puertas, sensibilidad y protección de agregados. |
| `s03_pipeline.py`, `s03_n3_validation.py`, `s03_construct_validation.py`, `s03_run_resolution.py` | Publicación descriptiva, inventario de fase A y bloqueo de la inferencia basal. |
| `s03_episode_audit.py`, `s03_longitudinal.py` | Conservan el día original para auditoría y exponen la sensibilidad posterior a la ventana. |
| `s03_n3_phasec_adjudication.py`, `s03_n3_resolution.py` | Contratos y trazabilidad documental actualizados. |
| `s03_n3_auto.py`, `s03_n3_calendar_day.py`, `s03_n3_controlled.py`, `s03_n3_decisions.py`, `s03_n3_approved_view.py` | Versionado de documentos, revisión de soporte y comprobación de la selección actual. |
| `s03_n3_measurement_source.py`, `s03_n3_measurement_review.py`, `s03_n3_source_adapter.py` | Construcción firmada de la selección compartida, auditoría histórica explícita y adaptador Step01/Context con etiqueta de procedencia. |
| `__init__.py`, `s03_characterization.py`, `s03_figure_qc.py`, `s03_incremental_value.py`, `s03_method_context.py`, `s03_n3_candidate_audit.py`, `s03_n3_incremental.py`, `s03_n3_resolution_support.py`, `s03_profiles.py`, `s03_rare_events.py`, `s03_release_reader.py`, `s03_reporting.py`, `s03_statistics.py`, `s03_uncertainty.py` | Conservan algoritmos, contratos de insumo o componentes descriptivos. |

## Ejecución habitual en Biowulf

Desde el checkout institucional, con el entorno Python del estudio activo,
ejecute el preflight y luego Graph03. El preflight verifica el release congelado.

```bash
python src/studies/graph/03_graph_community_characterization.py --preflight-only
python src/studies/graph/03_graph_community_characterization.py
```

`03_graph_baseline_window_qc.json` registra las huellas iniciales y finales
de Graph02 y las incidencias de fuentes. Los CSV de elegibilidad, demora,
perfiles de la ventana, linaje y solapamiento temporal contienen evidencia
agregada bajo la política de divulgación vigente.
