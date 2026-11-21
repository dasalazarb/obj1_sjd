# Inventario y migración de Graph03

| Entrada | Fuentes | Ventana y producto | Consumidores |
|---|---|---|---|
| `03_graph_community_characterization.py` | Graph02 publicado, Step10, Context y fuentes originales | Snapshot descriptivo N1/N2, caracterización 0–6 meses, QC, longitudinal original y sensibilidad posterior | Resumen del estudio, Phase C, Phase D y 03e |
| `03c_graph_n3_clinical_adjudication.py` | Inventario de Phase A, roles y release Graph02 | Revisión documental del snapshot y requisitos de la ventana actual | Phase D y revisión clínica |
| `03d_graph_n3_resolution_preflight.py` | Phase A, Phase C y release Graph02 | Preflight documental de fuentes, Pop, cronología y puertas estadísticas | Revisión clínica, propuesta de evidencia |
| `03e_graph_n3_controlled_inference.py` | Graph02, Step10, EDA, BTRIS y, para la ruta controlada, artefacto y decisiones firmados | Caracterización 0–6 meses por defecto, salida controlada bajo plan explícito | Reporte agregado 03e y revisión institucional |
| `03f_build_n3_measurement_source.py` | EDA, BTRIS, Step01, Step10, Context y release | Auditoría agregada por defecto. Construcción de evidencia privada con flags | 03e controlado y auto firmado |
| `run_graph_study.py` | Orquestador de todas las etapas | Sigue siendo una entrada completa de descubrimiento y caracterización | Resumen del estudio |

El checkout no contiene entradas 03a/03b. Las fases A y B se ejecutan desde Graph03 principal. A conserva el inventario del snapshot. B conserva sus archivos y devuelve un bloqueo explícito para la inferencia basada en el snapshot. La caracterización e inferencia de la ventana se publican en los nuevos contratos. Los comandos de Phase C y D siguen ejecutando diagnósticos sin estimadores. La ruta 03e usa una carpeta propia `03_graph_n3_baseline_0_to_6m`. Los runs previos y las revisiones firmadas quedan accesibles en sus ubicaciones históricas. Los planes y prespecificaciones anteriores requieren revisión y firma nuevas. Las claves privadas de auto incluyen la versión y hash de la política. El checkout no contiene un job Slurm de Graph03 para migrar.

## Módulos

| Módulos en `stage03/` | Cambio |
|---|---|
| `s03_baseline_window.py` | Nuevo contrato, calendario, bins, hash y elección de fecha con conflicto. |
| `s03_n3_exploratory.py`, `s03_n3_exploratory_source.py`, `s03_n3_exploratory_reporting.py`, `s03_n3_exploratory_models.py` | Motor compartido de identidad, procedencia, selección, puertas, sensibilidad y protección de agregados. |
| `s03_pipeline.py`, `s03_n3_validation.py`, `s03_run_resolution.py` | Consumo del motor, publicación diferenciada, inventario Phase A, bloqueo de Phase B sobre el snapshot y nueva ruta 03e. |
| `s03_episode_audit.py`, `s03_longitudinal.py` | Conservan el día original para auditoría y exponen la sensibilidad posterior a la ventana. |
| `s03_n3_phasec_adjudication.py`, `s03_n3_resolution.py` | Contratos y trazabilidad documental actualizados. |
| `s03_n3_auto.py`, `s03_n3_calendar_day.py`, `s03_n3_controlled.py`, `s03_n3_decisions.py`, `s03_n3_approved_view.py` | Versionado de documentos, revisión de soporte y comprobación de la selección actual. |
| `s03_n3_measurement_source.py`, `s03_n3_measurement_review.py`, `s03_n3_source_adapter.py` | Construcción firmada de la selección compartida, auditoría histórica explícita y adaptador Step01/Context con etiqueta de procedencia. |
| `__init__.py`, `s03_characterization.py`, `s03_construct_validation.py`, `s03_figure_qc.py`, `s03_incremental_value.py`, `s03_method_context.py`, `s03_n3_candidate_audit.py`, `s03_n3_incremental.py`, `s03_n3_resolution_support.py`, `s03_profiles.py`, `s03_rare_events.py`, `s03_release_reader.py`, `s03_reporting.py`, `s03_statistics.py`, `s03_uncertainty.py` | Conservan algoritmos, contratos de insumo o componentes descriptivos. |

## Ejecución habitual en Biowulf

Desde el checkout institucional, con el entorno Python del estudio activo, ejecute primero el preflight y luego las dos entradas independientes. Las rutas habituales de salida e insumos están en `configs/pipeline.yaml` y `configs/n3_auto.yaml`. El preflight verifica el release sin refit de Mapper.

```bash
python src/studies/graph/03_graph_community_characterization.py --preflight-only
python src/studies/graph/03_graph_community_characterization.py
python src/studies/graph/03e_graph_n3_controlled_inference.py
```

Consulte `03_graph_baseline_window_qc.json` para las huellas iniciales y finales de Graph02 y las incidencias de fuentes. `03_graph_baseline_eligibility.csv`, `03_graph_baseline_measurement_delay.csv`, `03_graph_n3_baseline_community_profiles.csv`, `03_graph_n3_baseline_inference.csv`, `03_graph_n3_role_lineage_audit.csv` y `03_graph_temporal_overlap_audit.csv` muestran la evidencia agregada. Los archivos `03e_*.csv` conservan los perfiles auxiliares, sensibilidades y razones específicas por representación. En 03e el solapamiento longitudinal remite a la auditoría de Graph03 principal. El comando `03f_build_n3_measurement_source.py` produce una auditoría de procedencia sin publicar filas clínicas.
