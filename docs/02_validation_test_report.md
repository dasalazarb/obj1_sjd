# Validación de la revisión metodológica del Script 02

Fecha: 2026-10-12. Python 3.12; entorno de la sesión
`/workspace/.venvs/obj1_sjd/bin/python`. Fixtures sintéticos; semilla 182 para
fuentes de prueba y semilla configurada 20260831 para Mapper/bootstrap.

## Resultado ejecutivo

**65 tests específicos pasan**: 37 del pipeline existente, 19 de la revisión
y nueve del arranque de CLI. Estos últimos cubren:
invocación sin flags, ID único, descubrimiento de archivo/sello, ambigüedad,
selección explícita, IDs inválidos, sello con archivo ausente y dry-run sin escritura.
La prueba end-to-end conserva la referencia, ejecuta ocho candidatos S4 y registra
dos alternativas no aplicables por listas clínicas ausentes. También comprueba
una reproducción paralela con informe de diff y referencia intacta.

El bootstrap reducido de CI usa B=2; el control negativo usa una réplica y B=1
por escenario para verificar el procedimiento. Es evidencia de funcionamiento
de software, **no** evidencia clínica de estabilidad con B=200 ni de H1.

La suite general no está verde por problemas **preexistentes**. La ejecución sin
exclusiones se detiene durante colección por dos módulos que ya faltaban en el
checkout base. Al excluir sólo esas dos colecciones: **417 passed, 10 failed,
1 skipped**. Los diez nombres de tests fallidos coinciden exactamente con el
log inicial `/workspace/setup/obj1_sjd/pytest.log`; ninguno pertenece a
`longitudinal_graph`. No se repararon módulos ajenos al alcance de esta revisión.

## Comandos y evidencia

```bash
python -m pytest tests/test_longitudinal_graph.py tests/test_longitudinal_methodology_review.py tests/test_longitudinal_mapper_cli.py -q
# 65 passed; 51.66 s. Incluye los módulos agrupados bajo _review/.

python -m pytest tests/test_longitudinal_methodology_review.py::test_candidate_end_to_end_preserves_reference_and_blocks_inference -q
# 1 passed; incluye reproducción paralela y diff, 20.94 s.

python -m pytest -q
# 2 errores de colección por archivos ya ausentes.

python -m pytest -q \
  --ignore=tests/test_btris_unmapped_lab_audit.py \
  --ignore=tests/test_pharma.py
# 417 passed, 10 failed, 1 skipped; 174.85 s.
```

Los módulos ausentes son `src/20_btris_visit_date_match_report.py` y
`src/studies/pharma/01_run_pharma.py`. Los fallos preexistentes afectan
comorbidities, cross_domain_correlations, missingness_observation_process,
pharma_lab_profile, pharma_main_plots, pros_longitudinal_labels y visit_dates.
Los logs completos y JUnit se conservaron en el directorio de trabajo de la sesión;
los resultados clínicos no se generaron ni se publicaron.

El dry-run real, sin archivos de cohorte, devuelve:

```json
{"status":"blocked","reason":"G0_frozen_reference_unavailable","outputs_written":false}
```

## Matriz de controles solicitados

| Control | Estado | Evidencia / límite |
| --- | --- | --- |
| 1. S3-primary read-only | PASS | Inventario idéntico antes/después; rutas históricas, overwrite y tampering rechazados; staging fallido no se publica |
| 2. Metadatos de referencia 159/497/98 | PASS fixture | Valores congelados del documento en fixture; no se afirman como datos ejecutados |
| 3. Partición exclusiva de visitas | PASS | Fixture 346 + 29 + 14 + 108 = 497; categorías de paciente separadas |
| 4. Grafo completo/filtrado | PASS sintético | 8/5 aristas, sum(degrees)=16, componentes=3, histogramas, endpoints y solapamiento real. Archivo clínico: SKIP por ausencia |
| 5. Jaccard/membresías múltiples | PASS | Jaccard en [0,1], matching uno-a-uno, comunidades solapadas y denominadores separados |
| 6. Remuestreo paciente | PASS | Todas las visitas de cada draw; soporte cuenta pacientes originales; separación train/test por paciente |
| 7. Equal patient mass | PASS | Masa igual en fit; copias bootstrap conservan masa por draw sin inflar soporte; tolerancia 1e-12 |
| 8. Missingness ≠ cero | PASS | Ceros válidos y NA distintos; procedencia, carry-forward y ausencia estructural unknown si no hay máscara/registro causal |
| 9. Fit train-only | PASS | Medianas/bounds/selección no cambian al transformar test extremo; PCA/Mapper se ajustan dentro de cada refit |
| 10. Circularidad | PASS | Derivado de input → descriptive_only_circular; no etiqueta independiente |
| 11. Admin ausente ≠ negativo | PASS | missing_source + G5 inconclusive/blocked; alias de protocolo trazado; calendario continuo sin bins |
| 12. Sensibilidad no elegible | PASS | Whitelists ausentes → not_applicable con razón; sin ARI inventado |
| 13. ARI single-label | PASS | ARI/NMI null para particiones degeneradas; universo y exclusiones conservados |
| 14. Consistencia temporal | PASS | Rechaza paciente diferente, duplicados, duración/orden inválidos y saltos de visita |
| 15. Gate Script 03 | PASS | Bloquea antes de escribir; aprobación vinculada a hashes; preview aislado/rotulado conserva gate oficial |
| 16. Reproducibilidad | PASS sintético | Registro idéntico, mismo seed/config genera grafo/embedding iguales; manifiestos incluyen hashes; timestamps/run IDs difieren por diseño |
| 17. Proyección OOS | SKIP científico | No se implementó proyección. PASS de declaración `held_out_projection=not_run`; no se reclama generalización |
| 18. Controles negativos | PASS sintético | Cambia covariación de X por columnas/bloques paciente; conserva keys y distribuciones marginales; fit y bootstrap se vuelven a ejecutar |

## Experimentos disponibles

La tabla completa está en `02_synthetic_scenario_comparison.csv`. Todos los
experimentos son **SYNTHETIC_FIXTURE_NOT_CLINICAL_COHORT**: 40 pacientes ficticios,
80 visitas, ocho features y 40 intervalos. Las columnas de estabilidad usan B=2,
insuficiente para validar un escenario científico. No se seleccionó ninguna
representación por ARI, cobertura, número de comunidades ni significación.

| Escenario sintético | Features | Visitas cubiertas | Intervalos grafo / soft / hard | Componentes / comunidades | Gate |
| --- | ---: | ---: | --- | --- | --- |
| S4-visit-comparator | 8 | 78.75% | 24 / 24 / 10 de 40 | 1 / 3 | blocked |
| S4-hscrp-log-robust | 8 | 61.25% | 16 / 13 / 11 de 40 | 2 / 4 | blocked |
| S4-hscrp-winsor | 8 | 76.25% | 25 / 25 / 15 de 40 | 1 / 3 | blocked |
| S4-no-hscrp | 7 | 62.50% | 13 / 9 / 9 de 40 | 2 / 5 | blocked |
| S4-patient-mass | 8 | 77.50% | 24 / 21 / 16 de 40 | 1 / 3 | blocked |
| S4-no-pro | 7 | 68.75% | 20 / 18 / 13 de 40 | 3 / 4 | blocked |
| S4-preimputation-pruning | 8 | 78.75% | 24 / 24 / 10 de 40 | 1 / 3 | blocked |
| S4-full80 | 8 | 78.75% | 24 / 24 / 10 de 40 | 1 / 3 | blocked |
| S4-pro-summary / S4-common-features | — | not_applicable | Whitelists no definidas/aprobadas | — | blocked |

Ramas: no definidas en todos los escenarios. Administrativo: inconclusive, T3
no probado. OOS: not_run. El fixture verifica que se calculan y se distinguen
los universos, no demuestra eficacia en Sjögren.

## Integridad y entregables

La referencia clínica solicitada no está disponible; su SHA antes/después es
**null**, no un hash inventado. Config histórico conserva
`352d8fc0e4be06a04a7a0535e3abda6549d1b2b9bd5e8725b9bb7edd5f02e841`.
El inventario de la referencia sintética conservado antes/después tiene SHA-256
`a4d8cc9ab9eae5121d78043f840e724cc4a2fd02866304eb3b7a0a8f50343f68`.
Esto no sustituye la auditoría clínica. Ver `02_reference_integrity.json`.

Auxiliares internos en `_review/`: `reference.py`, `graph_export.py`, `scenarios.py`,
`review_qc.py`, `review_runner.py`, `temporal_gate.py` y `review_cli.py`. La ejecución
sigue 01→02→03→04; estos módulos no son pasos adicionales. Se añadieron la política
`methodology_review.yaml` y las suites `test_longitudinal_methodology_review.py` y
`test_longitudinal_mapper_cli.py`. Se extendieron CLI de 02/03, bootstrap
y publicación; Script 01, Script 04 y config histórico no se alteraron.

Los bundles nuevos generan inventarios/hashes, comparación y crosswalk de
escenarios, ambos grafos y figuras individuales PC1/PC2/hsCRP, QC raw/redundancia/dominancia, tablas largas de
membresía, cobertura/SMD/IC, contratos de intervalos, administración/circularidad,
caracterización descriptiva, bootstrap/control negativo y selección/readiness.
Los artefactos clínicos quedan en almacenamiento restringido y nunca en CI pública.

## Estado científico y downstream

**Script 03: blocked. No existe aprobación científica registrada.** Falta G0
clínico en este entorno; no hay definición de ramas/H1, adaptación T3 aprobada,
validación independiente/circularidad mitigada, revisión de sesgo de selección ni
proyección OOS. Ninguno de los resultados sintéticos responde esas decisiones.
La implementación entrega diagnóstico y un paquete revisable; acepta resultados
nulos o un continuo sin declarar fenotipos nuevos.
