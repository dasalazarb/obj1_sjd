# Graph03 C/D: ejecución autónoma y resolución de bloqueos

## Cambio mínimo y auditoría arquitectónica

Base revisada: `cafb5f0` (main, 2026-11-05). El checkout inicial estaba en
`583bac2`; se actualizó antes de modificar Graph03 porque Phase D ya existía
upstream. No se crearon fases E/F ni se cambiaron Graph02, Mapper o EDA.

Árbol relevante: los entrypoints C/D llaman `run_phasec` / `run_resolution`;
D reutiliza `load_phasec_inputs`, los contratos de Graph02 y
`support_diagnostics`. `s03_n3_validation` produce Phase A con
`s03_n3_candidate_audit`; `s03_n3_incremental` ya implementa modelos revisados
sobre casos completos comunes y no necesita otro estimador en Phase D.

La fricción estaba en flags obligatorios, YAMLs con runs/commits/hashes de una
sola ejecución y la política que bloqueaba cualquier cambio documental del
código. Esos pins no son contratos científicos. En cambio, los SHA almacenados
en manifests/inventarios de datos, esquemas, claves, membresías, denominadores y
referencias entre runs sí protegen contra corrupción o mezcla de cohortes.
Se conservaron, junto con las restricciones de divulgación y revisión humana.

| Piezas reutilizadas | Uso |
|---|---|
| `common.PROJECT_ROOT` / raíz del entrypoint; `pipeline.yaml`, `config_path` | Rutas existentes; `OBJ1_SJD_ROOT` solo cambia la raíz de datos/outputs |
| `validate_release`, `safe_artifact`, `load_phasec_inputs` | Integridad, esquemas, claves, pertenencia y denominadores |
| `hard_members`, `observed_values`, `support_diagnostics` | Universo congelado y soporte A/B/C/D; sensibilidad R0 estable |
| `review_record`, `candidate_evidence_matrix`, `human_review_queue` | Evidencia firmada y adjudicación humana |
| Nueva `graph_paths` | Adapta las rutas del pipeline existente a la raíz de despliegue |
| Nueva `resolve_latest_compatible` | Único selector, parametrizado por dependencia, archivos y validador |
| Nueva `resolve_graph03_dependencies` | Vincula pares/tríos usando el selector; no mezcla ramas |
| Nueva `reviewed_pop_values` | Una normalización de Pop, únicamente con equivalencia humana firmada |
| Duplicación eliminada | Pins de producción en ambos YAML, recálculos SHA dentro del lector C y segunda plantilla de revisión idéntica de D |

Los cambios funcionales están en los dos entrypoints, sus dos YAML, los lectores
C/D, el pequeño módulo compartido `s03_run_resolution.py`, las pruebas existentes
y `test_graph03_run_resolution.py`. El workflow de Graph incluye ahora la regresión N3. El README y este documento describen el
contrato vigente. Los outputs históricos permanecen intactos.

## Un comando por fase

Desde la raíz, con las dependencias Python del proyecto instaladas:

```bash
python src/studies/graph/03c_graph_n3_clinical_adjudication.py
python src/studies/graph/03d_graph_n3_resolution_preflight.py
```

También funcionan con una ruta absoluta al script desde otro directorio. La
raíz se obtiene del código; `OBJ1_SJD_ROOT` permite otro despliegue de datos sin
copiar código ni crear configuraciones paralelas. Los outputs Graph02/Phase A
se toman de `configs/pipeline.yaml`. Las colecciones nuevas son, bajo
`outputs/tables/studies/graph/`:

- `03_graph_n3_clinical_adjudication/runs/<nuevo_id>`
- `03_graph_n3_resolution_preflight/runs/<nuevo_id>`

Se exploran exclusivamente los directorios `runs/` de esas colecciones y las
colecciones Graph02/Phase A configuradas. Una Phase C histórica fuera de ellas
puede seleccionarse con `--phasec-run`; no se busca por todo el disco.

Todos los flags son opcionales: `--graph03-run`, `--graph02-release`,
`--phasec-run` (D), `--output-dir` y `--config`. El override de output conserva
su significado histórico: **directorio exacto nuevo**, no una colección.
Ningún override evita la validación. Sin candidatos, se explica dónde se buscó,
los descartes y qué script upstream ejecutar. `--verbose` permite traceback.
`--preflight-only` de D se conserva por compatibilidad; D siempre es preflight.

El selector ordena por el timestamp UTC del run ID canónico
`YYYYMMDDTHHMMSSffffffZ_<8 hex>`; para IDs históricos usa `created_at_utc` con
zona horaria. Mtime solo desempata fechas iguales. Un ID sin fecha verificable
requiere un override explícito. Antes de aceptar un candidato, comprueba sus
contratos y su ascendencia. Phase D parte de Phase C y resuelve su Phase A y su
Graph02 como una unidad. Se imprimen ruta, run ID, motivo y hasta cinco
descartes; se registra la misma evidencia en `dependency_resolution` del QC.

Ejemplo sintético probado: entre Phase A
`20261101T020000000000Z_00000003` y
`20261102T020000000000Z_00000003` se elige la segunda. Si le falta
`03_graph_n3_source_inventory.csv`, se descarta y se usa la primera con su
Graph02 `20261101T010000000000Z_00000002`. Si Phase C referencia la primera,
D conserva ese par aunque existan Phase A/Graph02 más recientes.

## Integridad normal y estricta

Normal comprueba siempre los hashes existentes de artefactos, manifiestos e
inventarios, el esquema, la finalización y las referencias entre releases.
No pide hashes al investigador ni compara el commit actual con un pin histórico.
Un cambio de código queda registrado como `REPORTED_CODE_DRIFT`; no autoriza
ningún endpoint. La alteración de datos consumidos sigue siendo un error.

`--strict` comprueba además los pins históricos **disponibles** en un config
explícito y aplica el control de drift documental. No obliga a inventar pins
ausentes. Conserva el mecanismo existente de evidencia humana firmada: la
existencia de un archivo o la ausencia de alertas nunca equivale a aprobación.

C/D crean exclusivamente directorios nuevos, restringidos, y rechazan outputs
superpuestos con entradas congeladas. D conserva sus snapshots y comprobaciones
en `finally`, y elimina paquetes incompletos en caso de fallo.

## Bloqueos: evidencia y remedios

No hay releases privados, diccionario consolidado ni formularios originales en
este entorno. Las observaciones siguientes proceden del código y fixtures;
no establecen cobertura real de pacientes ni validez clínica.

| Bloqueo | Estado en ausencia de evidencia revisada | Evidencia / remedio |
|---|---|---|
| Flags, pins manuales y mezcla de runs | `RESOLVED` | CLI sin argumentos, contratos y ascendencia comprobados |
| `demo__protocol` | `PENDING_SOURCE` o `PENDING_HUMAN_REVIEW` si hay alias observado | Step10 `classify_spine_column` clasifica `source_protocol` como CONTEXT; `classify_pro_column` hace lo mismo con `parent_protocol`; `derive_public_name` genera `spine__source_protocol` / `pro__parent_protocol`. Step01 Pop selecciona un campo original y lo copia a `protocol` (contexto `pop__protocol`). Estos nombres se auditan en C/D; no prueban equivalencia con el ajuste metodológico |
| Baseline / era | `PENDING_HUMAN_REVIEW` | Step00 conserva el episodio y el indicador oficiales; Step11 `validate_integrated` selecciona `is_clinical_baseline`. El ancla es fecha del episodio, no del instrumento ni de ingreso a la base |
| Fechas de instrumentos | `PENDING_SOURCE` | Exigir formulario/ensayo, fecha real, zona horaria, ventana baseline y regla de repetición. Un basal consolidado sin fecha sigue descriptivo. No se crea timestamp |
| Serología | `PENDING_SOURCE` y revisión humana | `01_serological_profile.py:select_episode_analytes` conserva `selected_lab_date` desde `lab_date`, reglas de selección del ensayo, conflictos y unidades; su proximidad al ancla puede incluir observaciones posteriores. No se extrapola esta regla a cuestionarios |
| Pop | `PENDING_HUMAN_REVIEW` | `observed_patterns.py:verified_baseline_pop` conserva `Unclassifiable`; un DISPLAY de Step01 no aprueba equivalencia con `Unclassified`. Sin firma se conservan separados; con crosswalk verificado se normaliza una copia en memoria y se reconcilian frecuencias antes/después sin reasignar Pop1/2/3 |
| Independencia de candidatos | `PENDING_HUMAN_REVIEW` | Matriz existente por representación/fuente separa dependencia directa, mismo ensayo, instrumento y proxy. Biopsia, ANA, RF y proteinuria siguen pendientes de evidencia de ensayo/codificación/unidad/observación. ESSDAI/ESSPRI conservan veto de circularidad con Pop |
| Soporte R0 estable | `RESOLVED` técnicamente; inferencia pendiente | Se evalúa la subpoblación hard, covered, untied, supported, stable antes de mirar outcomes. Columnas `r0_stable_only_*` en el preflight existente; no se afirma estabilidad de todo R0 |
| Topología completa y R2 | `PENDING_HUMAN_REVIEW` | Membresías intactas; R2 exploratoria; gates originales preservados |
| Modelos A/B/C/D | `NOT_ESTIMABLE` sin contratos revisados | Una muestra común por contraste, umbrales originales, rango/df/eventos/separación/multiplicidad. `primary_reason` y `possible_remedy` identifican la acción principal; no hay coeficientes, p-valores ni aprobación automática |

El inventario de protocolo calcula la cobertura por campo observado; las tablas
de Pop calculan frecuencias y reconciliación de pacientes. D mantiene la política
existente de suprimir **todos** los conteos y denominadores en el paquete para
impedir reconstrucción entre tablas; no se relaja para publicar un número.
No se puede informar cobertura clínica real sin los insumos privados.

`candidate_evidence_matrix` y `human_review_queue` siguen siendo la matriz y cola
existentes, sin nuevos reportes de adjudicación. `adjudication_status` es
`PENDING` o `REJECTED` cuando hay un rechazo humano verificado. Este software
no produce `APPROVED`. Un contrato de inferencia aprobado se debe aplicar en
la implementación de modelos existente, después de resolver sus gates.

Outputs principales: QC C/D con dependencias, matriz/cola de evidencia,
`model_preflight.csv` con sensibilidad estable y razón/remedio, inventarios de
protocolo, cronología y Pop. La equivalencia Pop solo afecta una copia derivada;
publicar otro dataset requiere un release nuevo. No hay exportación de IDs,
fechas exactas, valores libres o categorías pequeñas.

## Verificación real

Se utilizó `/workspace/.venvs/obj1_sjd/bin/python` (Python 3.12), con las
bibliotecas del entorno ya instaladas. No se descargaron datos privados.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_graph03*.py tests/test_graph_pipeline.py \
  tests/test_graph_representation_sensitivity.py \
  tests/test_graph_characterization_tiers.py tests/test_graph_bootstrap_qc.py \
  tests/test_graph_feature_catalog.py tests/test_graph_lab_contract_v2.py \
  tests/test_graph_lab_ingestion.py tests/test_topological_phenotyping.py
# 376 passed, 273 warnings, 413.08 s

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_graph03_run_resolution.py::test_normal_d_reports_downstream_code_drift_without_adding_a_blocker
# 1 passed, 4.21 s (adición posterior a la colección anterior)
```

La regresión amplia y la prueba de drift aprobaron **377 pruebas relacionadas**, incluyendo Graph02 → Phase A,
Phase C → D, ausencia de estimación sin revisión, alteración real de artefactos,
overrides, selección cronológica, ramas incompatibles, privacidad e idempotencia.
Los warnings no se ocultaron. El workflow existente incluye estas pruebas.

`--help` de ambos entrypoints terminó con código 0. Ambos scripts se ejecutaron
sin flags desde fuera del checkout usando `OBJ1_SJD_ROOT` para el fixture:
Phase C produjo `COMPLETE_WITH_DOCUMENTARY_BLOCKERS` (11 archivos) y Phase D
`COMPLETE_WITH_BLOCKERS` (18 archivos). Seleccionaron el run válido del
2026-11-02 y descartaron el del 2026-11-03 por inventario ausente. Esos son
contratos sintéticos; no constituyen una corrida clínica ni cobertura real.

Una ejecución intermedia terminó con un segfault nativo en el lector C de
pandas. La regresión ampliada anterior se ejecutó de nuevo con los hilos
numéricos limitados y terminó sin ese fallo; no se modificó el análisis para
ocultarlo. `compileall` y `git diff --check` también terminaron correctamente.

La ejecución clínica completa sigue pendiente de los releases y fuentes
originales autorizados. Los contratos de medición/independencia y las decisiones
humanas siguen siendo restricciones reales.

La última comprobación de C/D añadió los casos de resumen `FAILED` y marcador
`03_graph_failure.json`, y volvió a ejecutar todos sus contratos sobre el código
final:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_graph03_run_resolution.py \
  tests/test_graph03_n3_phasec_adjudication.py tests/test_graph03_n3_resolution.py
# 96 passed, 1 warning, 22.78 s (incluye pruebas de la regresión anterior)
```

También se ejecutó la suite global real:

```bash
python -m pytest -q
# Se detuvo en colección: 2 errores por archivos ya ausentes en main.
python -m pytest -q --continue-on-collection-errors
# 1682 passed, 12 failed, 1 skipped, 2 errors, 4017 warnings, 1307.31 s
```

Los doce fallos son preexistentes: se reprodujeron en un worktree limpio de
`cafb5f0`, sin estos cambios. No se modificaron esos módulos para ocultarlos.
Once casos fallaron de nuevo en 6.33 s con este comando en la base:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_08_medication_temporal_matching.py::test_leap_year_boundaries_priority_and_patient_isolation \
  tests/test_comorbidities.py::test_family_burdens_are_separate_and_no_combined_total_exists \
  tests/test_cross_domain_correlations.py::test_pairwise_complete_n_and_uninterpretable_serology \
  tests/test_cross_domain_correlations.py::test_pcs_prespecified_formula_excludes_forbidden_variables \
  tests/test_explainwhy_inclusive_v2.py::test_historical_scientific_functions_and_inputs_remain_exact \
  tests/test_missingness_observation_process.py::test_pattern_deterministic_and_completeness_distinct \
  tests/test_missingness_observation_process.py::test_actual_max_gap_and_membership_reconcile \
  tests/test_missingness_observation_process.py::test_python39_compatible_no_pep604_union \
  tests/test_observed_patterns_v2.py::test_v1_endpoints_outputs_and_step14_source_are_unchanged \
  tests/test_pharma_main_plots.py::test_domain_ordinal_delta_preserves_score_levels \
  tests/test_pros_longitudinal_labels.py::test_longitudinal_descriptives_are_labelled_as_available_measurements
# 11 failed
python -m pytest -q tests/test_visit_dates.py::test_date_fragments
# El duodécimo fallo también se reprodujo en main.
python -m pytest -q tests/test_btris_unmapped_lab_audit.py tests/test_pharma.py
# 2 errores de colección también reproducidos en main.
```

Los errores de colección se deben a los archivos ausentes
`src/20_btris_visit_date_match_report.py` y
`src/studies/pharma/01_run_pharma.py`. Los fallos restantes corresponden a
medicación, comorbilidades, correlaciones, observación/missingness, snapshots
históricos, etiquetas PRO y fechas; ninguno corresponde a Graph03. La suite
global **no está en verde**; la regresión relacionada sí aprobó.
