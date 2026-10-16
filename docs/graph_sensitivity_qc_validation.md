# Validación de la corrección del bootstrap de sensibilidad Graph

El texto de la especificación original se conserva en
[graph_sensitivity_correction_spec_es.md](graph_sensitivity_correction_spec_es.md).
Los métodos y contratos de salida están en
[src/studies/graph/README.md](../src/studies/graph/README.md).

## Fallo y comportamiento corregido

Antes, un intento computacional fallido incrementaba el denominador de
estabilidad sin distinguirse de una ausencia de match topológico. El mapa S4
solo se validaba sobre features retenidas en la muestra completa; una feature
podía reaparecer en una réplica y fallar por mapping. Un grupo de serología
constante también abortaba la réplica. La salida `YES` no exigía reproducibilidad
entre representaciones.

Ahora se valida todo `full_values` antes de S0, se conservan los grupos LAB
constantes como bloques cero cuando hay otros grupos activos y se registran
los fallos no estimables. Cada intento conserva seed, selección reproducible,
etapa, causa, features y comunidades. Se entregan los estimadores condicional y
conservador. La estabilidad primaria y el conteo de comunidades estables son NA
cuando no se supera el QC operativo de 95%; no se presentan como cero biológico.
Los errores inesperados de código/esquema se propagan.

El presupuesto formal sigue siendo 60 réplicas, fracción 0.80; Jaccard de match
0.50, estabilidad 0.80, Mapper y PCA no cambian. No hay reintentos de reposición.
S3 y S4 conservan exactamente la misma matriz cruda, orden y missingness.

## Mapeo clínico

El mapa tiene 74 laboratorios explícitos; la tabla de cobertura de metadatos está
en [graph_sensitivity_group_coverage.csv](graph_sensitivity_group_coverage.csv).
Esto no demuestra que el universo clínico de una corrida esté completo: esa
condición se comprueba con su audit de preflight y bloquea cualquier lab pendiente.

| Laboratorio añadido | Grupo/subgrupo | Procedencia |
| --- | --- | --- |
| ACE (`angiotensin_converting_enzyme`) | `GRANULOMATOUS_DISEASE` / `ACE` | El investigador aprobó expresamente `GRANULOMATOUS_DISEASE` en esta conversación; el subgrupo identifica el analito. |
| `hepatitis_b_surface_antibody` | `INFECTIOUS_SEROLOGY` / `HEPATITIS` | Clasificación solicitada en la especificación, coherente con las serologías de hepatitis existentes. |

No se añadieron estos laboratorios artificialmente al modelo principal. Las
reglas de cobertura y redundancia siguen determinando su inclusión por muestra.

## Verificación ejecutada

```bash
MPLCONFIGDIR=/workspace/work/matplotlib XDG_CACHE_HOME=/workspace/work/cache \
  /workspace/.venvs/obj1_sjd/bin/python -m pytest \
  tests/test_graph_representation_sensitivity.py \
  tests/test_graph_bootstrap_qc.py \
  tests/test_graph_lab_ingestion.py \
  tests/test_topological_phenotyping.py \
  -q -W error::FutureWarning
# 54 passed in 16.42s
```

Se ejecutó en el Python 3.12 disponible. La sintaxis del script también se
validó con `ast.parse(..., feature_version=(3, 10))`; esto no sustituye ejecutar
la suite en el entorno NIH Python 3.10. No se modificaron paquetes ni filtros
de warnings del usuario.

Las pruebas verifican el universo que podría reingresar al bootstrap (incluida
una feature por debajo de la cobertura S4 completa), errores antes de cualquier
escenario, duplicados normalizados, grupo infeccioso constante, todas las LAB
degeneradas, señal diminuta real, refit por intento, reproducibilidad, denominadores
60/27, 60/60 y 60/0, fallos por etapa, propagación de `KeyError`, intersección
común de pacientes y bloqueo interpretativo por QC. Se preservan todos los CSV
previos, orden de pacientes y tres PNG en los casos sano e inválido. También se
inspeccionaron las figuras de estabilidad: los resultados inválidos se rotulan
como tales, sin barras de estabilidad cero.

El diff de las rutas protegidas frente a `origin/main` es vacío: Pharma,
Economic, Script 01 y Block A no se modificaron.

## Ejemplos auditables y nota antes/después

[graph_sensitivity_synthetic_qc/](graph_sensitivity_synthetic_qc/) contiene
auditorías y resúmenes producidos por la integración sintética de 24 pacientes.
Cada CSV está rotulado `SYNTHETIC_QC_ONLY_NOT_CLINICAL`; `metadata.json` conserva
la configuración de depuración. Se usó B=1 y Mapper de fixture, por lo que las
comunidades de ese ejemplo no constituyen evidencia científica.

| Escenario sintético | Solicitadas | Exitosas | Fallidas | QC | Comunidades que superan 0.80 en el fixture B=1 |
| --- | ---: | ---: | ---: | --- | ---: |
| S0 | 1 | 1 | 0 | PASS | 1 |
| S1 | 1 | 1 | 0 | PASS | 1 |
| S2 | 1 | 1 | 0 | PASS | 1 |
| S3 | 1 | 1 | 0 | PASS | 2 |
| S4 | 1 | 1 | 0 | PASS | 2 |

Los ejemplos de denominadores, probados con 60 intentos simulados, están en
`denominator_before_after_examples.csv`:

| Ejemplo sintético | Estimador anterior | Condicional corregido | Conservador | Primario interpretable |
| --- | ---: | ---: | ---: | --- |
| 27 éxitos, 27 matches, 33 fallos de Mapper | 0.45 | 1.00 | 0.45 | NA: QC fallido |
| 60 éxitos, 50 matches | 0.8333 | 0.8333 | 0.8333 | 0.8333: QC aprobado |
| Ningún éxito | 0 | NA | 0 | NA: bootstrap inválido |

Los 33 fallos del primer ejemplo son simulados; no se reconstruyó el recuento
real a partir de líneas de log. Las tablas de parejas distinguen Jaccard por
asignación dura de Jaccard por unión de nodos; ARI pairwise conserva n/total y el
análisis secundario usa la intersección válida común a los cinco escenarios.

## Límite clínico y conclusión

Se intentó ejecutar el CLI con los argumentos por defecto y `--dry-run`.
Está bloqueado porque no existe en este entorno
`outputs/tables/studies/pharma/00_profile_labs/00_pharma_graph_lab_candidates.csv`;
tampoco están disponibles el Parquet clínico ni el CSV real de Block A.
No se generaron resultados de los 159 pacientes ni se comprobó otro alias real
de edad al diagnóstico. Su ausencia sigue registrándose, sin sustituirla.

Por ello no se puede afirmar si cambió la conclusión de S4 ni decidir entre
núcleo reproducible, múltiples ramas o estructura global sensible con datos
clínicos. Esas preguntas permanecen no evaluables hasta ejecutar el presupuesto
formal sobre los insumos reales y examinar las auditorías. Además, los nuevos
umbrales científicos de similitud están en null: no se eligieron a partir de
los ARI conocidos. La salida entrega métricas descriptivas y bloquea afirmaciones
automáticas; cualquier criterio futuro requiere justificación previa explícita.
