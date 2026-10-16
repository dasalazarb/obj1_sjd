# Graph 01/02: contrato Pharma 00 v2 y liberación científica

Implementación sobre `dasalazarb/obj1_sjd`, `main`
`65476b7da416e7aa6b6fb6f9f04c5882dbfef4f9` (16 de octubre de 2026).
Los cambios afectan la selección de inputs y su trazabilidad. Un conjunto LAB
distinto produce una representación nueva: no se declara equivalencia numérica
con los resultados históricos ni validación clínica a partir de estabilidad.

| Cambio | Archivo | Razón | Prueba | Resultado / impacto científico |
|---|---|---|---|---|
| Contrato v2, aprobación estricta, fuente pública y tipos finitos | `src/studies/graph/_lab_contract.py` | Impedir CSV antiguos, pendientes, contexto y conversiones silenciosas | T01–T07, T09, T12 | Rechazo antes de PCA; Pharma 00 sigue siendo el único decisor de elegibilidad |
| Identidad por SHA-256, revisión externa que reconoce HOLD y auditorías | `_lab_contract.py`, `graph_lab_release.yaml` | Separar aprobación del ensayo de autorización de la ejecución | T03, T06–T07 | Plantilla pendiente; no se inventa evidencia clínica |
| Cobertura basal simétrica y contadores antes/después del cast | `01_run_topological_phenotyping.py` | Eliminar excepción categórica global | T08, T10 | Tanto numéricos como categorías al 69% quedan fuera; al 70% entran si son variables |
| Políticas con y sin overlap, cohorte idéntica y outputs por corrida | Ambos scripts, ambas configuraciones | Auditar dependencia conceptual con Pop y evitar sobrescrituras | T11–T14, T20 | `without_overlap` retira solo esa feature antes de imputación/PCA |
| Config canónica explícita y matriz canónica compartida | `02_graph_representation_sensitivity.py` | S0 debe representar los mismos inputs y parámetros de Graph 01 | T14–T15, configuración incompatible | Hash de matriz cruda y control de Mapper/PCA/semilla/comunidades |
| Mapa sobre todo el universo LAB codificable; bloque S4 no vacío | `02_graph_representation_sensitivity.py` | Cubrir features que pueden reentrar en bootstrap | T16–T18 y regresión bootstrap | Error de contrato si falta mapa; no estimabilidad si no hay señal LAB |
| Auditoría de perfil excluido, pendientes, cobertura, redundancia y PCA | Ambos scripts | No confundir no catalogado con perfilado pero no exportado | T05, T19 | El perfil completo solo aporta auditoría, nunca predictores |
| Fixtures públicos y autorizaciones exclusivamente sintéticas | `tests/graph_contract_fixtures.py`, tests Graph/Pharma de integración | Eliminar fixtures históricos que usaban referencia como medición o fuentes v1 | Batería indicada abajo | No se convierten tests inseguros en respaldos de compatibilidad |

## Uso

Los defaults de integrado, candidatos y métricas Block A 11 se conservan. Los
cuatro documentos de Pharma se resuelven junto a `--graph-lab-candidates`:
`00_pharma_profile_summary.json`, `00_pharma_lab_profile.csv`,
`00_pharma_expected_lab_audit.csv` y `00_pharma_downstream_impact.csv`.
También pueden indicarse explícitamente con `--pharma-profile-summary`,
`--pharma-lab-profile`, `--pharma-expected-audit` y `--pharma-impact-audit`.

```bash
# Generar Pharma 00 únicamente cuando esté disponible el Step 10 correspondiente.
python src/studies/pharma/00_profile_labs.py

# Primera ejecución: no requiere liberación; nunca ajusta modelos.
python src/studies/graph/01_run_topological_phenotyping.py --preflight-only
python src/studies/graph/02_graph_representation_sensitivity.py \
  --preflight-only --canonical-config src/studies/graph/config.yaml

# SOLO después de completar una revisión humana real y revisar el preflight:
python src/studies/graph/01_run_topological_phenotyping.py \
  --graph-release /ruta/release_revisada.yaml --feature-policy legacy_with_overlap
python src/studies/graph/02_graph_representation_sensitivity.py \
  --graph-release /ruta/release_revisada.yaml --feature-policy legacy_with_overlap \
  --canonical-config src/studies/graph/config.yaml

# Sensibilidad independiente: repetir ambos comandos con:
# --feature-policy without_overlap
```

Las salidas conservan sus nombres dentro de
`graph/<script>/<feature_policy>/<run_id>/` en cada raíz estándar de tablas,
figuras, logs, QC y analíticos. Por defecto el ID combina hora UTC y sufijo único.
`--run-id` permite especificarlo; un marcador exclusivo impide reutilizarlo.
Las dos políticas generan directorios distintos incluso con el mismo ID.
`--dry-run` sigue significando omitir figuras, y exige liberación igualmente.

Cada corrida exporta `<01|02>_graph_preflight.json` e
`<01|02>_graph_input_provenance.json`. El primero informa
`READY_FOR_SCIENTIFIC_RUN` o `BLOCKED`, causas, conteos por ensayo y features,
contexto, drift y hashes. Un preflight válido pero sin liberación termina después
de sus auditorías agregadas. Los errores estructurales también escriben un reporte
BLOCKED y terminan con error. No hay flags de bypass.

La identidad del integrado se comprueba por **bytes**, no por nombre de ruta o
fecha de modificación. Dos copias idénticas son el mismo input; cambiar bytes
invalida el resumen. La proyección del perfil completo debe coincidir con los
candidatos, y la liberación fija hashes de candidatos, resumen y ambas auditorías.
Pharma 00 no emite un ID común ni hashes de sus outputs en el resumen: no se
atribuye al mtime una garantía adicional de pertenecer a la misma corrida.
Debe regenerarse el paquete junto y revisarse la evidencia real.

La liberación exige `status: approved_for_analysis`, identidad, fecha ISO,
revisor, evidencia, `analysis_scope: graph01_graph02_baseline`, lista exacta de
stems ya aprobados por Pharma, hashes de los cinco archivos declarados en la
plantilla, reconocimiento literal del HOLD original y ambos flags de revisión.
Si hay drift se exige `schema_drift_reviewed: true`; 485 columnas no es una regla
universal. Cambiar un hash o añadir un ensayo pending invalida la liberación.

## Linaje y decisiones científicas conservadas

Step 10 `build_curated()` consume la salida de Block A 06 y verifica la matriz
triestado de overlap. En `src/derivations/overlap_flags.py`, el eje glandular
combina dominios clínicos (incluida tumefacción glandular ESSDAI) y el eje
extraglandular usa dominios ESSDAI. `ovl__status` representa la combinación de esos
dos ejes. Block A `01_pop_distribution.py::classify_pop()` usa ESSDAI total y
ESSPRI observado. Por tanto existe solapamiento conceptual de dominios; el código
no demuestra equivalencia matemática ni independencia. Se conserva el brazo
`legacy_with_overlap` y se instrumenta `without_overlap` sobre los mismos pacientes.

`age_dx` procede exclusivamente del CSV de Block A 11. Su productor calcula
edad al diagnóstico con fecha diagnóstica menos nacimiento cuando están
disponibles, y conserva el alias. Ese cálculo no certifica que el conocimiento de
la fecha estuviera disponible en baseline. Se registra
`temporal_lineage_unverified` y la huella del CSV. No se sustituye por edad en visita.
El brazo adicional sin edad queda como revisión posterior opcional.

POP, ESSDAI, ESSPRI y derivados `sero__` no son inputs LAB. Solo se admiten
`lab__...__value` numéricos y `lab__...__text` binarios aprobados. Unknown permanece
missing; texto no compatible se rechaza sin exportar los valores libres.
No se rearmonizan unidades ni se convierte censura en puntos.

S0 conserva la representación canónica. S1 aplica cobertura por familia; S2
añade redundancia observada por pares; S3 añade MFA por familia; S4 conserva
exactamente las columnas, orden, valores y NaN de S3 y añade balance LAB jerárquico.
Se conservan los asserts de matrices y los pesos cero de grupos degenerados.
La selección se recalcula en cada submuestra; la elegibilidad Pharma se congela.
Los errores de contrato y mapeo se propagan, sin contarlos como inestabilidad.
No cambian Mapper, fórmulas, thresholds, permutaciones, modelos longitudinales
ni denominadores Jaccard/ARI/QC.

## Validación y bloqueos reales

Python del entorno: `/workspace/.venvs/obj1_sjd/bin/python`.

```bash
python -m compileall src/studies/graph
python -m pytest -q tests/test_graph_lab_contract_v2.py \
  tests/test_graph_lab_ingestion.py tests/test_topological_phenotyping.py \
  tests/test_graph_representation_sensitivity.py tests/test_graph_bootstrap_qc.py \
  tests/test_pharma_lab_profile.py tests/test_pharma_lab_profile_integration.py \
  tests/test_studies_shared.py
git diff --check
```

La batería final y los límites de ejecución se registran al final de este documento.
Los tests sintéticos cubren T01–T20, incluyendo fuentes mal declaradas, NaN/inf,
revisión estricta, hashes, cobertura 69/70%, matrices S0/S3/S4, mapa de reentrada,
no estimabilidad y bloqueo antes del ajuste. Los tests existentes con ajustes
completos siguen usando solamente pacientes sintéticos y releases de prueba.

Dos errores del mapa S4 ya se reproducen en la base remota sin este refactor:

1. `test_five_audited_mappings_preserve_selection_stages`: cinco filas tienen
   cuatro celdas bajo un encabezado expandido; `mapping_status` queda NaN.
2. `test_approved_ace_and_hepatitis_surface_antibody_mapping`: ACE tiene
   `INFLAMMATION_TISSUE` en el CSV y el test espera `GRANULOMATOUS_DISEASE`.

No se modifica el mapa ni se resuelven discrepancias clínicas automáticamente.
Estos problemas bloquean S4 si afectan a un ensayo elegible; un pending o
descriptive_only fuera de ese universo no exige mapeo.

Otro límite del productor: `graph_candidates()` de Pharma 00, cuando el inventario
completo está vacío, emite solo las columnas históricas. Graph rechaza ese CSV
incompleto con CONTRACT_MISMATCH. Un CSV vacío con todos los encabezados v2 y
resumen v2 sí se diagnostica, pero bloquea inferencia por NO_APPROVED_LABS.
No se añade respaldo v1 ni se modifica Pharma 00 fuera de alcance.

El registro versionado conserva 10/10 ensayos `pending`, 0 `approved`; esto es
inspección de configuración, no un conteo obtenido de pacientes reales.
Se invocaron ambos scripts con `--preflight-only` y sus rutas reales por defecto:
ambos escribieron BLOCKED por ausencia del Parquet integrado. No estaban
disponibles los nuevos CSV/JSON reales para verificar su contenido. No se ejecutó
Mapper, bootstrap ni modelos longitudinales sobre datos reales, y no se regeneró
Pharma 00 sin su maestro. **Liberación científica real: BLOCKED** hasta disponer
del paquete recién generado, revisar el diff y registrar aprobaciones y evidencia.

Resultado de la batería final: **161 passed, 2 failed** (los dos fallos de mapa
reproducidos también en la base). Las **52 pruebas nuevas** del contrato pasan.
`compileall` y `git diff --check` pasan. No se certifica la batería completa del
repositorio ni resultados científicos reales.
