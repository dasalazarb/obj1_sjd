# Procedencia de las mediciones originales para Graph03 N3

> Referencia histórica del circuito previo a `graph03_baseline_window_v2`. Los comandos vigentes y
> la ruta de migración están en el [inventario de Graph03](graph03_script_inventory_and_migration.md).
> El [contrato de fuentes y linaje](graph03_source_and_lineage_contract.md) describe los requisitos
> de revisión y firma para la ventana de día 0 a seis meses calendario.

`03f_build_n3_measurement_source.py` audita fuentes originales frente a Graph02 y Graph03 Fase A elegidos explícitamente. La publicación revisada acredita la procedencia de la medición para el alcance firmado. La elegibilidad de un contraste depende de la revisión del estimando y del preflight de [03e](graph03_controlled_inference.md).

El contrato v1 de timestamps sigue siendo el predeterminado. El [contrato opcional de día calendario](graph03_n3_calendar_day.md) describe el cotejo con BTRIS, la revisión firmada de una definición comparable y los comandos para laboratorios sin identidad verificable de ensayo o espécimen.

## Inspección de código y estado de la evidencia

| Código inspeccionado | Hallazgo |
| --- | --- |
| `03e_graph_n3_controlled_inference.py`, `n3_controlled_inference.yaml` | El modo automático puede emitir `SOURCE_CONTRACT_NOT_READY`. `--prepare-plan` crea un borrador sin firma. La configuración de producción mantiene vacíos los outcomes, revisores y decisiones. |
| `s03_n3_source_adapter.py`, `01_serological_profile.py`, `10_build_integrated_longitudinal_dataset.py` | Step01 conserva `selected_lab_date` del registro seleccionado por menor distancia absoluta al anchor. En un empate favorece anchor/pre-anchor y puede usar una mediana para valores repetidos. Step10 propaga la fecha de selección a contexto, y clasifica o armoniza los campos analíticos según su contrato. El adaptador anterior produce candidatos con assay y muestra nulos a partir de estos derivados. |
| `s03_n3_approved_view.py`, `s03_n3_controlled.py` | Los hashes anclan el plan. El consumidor selecciona el último resultado elegible dentro de la ventana aprobada, mantiene los grupos desconocidos y bloquea empates discordantes. El preflight aplica el contrato de baseline, comparadores y población común de los modelos A/B/C/D. |
| `s03_n3_decisions.py`, `s03_n3_phasec_adjudication.py`, `validation_03.yaml` | Las decisiones de 03e requieren alcance por representación/outcome y evidencia con hash. Los roles congelados y el grafo de dependencias permiten identificar features y proxies de descubrimiento. La comparación Pop requiere revisión adicional. |
| `00_build_visit_spine.py`, `01_essdai_esspri.py`, `01_extended_clinical_phenotype.py` | El spine conserva el anchor y el baseline oficiales de episodio. ESSDAI toma el total del instrumento, ESSPRI promedia tres componentes completos y la derivación de biopsia enmascara valores discordantes de patología. Estos valores requieren cotejo con el registro original fechado. |

También se revisaron los contratos de [Phase C](graph03_phasec_endpoint_review_protocol.md) y [caracterización por niveles](graph03_characterization_tiers.md).

| Outcome | Origen que debe examinarse | Estado institucional en esta ejecución |
| --- | --- | --- |
| `lab__ana_status__text` | EDA/BTRIS `canonical_analyte=ana_status`, texto original | `UNAVAILABLE`. EDA institucional inaccesible. Se requiere identidad del ensayo y descartar el solapamiento con HEp-2. |
| `lab__ana_hep2_status__text` | EDA/BTRIS `canonical_analyte=ana_hep2_status`, texto original | `UNAVAILABLE`. Se requiere método HEp-2 documentado y distinción frente a ANA general. |
| `lab__rheumatoid_factor__text` | EDA/BTRIS `canonical_analyte=rheumatoid_factor`, categoría original | `UNAVAILABLE`. Se requiere convención cualitativa por ensayo. Comparte la hipótesis de RF numérico. |
| `lab__rheumatoid_factor__value` | Misma fuente EDA, `result_numeric_exact` con operador y límite disponibles para auditoría | `UNAVAILABLE`. Se requieren unidad, comparabilidad y resolución de censura. |
| `lab__urine_protein_qualitative__text` | EDA/BTRIS `canonical_analyte=urine_protein_qualitative`, categoría original | `UNAVAILABLE`. Se requieren método y escala de la medición cualitativa. |
| `essdai__total` | Registro original CTDB de evaluación ESSDAI, `essdai__essdai_total_score` | `BLOCKED`. `ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED`. Dependencia definitoria de Pop. |
| `esspri__total` | Cuestionario original CTDB, componentes de sequedad, fatiga y dolor | `BLOCKED`. `ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED`. Dependencia definitoria de Pop. |
| `ext__biopsy_focus_score` | Registro original de patología CTDB, `biopsy_pathology__f_score` | `BLOCKED`. `ORIGINAL_INSTRUMENT_DATE_NOT_VERIFIED`. Se requiere fecha de toma o informe según el estimando, además de sitio, escala e indicación. |

`UNAVAILABLE` describe el acceso al insumo en este entorno. `NOT_VERIFIED` describe una búsqueda de la fuente original que sigue abierta. Con datos reconciliados, `CANDIDATE_UNAPPROVED` identifica observaciones pendientes. `VERIFIED` solo se asigna a la identidad basal verificada o al alcance instrumental de una revisión externa validada.

- El campo `lab_date` de EDA es la **fecha instrumental candidata**. `specimen_datetime` conserva evidencia independiente de recolección. Se registra el valor original, su precisión, el offset y la comparación temporal. En el modo revisado v1 se exige un timestamp instrumental con offset explícito. Día, medianoche sin precisión demostrable, hora local sin offset y día del baseline quedan bloqueados. Se conservan en el candidato las mediciones fuera de ventana y posteriores. El dossier fija el significado UTC del corte basal y el intervalo.
- El campo original `assay` se conserva como evidencia del **ensayo**. Un revisor acredita el método y la correspondencia con un identificador de ensayo concreto. Los ensayos y unidades mezclados requieren adjudicación. ANA general y HEp-2 requieren identidades originales y revisadas distintas.
- La configuración de la **clave de muestra** `eda_specimen_id_column` comienza en nulo. Queda pendiente verificar si existe una clave original de muestra. El candidato conserva orden, fecha de espécimen y localizador de fila como campos distintos. La muestra revisada requiere una clave de muestra original y verificada.
- El candidato preserva el **resultado y la unidad originales**, los operadores, los límites y los tokens administrativos. La comparación Step01/Step10 audita fechas, valores y unidades propagados. La repetición de una medición y el valor derivado en Step01 quedan registrados por separado. Los desconocidos y no realizados solo pueden corresponder a categorías faltantes en la convención autorizada.
- El inventario detecta los alias de **protocolo de investigación**. Se requiere acreditar la asignación real a nivel paciente, el diccionario de códigos, la fecha efectiva y UTC. Las asignaciones múltiples requieren adjudicación institucional antes del contrato de asignación única de 03e.

## Operación en un directorio privado

Los siguientes ejemplos usan nombres sintéticos. El operador puede resolver las rutas mediante opciones explícitas o la configuración. El productor toma la identidad oficial del `baseline_raw` del release Graph02. Si se pasa `--baseline`, debe tener el mismo SHA-256 que el original. Un release puede contener un subconjunto del baseline de Step10 y sus pares y fechas deben reconciliarse exactamente. EDA entrega la asignación original al episodio. Step01 aporta únicamente la normalización documentada del identificador de paciente para reconciliar la clave conjunta. `--spine` y las fuentes opcionales de cuestionarios, patología y protocolo permiten inventariar sus esquemas.

```bash
PY=/usr/local/Anaconda/envs/py3.10/bin/python
: "${REPO:?Ruta al clon de obj1_sjd}"
: "${N3_FIXTURES:?Directorio privado con insumos sintéticos}"
: "${N3_PRIVATE_PARENT:?Directorio privado existente, fuera del repositorio y de los releases}"
cd "$REPO"
umask 077
PRIVATE="$(mktemp -d "$N3_PRIVATE_PARENT/graph03_n3.XXXXXXXX")"
G02="$N3_FIXTURES/graph02/02_graph_release_manifest.json"
G03="$N3_FIXTURES/graph03"
CTX="$N3_FIXTURES/context.parquet"
COMMON_INPUTS=(
  --eda-long "$N3_FIXTURES/eda.parquet"
  --step01 "$N3_FIXTURES/step01.parquet"
  --step10 "$N3_FIXTURES/step10.parquet"
  --context "$CTX"
  --graph02-release "$G02"
  --graph03-run "$G03"
  --output-root "$PRIVATE"
)

"$PY" src/studies/graph/03f_build_n3_measurement_source.py --help
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --audit-only "${COMMON_INPUTS[@]}" --run-id audit
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --build-candidate "${COMMON_INPUTS[@]}" --run-id candidate
```

`audit-only` publica `provenance.json` y `readiness.csv` con campos conocidos, hashes, bloqueos y agregados permitidos por la política de celda mínima. Si alguna partición vinculada contiene una celda pequeña, se suprimen en conjunto los agregados del inventario restringido. La matriz usa `feature_sets`, roles y dependencias de la representación congelada y deja soporte y decisiones de 03e pendientes. `build-candidate` añade `unapproved_observations.parquet` y `private_ledger.parquet` cuando hay insumos reconciliados. El ledger usa el desplazamiento físico de la fila en el EDA hasheado y registra exclusiones de identidad y bloqueos de procedencia. Para cada observación reconciliada el candidato conserva los campos originales seleccionados. Los campos de contrato `measurement_date`, `assay_id` y `sample_id` permanecen nulos en el candidato.

Las salidas se escriben con `umask 077`, directorios `0700` y archivos `0600`. Un run completo aparece mediante renombrado de directorio bajo `runs/<run_id>`. Cualquier colisión se rechaza. El manifiesto incluye hashes SHA-256 de fuentes, archivos congelados, código, reglas, configuración y productos. La revisión fija también las versiones de Python, pandas y PyArrow. Se recalculan hashes antes de publicar. Un cambio de fuente o release aborta el run. Todos los artefactos del run son `INTERNAL_RESTRICTED` y su divulgación se decide institucionalmente.

## Revisión externa y preflight

El custodio institucional debe ubicar el parquet original EDA/BTRIS, confirmar el campo de muestra, método, unidad y significado temporal, y proporcionar un expediente firmado con una clave institucional distribuida por un canal autorizado. Si se cambia `eda_specimen_id_column` en una configuración privada, se ejecuta de nuevo la auditoría con esa misma configuración para obtener los nuevos hashes.

El expediente YAML usa `schema_version: graph03_n3_measurement_review_v1`, `decision: APPROVED_SOURCE_PROVENANCE`, identificadores institucionales `reviewer` y `review_id`, `input_sha256` igual a `review_binding` del inventario y una lista `evidence` con rutas relativas y SHA-256 de los anexos. Cada entrada de `outcomes` aporta la procedencia del episodio, la identidad de ensayo y muestra, la unidad, la semántica de resultado y tiempo, y el manejo de datos no disponibles. El expediente contiene `assay_field`, `original_assay`, `sample_field`, `date_field`, `unit_field`, `value_field`, los campos de verificación de `AFFIRMATIONS` y el contrato `measurement_review` de 03e, con ventana, ensayo, unidad y convención de categorías. Consulta los fixtures sintéticos en `tests/test_graph03_n3_measurement_source.py` para la forma exacta.

El verificador necesita OpenSSL y una firma separada válida para `openssl dgst -sha256 -verify`. La clave pública debe estar bajo control institucional. Un expediente sin procedencia, con anexos alterados, con firma no verificable o con alcance distinto queda bloqueado. Las contradicciones de fecha o valor entre Step01 y Step10 bloquean la publicación revisada. El modo v1 publica resultados laboratoriales binarios o numéricos exactos. Resolver un instrumento no laboratorial requiere evidencia y una transformación revisada de su fuente original. El archivo `private_reviewed_ledger.parquet` vincula el desplazamiento de cada observación revisada con la fila EDA y registra las filas fuera de ventana o posteriores al baseline. Los documentos firmados, manifiestos privados y datos se conservan fuera de Git.

```bash
: "${REVIEW_CONFIG:?Configuración privada usada en la auditoría firmada}"
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --audit-only "${COMMON_INPUTS[@]}" \
  --config "$REVIEW_CONFIG" --run-id audit_review_configuration

: "${REVIEW_YAML:?Expediente firmado fuera de Git}"
: "${REVIEW_SIGNATURE:?Firma separada emitida por el custodio}"
: "${INSTITUTIONAL_PUBLIC_KEY:?Clave pública institucional autorizada}"
: "${DECISION_ROOT:?Directorio institucional de decisiones de 03e}"

"$PY" src/studies/graph/03f_build_n3_measurement_source.py --build-reviewed "${COMMON_INPUTS[@]}" \
  --config "$REVIEW_CONFIG" --review "$REVIEW_YAML" --signature "$REVIEW_SIGNATURE" \
  --trust-key "$INSTITUTIONAL_PUBLIC_KEY" --run-id reviewed

MEASUREMENTS="$PRIVATE/runs/reviewed/reviewed_instrument_measurements.parquet"
INFERENCE_INPUTS=(--graph02-release "$G02" --graph03-run "$G03" --context "$CTX" \
  --measurements "$MEASUREMENTS" --decision-root "$DECISION_ROOT")
"$PY" src/studies/graph/03e_graph_n3_controlled_inference.py "${INFERENCE_INPUTS[@]}" \
  --prepare-plan --plan "$PRIVATE/draft.yaml" --output-dir "$PRIVATE/unsigned_templates"

# Cuando exista un plan fijado y revisado por el equipo clínico.
: "${REVIEWED_PLAN:?Plan de 03e revisado, con hashes y decisiones por par}"
"$PY" src/studies/graph/03e_graph_n3_controlled_inference.py "${INFERENCE_INPUTS[@]}" \
  --plan "$REVIEWED_PLAN" --output-dir "$PRIVATE/preflight"
```

Las decisiones de 03e cubren protocolo, baseline y era, medición, independencia, Pop en caso de aplicar, topología y diseño por representación/outcome. Revisa `preflight.csv` y `blockers.csv` antes de solicitar `--execute` para los pares elegibles. El archivo canónico de Step10 y el selector automático de releases permanecen fuera de este flujo.

## Pruebas

En el entorno de desarrollo, Python 3.12.14 completó las 171 pruebas de las seis suites con una advertencia de pandas en una prueba existente de resolución de runs. El modo de auditoría local sin los insumos institucionales devolvió `SOURCE_CONTRACT_NOT_READY` y escribió solo los dos informes restringidos.

Biowulf y los insumos institucionales no estaban disponibles en el entorno de desarrollo. El comando de abajo usa el intérprete esperado en Biowulf y también admite el intérprete de desarrollo mediante `PY`. Los fixtures generan releases temporales, timestamps, claves y firmas exclusivamente sintéticos.

```bash
PY="${PY:-/usr/local/Anaconda/envs/py3.10/bin/python}"
TEST_TMP="$(mktemp -d)"
"$PY" -m pytest -q --basetemp "$TEST_TMP/pytest" \
  tests/test_graph03_n3_source_adapter.py \
  tests/test_graph03_n3_controlled.py \
  tests/test_graph03_n3_phasec_adjudication.py \
  tests/test_graph03_n3_resolution.py \
  tests/test_graph03_run_resolution.py \
  tests/test_graph03_n3_measurement_source.py
```

La prueba de integración invoca el CLI de auditoría con fuentes bloqueadas y verifica una publicación sintética firmada. Luego pasa la fuente revisada a `03e --prepare-plan`, ejecuta preflight y confirma que la falta de decisiones separadas impide estimar el contraste. También compara todos los hashes de los dos releases sintéticos antes y después.
