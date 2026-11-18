# Graph03 N3 con recolección por día calendario

`calendar_day_prebaseline` es un contrato opcional de 03f y 03e para asociaciones exploratorias con membresías congeladas. Su alcance temporal es el día de recolección. La precedencia intradía, la identidad del espécimen y de la plataforma y la causalidad quedan fuera del alcance. El contrato v1 de timestamps con offset y ensayo y muestra verificados sigue siendo el predeterminado.

## Selección y procedencia

- Cotejar `Collected Date Time` en BTRIS raw e intermedio para cada protocolo 11D y 15D con `lab_date` en el parquet EDA de Step20. Inspeccionar los encabezados originales y documentar en el expediente dónde Step20 reduce la precisión a día. El código compara los días escritos en las fuentes sin conversiones de zona. Una fecha ausente, inválida o discordante se excluye. Un día interpretable de `specimen_datetime` que contradiga el cotejo también excluye la fila.
- La unión de los tres insumos usa los identificadores y nombres originales, con correspondencia literal. `Order ID` es parte de la clave de procedencia. Si una clave se repite en registros distintos del mismo CSV o de varios archivos, añadir un discriminador original documentado cuando exista. Si el vínculo sigue ambiguo, el registro queda bloqueado. La existencia de nombres o etiquetas de ensayo solo sirve para delimitar la definición comparable del crosswalk.
- Reconciliar el paciente con el baseline oficial del release Graph02 y con el episodio de origen que consta en EDA y Step10. Las recolecciones elegibles de otro episodio del mismo paciente conservan su `source_episode_id`. Step01 y contexto Step10 auditan la selección derivada por anchor por separado.
- Fijar la ventana por outcome antes de analizar asociaciones. La regla principal usa `1 <= baseline_day - collection_day <= lookback_days` y toma el último día elegible. El mismo día, días posteriores y días fuera de ventana se excluyen. Una sensibilidad con diferencia de al menos dos días requiere su propia configuración, expediente, artefacto y plan. En el último día se colapsan duplicados comprobablemente idénticos. Discordancia en resultado, unidad, definición o identidad lógica genera `AMBIGUOUS_SAME_DAY` y excluye al paciente para ese outcome sin recurrir a un día anterior.
- El crosswalk firmado restringe el outcome a combinaciones comparables de nombre original de orden, nombre original de observación y etiqueta EDA `assay` (puede ser nula), con una unidad exacta. La definición describe la prueba clínica comparable dentro de ese alcance. `assay_id` y `sample_id` quedan nulos. Los tokens desconocidos o no realizados se tratan como faltantes. Los valores censurados, tokens ajenos a la convención y unidades fuera de alcance se excluyen. Incluir ANA general y HEp-2 juntos exige evidencia revisada de que son hipótesis distintas y definiciones de origen disjuntas. Las dos codificaciones de RF representan la misma hipótesis por representación.

## Configuración y ejecución privada

En una copia privada de `src/studies/graph/configs/n3_measurement_source.yaml`, usar rutas absolutas para los insumos de `source_paths` o pasarlos por CLI. Las rutas relativas se resuelven desde el archivo de configuración. Activar `schema_version: graph03_n3_measurement_source_v2` y definir esta sección según el diccionario institucional de columnas. El valor de `lookback_days` se decide en el protocolo antes de evaluar resultados. Solo se procesan los outcomes declarados en `windows`.

```yaml
calendar_day:
  contract: calendar_day_prebaseline
  windows:
    # Añadir una entrada por columna analítica con los dos enteros aprobados.
    # lab__rheumatoid_factor__value:
    #   minimum_days_before_baseline: 1
    #   lookback_days: <entero preespecificado>
  linkage_fields:
    patient_id: {eda: patient_id, raw: Patient ID, intermediate: Patient ID}
    original_order_id: {eda: order_identifier, raw: Order ID, intermediate: Order ID}
    original_order_name: {eda: order_name_original, raw: Order Name, intermediate: Order Name}
    original_observation_name: {eda: observation_name_original, raw: Observation Name, intermediate: Observation Name}
    # Si se acredita un discriminador original, agregar record_id con sus tres columnas.
  date_formats:
    raw: ISO8601
    intermediate: ISO8601
```

Usar un formato explícito de año de cuatro dígitos cuando algún CSV no use ISO 8601. Configurar `cluster_name_original` como nombre EDA de observación solo si el diccionario confirma que corresponde a `Observation Name`. Cada ruta BTRIS acepta un CSV o un directorio con archivos `Lab*.csv`. El inventario fija por SHA-256 cada archivo y el conjunto de archivos de cada ruta.

```bash
: "${N3_PRIVATE_PARENT:?Directorio institucional privado existente, fuera del repositorio}"
: "${SOURCE_CONFIG:?Configuración privada v2 con ventanas preespecificadas}"
: "${G02:?Manifest del release congelado de Graph02}"
: "${G03:?Directorio congelado de Graph03 Fase A}"
: "${EDA_LONG:?Parquet original EDA Step20}"
: "${STEP01:?Parquet de analitos de Step01}"
: "${STEP10:?Parquet analítico de Step10}"
: "${CONTEXT:?Parquet de contexto de Step10}"
: "${DATA_ROOT:?Directorio que contiene data_raw y data_intermediate de BTRIS}"
PY="${PY:-python}"
umask 077
RUN_ROOT="$(mktemp -d "$N3_PRIVATE_PARENT/graph03-calendar.XXXXXXXX")"
SOURCE_INPUTS=(
  --config "$SOURCE_CONFIG" --output-root "$RUN_ROOT"
  --graph02-release "$G02" --graph03-run "$G03"
  --eda-long "$EDA_LONG" --step01 "$STEP01" --step10 "$STEP10" --context "$CONTEXT"
  --btris-raw-11d "$DATA_ROOT/data_raw/BTRIS/11D"
  --btris-raw-15d "$DATA_ROOT/data_raw/BTRIS/15D"
  --btris-intermediate-11d "$DATA_ROOT/data_intermediate/BTRIS/11D"
  --btris-intermediate-15d "$DATA_ROOT/data_intermediate/BTRIS/15D"
)
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --audit-only "${SOURCE_INPUTS[@]}" --run-id audit_v2
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --build-candidate "${SOURCE_INPUTS[@]}" --run-id candidate_v2
```

`calendar_day_eligibility.csv` cuenta pacientes distintos por outcome. `day_available`, `eligible_prebaseline`, `same_day_excluded`, `postbaseline_excluded`, `ambiguity_excluded` y `test_definition_verified` pueden superponerse. En un candidato, la elegibilidad es preliminar y la columna de definición verificada queda vacía. En un run firmado, la elegibilidad cuenta selecciones que además cumplen el crosswalk, la unidad y el resultado revisados. Las celdas se retienen en conjunto si las particiones permiten identificar grupos por debajo del umbral de divulgación. El CSV, el manifiesto y los Parquet y ledgers se mantienen como `INTERNAL_RESTRICTED` fuera de Git. El ledger privado identifica la fila EDA fijada por hash, su origen y el motivo de exclusión.

## Revisión de la fuente y del par

El expediente externo usa `schema_version: graph03_n3_measurement_review_v2`, `decision: APPROVED_SOURCE_PROVENANCE`, `reviewer`, `review_id`, `input_sha256` igual a `review_binding` de la auditoría y `evidence` con anexos relativos hasheados. Por outcome, `CALENDAR_AFFIRMATIONS` describe las afirmaciones requeridas. El crosswalk `comparable_original_tests` lista tuplas exactas de `original_order_name`, `original_observation_name` y `assay`. El expediente fija `date_field: lab_date`, `unit_field: unit`, el campo original de resultado y un `measurement_review` con el tipo, definición, unidad, ventana, reglas de selección y convención de categorías. Esta revisión declara `time_resolution: calendar_day` e `identity_contract: reviewed_test_definition_without_specimen_id`. El expediente debe documentar consentimiento y uso institucional, truncamiento temporal, comparabilidad y riesgo residual. El fixture sintético de [pruebas v2](../tests/test_graph03_n3_calendar_day.py) muestra la estructura de los campos sin representar una aprobación clínica.

```bash
: "${SOURCE_REVIEW:?Expediente YAML firmado por el custodio}"
: "${SOURCE_SIGNATURE:?Firma separada del expediente}"
: "${SOURCE_TRUST_KEY:?Clave pública institucional autorizada para 03f}"
: "${DECISION_ROOT:?Directorio institucional de decisiones de 03e}"
"$PY" src/studies/graph/03f_build_n3_measurement_source.py --build-reviewed "${SOURCE_INPUTS[@]}" \
  --review "$SOURCE_REVIEW" --signature "$SOURCE_SIGNATURE" --trust-key "$SOURCE_TRUST_KEY" --run-id reviewed_v2

MEASUREMENTS="$RUN_ROOT/runs/reviewed_v2/reviewed_instrument_measurements.parquet"
INFERENCE_INPUTS=(--graph02-release "$G02" --graph03-run "$G03" --context "$CONTEXT"
  --measurements "$MEASUREMENTS" --decision-root "$DECISION_ROOT")
"$PY" src/studies/graph/03e_graph_n3_controlled_inference.py "${INFERENCE_INPUTS[@]}" \
  --prepare-plan --plan "$RUN_ROOT/draft_v2.yaml" --output-dir "$RUN_ROOT/templates_v2"

: "${REVIEWED_PLAN:?Plan 03e con decisiones por representación y outcome}"
: "${PAIR_TRUST_KEY:?Clave pública institucional autorizada para 03e}"
"$PY" src/studies/graph/03e_graph_n3_controlled_inference.py "${INFERENCE_INPUTS[@]}" \
  --plan "$REVIEWED_PLAN" --measurement-trust-key "$PAIR_TRUST_KEY" --output-dir "$RUN_ROOT/preflight_v2"
```

La decisión `measurement` de 03e debe reproducir el `measurement_review` firmado en `details`, agregar `source_review_sha256` con el hash del expediente de 03f y apuntar al archivo de firma separada mediante `signature_path` relativo a `--decision-root`. La firma del par cubre el contrato, su alcance y los hashes del plan y del Parquet revisado. La institución verifica el expediente de procedencia antes de firmar el par. El código valida las firmas con las claves públicas provistas por el canal institucional.

`SOURCE_REVIEWED` acredita el alcance del expediente de 03f. `PAIR_REVIEW_PENDING` señala la revisión separada en 03e. El preflight genera `READY` o `NOT_ESTIMABLE` por modelo según las compuertas de protocolo, baseline, medición, temporalidad, independencia, Pop cuando aplica, topología, diseño y soporte. `--execute` estima únicamente modelos `READY`. El protocolo de investigación `demo__protocol` requiere acreditación semántica y fecha efectiva. ESSDAI, ESSPRI y biopsia requieren sus propias fechas instrumentales y revisión. La auditoría y las pruebas comparan los SHA-256 de los releases congelados antes y después.

En el entorno de implementación faltaban BTRIS raw/intermedio, el parquet institucional EDA y los releases congelados. La revisión de la correspondencia de columnas, la disponibilidad clínica por outcome, la elegibilidad observada y las aprobaciones institucionales siguen pendientes. Los flujos v1 y v2 se prueban con insumos sintéticos.
