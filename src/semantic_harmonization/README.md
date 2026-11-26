# Armonización semántica clínica SjD

La V2 de `obj1_sjd` publica la tabla clínica efectiva y el candidato armonizado después de Step 10. Ambos conservan el esquema, las claves y el orden del original. La tabla efectiva aplica correcciones validadas celda por celda y conserva el original literal cuando la corrección es parcial, insegura o incompatible con el tipo fuente. El candidato armonizado conserva todos los valores originales. Añade campos `harm__` y tablas largas con resultados exactos, límites y rangos, serología, hallazgos, alelos y medicamentos. El diccionario conserva originales, evidencia, contexto del ensayo y frecuencias. El reporte HTML funciona sin conexión y el directorio de cada ejecución tiene permisos restrictivos.

## Entradas y productores

| Entrada | Productor y uso |
| --- | --- |
| `obj1_sjd/data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet` | `obj1_sjd/src/block_A/10_build_integrated_longitudinal_dataset.py`. La unidad es paciente y episodio. |
| `10_integrated_longitudinal_context.parquet` en el mismo directorio | Lo escribe `_write_audits` en Step 10. Se valida un conjunto idéntico de claves y se usa la metadata disponible por analito. La fecha de medición y el estado de referencia tienen significados distintos del tiempo real de disponibilidad y de un resultado explícito. |
| `paths.source_events` (opcional) | Parquet autorizado disponible para `obj1_sjd` con contrato de observación BTRIS. Step 01 documenta el origen upstream en `src/block_A/01_serological_profile.py`. El módulo lee esta entrada y escribe un sidecar con una fila por observación, los campos originales y el mapeo. |
| `Consolidated_Codebook_all_columns.xlsx` | Hoja `Consolidated_Codebook`. Configura la ruta si el archivo queda fuera de `obj1_sjd` o si existe más de una copia. |
| `obj1_sjd/src/studies/pharma/expected_lab_concepts.yaml` | Proveniencia para identidad y elegibilidad con el perfil original de Pharma. La aprobación de elegibilidad y el significado clínico se consultan mediante sus propios contratos. |

La armonización empieza con el producto existente de Step 10. El Parquet opcional de mediciones se consulta en modo lectura desde el mismo módulo. El pipeline se ejecuta desde `obj1_sjd` con los datos y recursos configurados. El candidato incorpora las reconstrucciones de historia serológica y HLA respaldadas por fechas de disponibilidad explícitas en los datos fuente. La fecha de laboratorio o de visita se conserva como procedencia. Las definiciones de baseline de Step 20b, los scores, Pop, Oxford y el consenso retrospectivo permanecen auditados con sus resultados originales cuando faltan componentes o reglas verificables.

El perfil de elegibilidad compara el master y el candidato mediante `obj1_sjd/src/studies/pharma/00_profile_labs.py`, con el registro original y el mismo contexto. Las features nuevas se reportan por separado con su alcance temporal y quedan sujetas a las reglas de entrada de cada estudio. El perfil también compara el catálogo exploratorio de Graph con sus reglas originales. Graph y sus configuraciones viven en `obj1_sjd/src/studies/graph/`. Los resultados de modelos Pharma, Graph y Economic se marcan como `not run` en esta ejecución de armonización.

## Entorno aislado y Biowulf

Usa Python 3.12 y un directorio autorizado de transferencia con los modelos y metadatos permitidos. `requirements.lock` se verificó con pruebas de CPU. `requirements-gpu.resolved.lock` fija la resolución Linux x86-64 de vLLM, Torch, Transformers y Harmony. La asignación Slurm y la compatibilidad CUDA/MXFP4 necesitan el smoke del job en el hardware destino. El modelo GPT-OSS se carga exclusivamente desde los pesos locales configurados.

```bash
python3.12 -m venv .venv-semantic
. .venv-semantic/bin/activate
python -m pip install -r src/semantic_harmonization/requirements-gpu.resolved.lock
bash src/semantic_harmonization/run_prepare.sh
sbatch src/semantic_harmonization/run_harmonization.sbatch
```

Los scripts se ejecutan sin argumentos. `run_prepare.sh` resuelve el root desde su ubicación. El trabajo Slurm usa `SLURM_SUBMIT_DIR` o un `OBJ1_ROOT` explícito y valida el root antes de iniciar Python. `SJD_SEMANTIC_VENV` permite usar un entorno existente. Configura rutas privadas en `src/semantic_harmonization/configs/local.yaml`. Las rutas relativas se resuelven desde `obj1_sjd`.

```yaml
paths:
  codebook: /ruta/autorizada/Consolidated_Codebook_all_columns.xlsx
  source_events: null
```

Los defaults en `configs/pipeline.yaml` incluyen GPT-OSS en `/data/salazarda/data/models/gpt-oss-21b`, SapBERT en el directorio hermano y la ruta privada de conocimiento. La preparación registra y verifica los hashes de los pesos y del vocabulario Harmony. Descarga SapBERT, IPD-IMGT/HLA y UCUM por revisión fijada. Si la transferencia del vocabulario Harmony no está permitida, se puede colocar el archivo oficial `o200k_base.tiktoken` en `<knowledge_dir>/harmony/`. La preparación verifica su SHA-256. Los releases autorizados de RxNorm, LOINC, SNOMED y protocolos pueden declararse en `resources.optional` con ruta, versión y administrador opcional. RxNorm se utiliza para coincidencias exactas sin ambigüedad. LOINC y SNOMED quedan registrados como fuentes disponibles y no generan códigos mediante similitud. Para un protocolo local de recuperación, el YAML debe declarar una lista `documents` con `rule_id`, `title`, `aliases`, `text`, `concept_id`, `assay_key` y, cuando correspondan, `method` y `units`.

Los protocolos que declaren un umbral o una escala ordinal necesitan coincidencia exacta de ensayo, método y `protocol_version`. Para un `positive_threshold` se declaran además `units`, el `operator` (`GT`, `GE`, `LT` o `LE`), el `value` decimal y, si el protocolo lo respalda, `otherwise: negative`. Una concentración censurada solo adquiere categoría si el intervalo completo satisface la regla. `ordinal_scale` contiene los grados documentados y sus valores enteros.

El job clínico trabaja con recursos locales verificados. Inicializa GPT-OSS y ejecuta su smoke cuando quedan propuestas residuales por evaluar o el gold requiere inferencia. Los textos que exceden la ventana permanecen íntegros en la procedencia con una abstención de inferencia. RAG recupera documentos del codebook, del registro y de los protocolos locales por coincidencias compatibles. Las representaciones SapBERT se calculan sobre conceptos y aliases. Las tablas de coincidencia UCUM, IPD y RxNorm respaldan solamente sus campos correspondientes.

## Fuente común de Pharma, Economic y Graph

Los consumidores siguen usando el Step 10 original con la configuración versionada. Una activación explícita se registra en el archivo local `configs/input_selection.local.yaml` del módulo.

```yaml
mode: candidate
run_id: <sha256 completo del run validado>
```

El resolver verifica el run completado, cada Parquet y sus sidecars contra `candidate_source_manifest.json` y `run_manifest.json`. Los roles `integrated`, `effective_integrated`, `effective_observations`, `effective_feature_manifest`, `typed_observations` y `semantic_facts` están disponibles en la misma selección. Para explorar texto y categorías literales, usa `effective_integrated`. Para codificar predictores, consulta los manifiestos tipados y los enlaces a hechos aceptados y aplica la elegibilidad del estudio. El run incorpora el perfil de Pharma 00 y las tablas de selección de Graph producidos sobre el mismo candidato. Pharma 01 y Graph 01/02 los consumen desde ese paquete. Graph 03 valida también la procedencia de Graph 02. Las etapas posteriores de Pharma comprueban los hashes de sus productos intermedios en los directorios del candidato. El adaptador `src.studies.economic.resolve_inputs` expone la misma versión a un futuro productor. Economic sigue planificado.

Los outputs de los estudios activados se escriben en subdirectorios del run. Para volver a los defaults, elimina el override local o usa `mode: original` y `run_id: null`. La selección de un candidato exige su publicación técnica completa. Las políticas de cobertura y aprobación clínica siguen en los estudios.

## Resultados y verificación

Los run IDs se derivan de los hashes de entradas, código, dependencias, configuración y recursos. Se verifican los checkpoints y se reanudan las extracciones ya validadas. Las salidas analíticas se publican bajo `data/analytic/semantic_harmonization/<run_id>/`, y el reporte y los artefactos de QC bajo `outputs/tables/qc/semantic_harmonization/<run_id>/` de `obj1_sjd`. Ambos deben tener el mismo ID. `run_manifest.json` del directorio de QC confirma una publicación completa. Un cambio de archivo o fallo técnico detiene la publicación. Los logs de Slurm residen en `work/semantic_harmonization/` en `obj1_sjd`. El inventario de traslado se registra aparte en `outputs/tables/qc/semantic_harmonization/migration_manifest.json`.

`paths.audit` permite incorporar el inventario HTML marginal como referencia de procedencia. `paths.gold` permite evaluar anotaciones independientes en JSONL con `mapping_id`, `family`, `group` y `expected`, más `expected_document_ids` si se evaluará recuperación. El grupo debe reunir variantes cercanas del mismo ensayo. El holdout preserva los grupos y la estratificación por familia. Sin anotaciones independientes suficientes, el reporte presenta métricas descriptivas y registra la evaluación clínica como pendiente.

Desde la raíz y dentro del entorno aislado, `python -m pytest tests/test_semantic_harmonization.py` usa fixtures sintéticos para parsers, inferencia verificada, integridad del candidato, caché y el reporte. Cuando Chromium y Playwright para Node están disponibles, el test también verifica búsqueda, filtros, paginación y exportaciones. El Parquet clínico, el codebook operativo y los pesos de NIH no estaban disponibles en el entorno de desarrollo. La ejecución clínica y el smoke de GPU requieren esas fuentes en Biowulf.

## Armonización semántica V2

El directorio QC incluye `semantic_segments.parquet` con los offsets Unicode exactos por fuente, `deterministic_facts.parquet` (M1) y `semantic_facts.parquet` (M4). Cada hecho usa un struct tipado y se enlaza por mapping al linaje original. Los registros de referencia, método y proceso permanecen junto a su evidencia en QC. Los Parquet analíticos incluyen `patient_visit_semantic_observations.parquet` con un enlace a los hechos por observación y `analytical_representation_manifest.parquet` con los roles y reglas de uso. Los enlaces prospectivos requieren disponibilidad documentada antes o durante el episodio.

La referencia V1 se busca por `comparison.baseline_run_id` bajo el directorio QC configurado. Para usar una ubicación local explícita, configura `paths.baseline` en `configs/local.yaml`. La comparación comprueba manifiesto, huellas de los insumos y política del inventario. La comparación del subconjunto V1 se reconstruye a partir del diccionario publicado. `baseline_residual_disposition.csv` y los CSV de cobertura se generan dentro del directorio QC restringido. Si falta la V1, el reporte registra esa ausencia. Si falta `paths.gold`, el estado de la evaluación clínica es `not performed`. El gold V2 usa `schema_version: semantic-facts-v2`, `mapping_id`, `family`, `group`, `expected_facts` y, opcionalmente, `segment_roles`, `expected_document_ids` y `adjudication`. El lector sigue aceptando las anotaciones legacy.

Desde la raíz del repositorio en un nodo CPU autorizado, verifica recursos, identidad y tests con el mismo contenedor, home y bind del launcher Slurm. El comando solo imprime el estado y la identidad.

```bash
module load apptainer
BASE=/data/salazarda
REPO="$BASE/data/obj1_sjd"
IMAGE="$BASE/containers/vllm-0.11.2.sif"
PYTHON="$BASE/sjd_container_env/bin/python"
umask 077
export APPTAINERENV_SJD_CANONICAL_REPO="$REPO"
read -r APPTAINERENV_SJD_CONTAINER_SHA256 _ < <(sha256sum -- "$IMAGE")
export APPTAINERENV_SJD_CONTAINER_SHA256 APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_DISABLE_TELEMETRY=1 APPTAINERENV_VLLM_NO_USAGE_STATS=1
export APPTAINERENV_DO_NOT_TRACK=1 APPTAINERENV_WANDB_MODE=disabled
container=(apptainer exec --cleanenv --home "$BASE/cache/container-home:/home/salazarda"
           --bind "$BASE:$BASE" --pwd "$REPO" "$IMAGE" "$PYTHON")
"${container[@]}" -m src.semantic_harmonization.preflight
"${container[@]}" -m pytest tests/test_semantic_harmonization.py
"${container[@]}" -m src.semantic_harmonization.audit_cpu
```

Tras el preflight, usa el launcher existente desde la raíz canónica y comprueba el estado final del job con `sacct`. El smoke de producción incluye el conjunto completo de hallazgos de crioglobulinas, la negación de inmunofijación, la abstención para ensayos no realizados y validación de los esquemas por perfil. El candidato requiere revisión científica antes de seleccionarlo para los estudios downstream.

### Reporte comparativo a partir de runs publicados

Desde la raíz del repositorio, en el entorno Python del módulo:

```bash
python -m src.semantic_harmonization.report_existing
```

El comando verifica los manifiestos de QC y de datos analíticos. Su alcance es regenerar el reporte y sus tablas a partir de un run sellado. Lee los artefactos publicados con dependencias de CPU y genera un HTML offline junto con tablas de progreso, comparación y pendientes. Guarda la salida en `outputs/tables/qc/semantic_harmonization_reports/<report_id>/`. El directorio y sus archivos son privados. Los runs fuente conservan sus hashes.

Si hay más de un V2 elegible, configura la selección en `src/semantic_harmonization/configs/report_selection.local.yaml`. El archivo es local y está excluido de Git.

```yaml
report_selection:
  target_run_id: null
  baseline_run_id: null
  previous_v2_run_id: null
  output_root: outputs/tables/qc/semantic_harmonization_reports
```

Usa el ID completo del V2 publicado en `target_run_id`. Con `baseline_run_id: null`, el comando usa `paths.baseline` o `comparison.baseline_run_id`. Un reporte sin V1 presenta Original y V2. Si difieren las huellas clínicas o la política de las variables compartidas, el explorador muestra escenarios separados. Los deltas pareados requieren además el mismo grano de ocurrencias. La proyección V1 se basa en su diccionario publicado cuando no existen hechos tipados.

El progreso usa unidades de trabajo reproducibles a partir de los segmentos originales, los contratos de campos y la evidencia determinística. El backlog coteja la cobertura con los hechos finales M4. La extracción semántica y los vínculos escritos en el candidato se pueden consultar en el detalle de cada variable. La sección Methods documenta los denominadores y los artefactos que faltan.


### Cierre operativo V2

Desde la raíz del repositorio autorizado en Biowulf, con recursos ya preparados, ejecuta:

```bash
sbatch src/semantic_harmonization/run_harmonization.sbatch
```

El pipeline revalida propuestas selladas contra las huellas de fuente, modelo, perfil, prompt, contexto de recuperación y linaje. La revalidación de hallazgos históricos también requiere citas literales y compatibilidad con los hechos presentes. En un entorno Python compatible, `python -m src.semantic_harmonization.run_pipeline` ejecuta la misma materialización y omite GPT-OSS si las reglas y las extracciones compatibles cubren el residual. La ruta con Slurm cubre los casos en que la inferencia local sigue siendo necesaria. El log queda en `work/semantic_harmonization/job-<job_id>.log`.

El reporte compara Original, V1, V2 anterior, V2 reciente y V2 final por identidad de mapping, huellas clínicas y linaje. Los IDs se fijan en `configs/pipeline.yaml`. `historical_work_crosswalk` conserva los IDs del trabajo congelado cuando se puede autenticar el reporte de referencia. Las fuentes ausentes o con distinto linaje se muestran como comparaciones no evaluables. `regression_comparison` registra ambas citas, equivalencias, restauraciones, retiros respaldados y pérdidas sin explicación. HLA distingue el token reportado, el locus de búsqueda, los grupos G/P y el grado de verificación IPD. Un prefijo compatible de IPD conserva la resolución reportada.

El launcher imprime los IDs del run y del reporte, las rutas del Parquet efectivo, del CSV efectivo y del HTML, y el estado del selector. `current_v2.json` se actualiza de forma atómica cuando los tres cruces son evaluables, el trabajo histórico está autenticado, los enlaces de materialización están completos y las comparaciones carecen de pérdidas clínicas sin explicación. El reporte indica qué comprobación impidió el cambio cuando conserva el selector anterior. `report_existing` usa ese selector cuando el YAML local omite `target_run_id`. Un ID explícito permite consultar una publicación anterior. La activación de estudios sigue usando `input_selection.local.yaml`.

| Asset en el run analítico | Uso |
| --- | --- |
| `10_integrated_longitudinal_clinical_episode_effective.parquet` | Tabla clínica efectiva lista para consulta. Mantiene el esquema completo de Step 10 y una fila por episodio. |
| `10_integrated_longitudinal_clinical_episode_harmonized.parquet` | Candidato con las columnas originales intactas y las proyecciones tipadas `harm__`. |
| `effective_observations.parquet` | Una fila por celda elegible de episodio y una por observación BTRIS configurada. Incluye original, efectivo, decisión, hechos, evidencia y disponibilidad. |
| `effective_feature_manifest.parquet` | Campos originales, roles, tipos Arrow, cobertura y uso permitido. |

Los artefactos QC `effective_value_audit`, `effective_variable_summary` y `effective_family_summary` cuentan decisiones por grano y mantienen por separado conservación, transformación, hechos tipados y disponibilidad. El HTML muestra el efectivo publicado junto al original y a los resultados estructurados. La retención del original deja pendiente la interpretación del residuo. Los campos estructurados sin evidencia conservan sus nulos.

Cada tabla de armonización se publica en Parquet y CSV, con `*.csv.schema.json` para tipos, número de filas y SHA-256. Parquet define los tipos canónicos. El CSV UTF-8 usa `\N` para null, conserva el texto vacío y agrega un apóstrofo a texto con prefijo activo en hojas de cálculo, apóstrofo inicial o barra invertida inicial. Quita ese primer apóstrofo para recuperar el texto original. Las demás celdas usan JSON canónico, con texto ISO para fechas y hexadecimal para bytes. El pipeline coteja las filas CSV con el Parquet antes de publicar. Los exports filtrados del HTML usan una presentación independiente para la hoja de cálculo.

Los hechos clínicos tipados van por identificador desde M4 hasta `patient_visit_semantic_observations`. Los enlaces `as_of_fact_ids` requieren disponibilidad asociada a la observación y anterior o igual al ancla. La cobertura por feature distingue representación descriptiva y evidencia temporal. Pharma 00 lee el manifiesto de features del candidato y publica un perfil separado. El catálogo Graph ejecuta un smoke de codificación de baseline con sus reglas de soporte. `downstream_feature_readiness` registra esas comprobaciones, los motivos de exclusión y el estado de consumo por modelo. Las ejecuciones de modelos se registran como `not run`. Los límites, rangos, comentarios, CSV clínicos, reportes y tablas de linaje permanecen en el entorno autorizado.
