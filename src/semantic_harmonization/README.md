# Armonización semántica clínica SjD

Este módulo de `obj1_sjd` prepara un candidato independiente después de Step 10. Conserva las columnas, valores, tipos, orden y claves originales. Añade campos `harm__` y tablas largas con resultados exactos, límites y rangos, serología, hallazgos, alelos y medicamentos. El diccionario conserva originales, evidencia, contexto del ensayo y frecuencias. El reporte HTML funciona sin conexión y el directorio de cada ejecución tiene permisos restrictivos.

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

El job clínico trabaja sin acceso remoto de inferencia y pasa un smoke de GPT-OSS y Harmony antes de procesar textos narrativos. Los textos que exceden la ventana permanecen íntegros en la procedencia con una abstención de inferencia. RAG recupera documentos del codebook, del registro y de los protocolos locales por coincidencias compatibles. Las representaciones SapBERT se calculan sobre conceptos y aliases. Las tablas de coincidencia UCUM, IPD y RxNorm respaldan solamente sus campos correspondientes.

## Fuente común de Pharma, Economic y Graph

Los consumidores siguen usando el Step 10 original con la configuración versionada. Una activación explícita se registra en el archivo local `configs/input_selection.local.yaml` del módulo.

```yaml
mode: candidate
run_id: <sha256 completo del run validado>
```

El resolver verifica el run completado, el Parquet y sus sidecars contra `candidate_source_manifest.json` y `run_manifest.json`. El run incorpora el perfil de Pharma 00 y las tablas de selección de Graph producidos sobre el mismo candidato. Pharma 01 y Graph 01/02 los consumen desde ese paquete. Graph 03 valida también la procedencia de Graph 02. Las etapas posteriores de Pharma comprueban los hashes de sus productos intermedios en los directorios del candidato. El adaptador `src.studies.economic.resolve_inputs` expone la misma versión a un futuro productor. Economic sigue planificado.

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
