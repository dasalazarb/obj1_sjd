# Armonización semántica clínica SjD

La V2 de `obj1_sjd` publica la tabla clínica Effective y sus sidecars tipados después de Step 10. Effective conserva el esquema Arrow, las claves, el orden y los nulos del original. Solo aplica correcciones completas, validadas y representables en el tipo de origen. En los demás casos conserva el valor literal y registra los hallazgos respaldados en las tablas tipadas.

## Entradas y productores

| Entrada | Productor y uso |
| --- | --- |
| `obj1_sjd/data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet` | `obj1_sjd/src/block_A/10_build_integrated_longitudinal_dataset.py`. La unidad es paciente y episodio. |
| `10_integrated_longitudinal_context.parquet` en el mismo directorio | Lo escribe `_write_audits` en Step 10. Se valida un conjunto idéntico de claves y se usa la metadata disponible por analito. La fecha de medición y el estado de referencia tienen significados distintos del tiempo real de disponibilidad y de un resultado explícito. |
| `paths.source_events` (opcional) | Parquet autorizado disponible para `obj1_sjd` con contrato de observación BTRIS. Step 01 documenta el origen upstream en `src/block_A/01_serological_profile.py`. El módulo lee esta entrada y escribe un sidecar con una fila por observación, los campos originales y el mapeo. |
| `Consolidated_Codebook_all_columns.xlsx` | Hoja `Consolidated_Codebook`. Configura la ruta si el archivo queda fuera de `obj1_sjd` o si existe más de una copia. |
| `obj1_sjd/src/studies/pharma/expected_lab_concepts.yaml` | Metadatos del registro usados en la recuperación de conocimiento local. |

La armonización usa los archivos existentes de Step 10 y lee el Parquet opcional de mediciones. Los sidecars de disponibilidad temporal conservan la procedencia de la observación. Solo registran disponibilidad por episodio cuando la evidencia documenta una fecha compatible con el ancla.

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

El job clínico trabaja con recursos locales verificados. Intenta inicializar GPT-OSS y ejecuta su smoke por perfil cuando hay texto residual elegible. Los perfiles que fallan recurren a los parsers y al original literal donde corresponda. Los textos que exceden la ventana permanecen íntegros en la procedencia con una abstención de inferencia. RAG recupera documentos del codebook, del registro y de los protocolos locales por coincidencias compatibles. Las representaciones SapBERT se calculan sobre conceptos y aliases. Las tablas de coincidencia UCUM, IPD y RxNorm respaldan solamente sus campos correspondientes.

## Fuente común de Pharma, Economic y Graph

Los consumidores siguen usando el Step 10 original con la configuración versionada. Una activación explícita se registra en el archivo local `configs/input_selection.local.yaml` del módulo.

```yaml
mode: candidate
run_id: <sha256 completo del run validado>
```

El resolver verifica cada activo registrado contra `candidate_source_manifest.json` y `run_manifest.json`. Para una publicación Effective, `integrated` y `effective_integrated` resuelven al mismo Parquet. Los roles de contexto y sidecars tipados conservan sus rutas verificadas. Para codificar predictores, consulta los manifiestos tipados y aplica las reglas de elegibilidad del estudio.

Pharma 00 puede perfilar el Effective en una ejecución posterior con el run seleccionado. Su directorio predeterminado de salida está bajo `outputs/tables/studies/pharma/00_profile_labs/semantic_harmonization/<run_id>/`. Los archivos de Pharma 00 y Graph se producen fuera de los directorios sellados de armonización. La selección de un run técnico no activa los modelos de estudio. Para volver a la fuente original, usa `mode: original` y `run_id: null`.

## Resultados y verificación

Los IDs se derivan de las huellas de los insumos, código, configuración, recursos y dependencias utilizados. En una reanudación, las propuestas guardadas en el checkpoint se contrastan con la evidencia y la recuperación vigentes. El Parquet y el CSV Effective aparecen en `data/analytic/semantic_harmonization/<run_id>/`. Los artefactos de QC están en `outputs/tables/qc/semantic_harmonization/<run_id>/`, donde `run_manifest.json` registra los hashes de ambas ubicaciones. El log de Slurm está en `work/semantic_harmonization/job-<job_id>.log` y termina con las dos rutas y el ID de la ejecución.

Ejecuta `python -m pytest tests/test_semantic_harmonization.py` desde la raíz para verificar con datos sintéticos el esquema, la conservación literal, el fallback por perfil, la reanudación y el bloqueo de una publicación con archivos inválidos.

## Armonización semántica V2

El directorio QC incluye `semantic_segments.parquet` con los offsets Unicode exactos por fuente, `deterministic_facts.parquet` (M1) y `semantic_facts.parquet` (M4). Cada hecho usa un struct tipado y se enlaza por mapping al linaje original. Los registros de referencia, método y proceso permanecen junto a su evidencia en QC. Los Parquet analíticos incluyen `patient_visit_semantic_observations.parquet` con un enlace a los hechos por observación y `analytical_representation_manifest.parquet` con los roles y reglas de uso. Los enlaces prospectivos requieren disponibilidad documentada antes o durante el episodio.

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

El smoke de producción valida de forma independiente `finding`, ANA, ANA narrativa, HLA y morfología CBC. Cada caso que falla admite un reintento. Un perfil que falla una prueba requerida queda deshabilitado para nuevas propuestas. `finding` es compartido por inmunofijación y crioglobulinas. Los resultados de cada intento y los conteos por perfil quedan en las tablas de QC. La publicación comprueba la integridad técnica y registra la evaluación clínica como `not performed`.

### Reporte comparativo a partir de runs publicados

Desde la raíz del repositorio, en el entorno Python del módulo:

```bash
python -m src.semantic_harmonization.report_existing
```

Este comando se ejecuta por separado del launcher y verifica los manifiestos de QC y de datos analíticos. Su alcance es regenerar el reporte y sus tablas a partir de un run sellado. Lee los artefactos publicados con dependencias de CPU y genera un HTML offline junto con tablas de progreso, comparación y pendientes. Guarda la salida en `outputs/tables/qc/semantic_harmonization_reports/<report_id>/`. El directorio y sus archivos son privados. Los runs fuente conservan sus hashes.

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

Con los recursos locales preparados, `python -m src.semantic_harmonization.run_pipeline` ejecuta la misma materialización. El valor predeterminado `model.inference_mode: auto` intenta la inferencia local cuando hay texto residual elegible. Configura `model.inference_mode: disabled` en `configs/local.yaml` para ejecutar solo los parsers y los recursos estructurados. El launcher deja `current_v2.json` intacto. Identifica la publicación por el run ID que aparece en el log.

| Asset en el run analítico | Uso |
| --- | --- |
| `10_integrated_longitudinal_clinical_episode_effective.parquet` | Tabla clínica efectiva lista para consulta. Mantiene el esquema completo de Step 10 y una fila por episodio. |
| `effective_observations.parquet` | Una fila por celda elegible de episodio y una por observación BTRIS configurada. Incluye original, efectivo, decisión, hechos, evidencia y disponibilidad. |
| `effective_feature_manifest.parquet` | Campos originales, roles, tipos Arrow, cobertura y uso permitido. |

Los artefactos QC `effective_value_audit`, `effective_variable_summary` y `effective_family_summary` cuentan decisiones por grano y mantienen por separado conservación, transformación, hechos tipados y disponibilidad. La retención del original deja pendiente la interpretación del residuo. Los campos estructurados sin evidencia conservan sus nulos.

Cada tabla de armonización se publica en Parquet y CSV, con `*.csv.schema.json` para tipos, número de filas y SHA-256. Parquet define los tipos canónicos. El CSV UTF-8 usa `\N` para null, conserva el texto vacío y agrega un apóstrofo a texto con prefijo activo en hojas de cálculo, apóstrofo inicial o barra invertida inicial. Quita ese primer apóstrofo para recuperar el texto original. Las demás celdas usan JSON canónico, con texto ISO para fechas y hexadecimal para bytes. El pipeline coteja las filas CSV con el Parquet antes de publicar. Los exports filtrados del HTML usan una presentación independiente para la hoja de cálculo.

Los hechos clínicos tipados van por identificador desde M4 hasta `patient_visit_semantic_observations`. Los enlaces `as_of_fact_ids` requieren disponibilidad asociada a la observación y anterior o igual al ancla. La cobertura por feature distingue representación descriptiva y evidencia temporal. `downstream_feature_readiness` conserva esa evidencia y registra los modelos como `not run`. Los CSV clínicos y las tablas de linaje permanecen en el entorno autorizado.
