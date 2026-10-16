# Pharma 00 → Graph 01 → Graph 02

El contrato `graph_features_v1` inventaría cada columna real de Step 10 por su
nombre exacto. Las familias son LAB, SERO, PRO, EXT, CLINICAL y METADATA.
`analysis_mode: exploratory` permite fenotipado basal técnicamente elegible con
aprobaciones clínicas pendientes. `clinical_release` conserva las revisiones,
versiones, hashes y aprobaciones de ensayo del contrato clínico anterior.
Step 10, PCA, RobustScaler, Mapper, comunidades, ARI y bootstrap no se modifican.

## Ejecutar en Biowulf

```bash
cd /vf/users/salazarda/data/obj1_sjd
PY=/usr/local/Anaconda/envs/py3.10/bin/python

# Produce un run nuevo; imprime la ruta del perfil.
$PY src/studies/pharma/00_profile_labs.py

# Por defecto encuentra el último run Pharma; para fijar uno usar
# --graph-lab-candidates /ruta/al/run/00_pharma_graph_lab_candidates.csv.
$PY src/studies/graph/01_run_topological_phenotyping.py \
  --analysis-mode exploratory --feature-policy phenotype_independent --preflight-only
$PY src/studies/graph/01_run_topological_phenotyping.py \
  --analysis-mode exploratory --feature-policy phenotype_independent --quick-exploratory
$PY src/studies/graph/01_run_topological_phenotyping.py \
  --analysis-mode exploratory --feature-policy phenotype_independent

$PY src/studies/graph/02_graph_representation_sensitivity.py \
  --analysis-mode exploratory --feature-policy phenotype_independent --preflight-only
$PY src/studies/graph/02_graph_representation_sensitivity.py \
  --analysis-mode exploratory --feature-policy phenotype_independent
```

Los parámetros completos siguen en `config.yaml` / `config_sensitivity.yaml`.
`--quick-exploratory` etiqueta el run, usa 2 réplicas bootstrap y 19 permutaciones,
y desactiva homología persistente. Sirve para comprobar ejecución; sus resultados
no establecen estabilidad científica. Las dependencias son las habituales del
proyecto: pandas, NumPy, PyArrow, PyYAML, SciPy, scikit-learn, networkx,
matplotlib, statsmodels y kmapper; ripser si se habilita homología persistente.
La validación local se realizó con Python 3.12; el código usa sintaxis compatible
con Python 3.10, pero el intérprete Biowulf indicado no está disponible aquí.

Cada script escribe un directorio nuevo `runs/<UTC_uuid>/` bajo sus raíces
habituales. Graph admite `--output-dir` o `--run-id`; se rechazan destinos ya
usados. Pharma admite `--output /nuevo/directorio/00_pharma_lab_profile.csv`.
Pharma 01 encuentra los perfiles versionados y conserva `recommended_use` y sus
compuertas longitudinales originales. Los CSV antiguos no pueden sustituir un
catálogo nuevo: Graph comprueba versión y hashes de entrada/salidas de Pharma.
Para cambiar parámetros de S0, proporcionar configuraciones concordantes mediante
`--config` y `--canonical-config` en Graph 02.

## Selección e interpretación

* Cobertura basal canónica: 0.70, con el mismo denominador para números y
  categorías. Los datos no parseables, constantes y sin cobertura permanecen en
  el catálogo con su razón. El inventario nunca depende del mapa S4 ni del
  diccionario de conceptos LAB.
* Se evalúan valor y texto por separado. Para dos representaciones del mismo
  ensayo se prefiere la de mayor cobertura y, en empate, el valor numérico. Los
  nombres compuestos se conservan. No se fusionan Ro52/Ro60/SSA, IgG/IgG total ni
  CRP/hsCRP. `__unit` y `__reference_status` son contexto.
* Texto binario requiere un conjunto completo de códigos reconocidos. TRACE,
  1+, comentarios y censura no se convierten en ceros. Un ordinal textual requiere
  `ordinal_categories` del ensayo en el registro. Categorías nominales requieren
  niveles explícitos; missing queda como NaN antes de la imputación por mediana
  ya existente. Múltiples unidades sin armonizar excluyen técnicamente; unidades
  desconocidas se señalan para revisión/sensibilidad.
* `phenotype_independent` reserva POP, ESSDAI/ESSPRI totales y los componentes de
  ESSPRI; excluye ESSDAI/OVL ligados a la definición POP.
  `clinical_inclusive` incorpora dominios ESSDAI, overlap y componentes ESSPRI,
  conservando los totales como outcomes. Declara riesgo de circularidad. El total
  ESSPRI no constituye validación independiente de sus componentes. PRO usados en
  discovery sólo permiten interpretar sus trayectorias posteriores de forma
  condicional. `legacy_with_overlap` conserva la comparación con overlap y amplía
  el catálogo; no reproduce bit a bit experimentos anteriores. El flag histórico
  `without_overlap` se traduce a `phenotype_independent` y se registra así.
* SERO siempre tiene familia propia. Consenso por paciente nunca entra en basal.
  Baseline/as-of/cumulativos requieren metadatos por columna exacta en
  `feature_metadata`: `temporal_validation` con `no_future_data`,
  `no_criteria_circularity`, `knowledge_date_verified` verdaderos y `decision_id`
  documentado. Las nominales requieren además `encoding: nominal` y `categories`.
  No hay ninguna validación temporal por defecto. La sensibilidad con/sin SERO
  requiere dos configuraciones previamente revisadas; no se infiere equivalencia
  clínica con LAB de sus nombres.
* `age_dx` ausente no bloquea ni se sustituye por otra edad. Sólo
  `demo__age_at_baseline` es predictor demográfico basal del nuevo catálogo;
  `age_dx` del fichero externo queda fuera del embedding hasta verificar linaje.

S0 conserva la matriz canónica Graph 01 sin pruning adicional. Todos los
escenarios parten de ese universo seguro con cobertura canónica; S1 vuelve a
aplicar cobertura por familia (y la reajusta dentro de cada réplica), S2 añade la
reducción de redundancia existente, S3 balancea cinco bloques independientes
LAB/SERO/PRO/EXT/CLINICAL, S4 añade subgrupos sólo dentro de LAB. S3/S4 tienen
exactamente las mismas columnas y missingness; cambia el peso. MFA conserva
RobustScaler y normalización por el primer valor singular de cada bloque; S4
normaliza subgrupos y después el bloque LAB. Los pesos quedan en las auditorías.
Las familias vacías se notifican expresamente.

Schirmer derecho/izquierdo/mínimo y las otras medidas EXT se inventarían siempre.
El análisis canónico conserva lateralidades y derivados técnicamente válidos;
S2–S4 permiten comparar la reducción. Con cobertura igual y correlación alta se
prefiere Schirmer/TBUT mínimo y Oxford/Van Bijsterveld máximo. La auditoría de
redundancia muestra relaciones; correlación no establece equivalencia clínica.

Un mapa S4 incompleto produce su auditoría y bloquea únicamente S4. El run
completo termina S0–S3 y guarda sus resultados antes de fallar en S4. El preflight
examina todos los escenarios, escribe las tablas y devuelve error si S4 queda
bloqueado. No se inventan grupos para nombres desconocidos. Las cinco filas
truncadas del mapa suministrado se repararon conservando sus asignaciones.
Los 23 stems adicionales mencionados en la solicitud no se pueden completar sin
sus nombres reales; cualquier seleccionado sin mapa aparece como MISSING.
`--allow-without-lab` permite una sensibilidad explícita sin LAB para Graph 01 y
S0–S3; S4 sigue requiriendo LAB. Nunca se anuncia una modalidad ausente.

## Archivos para revisar y presentar

1. Pharma: `00_pharma_lab_profile.csv`, `00_pharma_graph_lab_candidates.csv`,
   `00_pharma_graph_lab_selection_audit.csv`, `00_pharma_feature_namespace_audit.csv`
   y `00_pharma_profile_summary.json`. La elegibilidad basal exploratoria es
   paralela a `longitudinal_eligibility`; no promueve aprobaciones clínicas.
2. Graph 01: `01_graph_all_source_columns_audit.csv` es la trazabilidad
   input→Mapper: `source_column`, familia, disponibilidad, observados basales,
   parseabilidad, elegibilidad técnica, cobertura, redundancia, inclusión PCA y
   causa de exclusión. Puede haber varias filas por fuente nominal one-hot.
   Revisar además manifest, embedding feature set, family/domain flow,
   redundancy audit y selection summary/preflight JSON.
3. Graph 02: `02_graph_source_to_mapper_trace.csv`,
   `02_graph_scenario_feature_sets.csv`, `02_graph_multimodal_family_flow.csv`,
   `02_graph_multimodal_domain_flow.csv`, `02_graph_S4_lab_group_mapping_audit.csv`
   y `02_graph_input_manifest_alignment.json`. Este último compara realmente la
   matriz S0 con una reconstrucción Graph 01 usando su configuración canónica.
4. Para resultados: varianza/cargas PCA, resumen Mapper/comunidades, QC de cada
   bootstrap, ARI sobre intersección común y Jaccard por separado para uniones de
   nodos y asignaciones duras. Los runs quick no son evidencia clínica.

Las auditorías y los fingerprints no contienen IDs ni filas individuales.
Los archivos analíticos históricos `*patient_embedding*` / `*membership*` sí son
salidas individuales: mantenerlos en el entorno clínico y no presentarlos como
artefactos de auditoría ni publicarlos. Preflight no genera esos archivos.

La ejecución sintética, las pruebas y las limitaciones de datos reales están en
[graph_feature_catalog_validation.md](../../../docs/graph_feature_catalog_validation.md).
