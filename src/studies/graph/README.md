# Graph 01, 02 y 03

Graph02 descubre y evalúa topología basal. Graph03 caracteriza las comunidades
publicadas por Graph02. Graph03e analiza las mediciones N3 por día calendario.
Los resultados por defecto son **exploratorios**, sin validación clínica confirmatoria.

## Ejecución completa sin flags

Desde la raíz del repositorio, con los insumos Step10/Pharma00 ya producidos:

```bash
python src/studies/graph/run_graph_study.py
```

También funcionan individualmente sin argumentos:

```bash
python src/studies/graph/01_run_topological_phenotyping.py
python src/studies/graph/02_graph_representation_sensitivity.py
python src/studies/graph/03_graph_community_characterization.py
python src/studies/graph/03e_graph_n3_controlled_inference.py
```

La configuración principal es [configs/pipeline.yaml](configs/pipeline.yaml).
Incluye por referencia las configuraciones canónica e histórica, el protocolo de
experimentos, el mapa clínico y la validación de 03. Todos los análisis aplicables,
tablas y figuras PNG/PDF se generan en la misma llamada. No requiere generar
figuras o activar comparadores en un segundo comando.

Un comando sin flags sigue necesitando fuentes reales válidas. No crea datos de
ejemplo cuando falta Step10. El modo `clinical_release` requiere explícitamente las
aprobaciones vigentes. Graph01 y Graph02 buscan el perfil Pharma00 configurado.
Para instalaciones donde Pharma00 solo haya publicado paquetes versionados,
también se admite un paquete upstream inequívoco. Un manifiesto de fuentes
opcional permite indicar un perfil exacto (rutas relativas a ese JSON):

```json
{
  "schema_version": "graph_source_v1",
  "feature_policy": "phenotype_independent",
  "integrated": {"path": "master.parquet", "sha256": "<SHA256>"},
  "pharma_candidates": {"path": "profile/00_pharma_graph_lab_candidates.csv", "sha256": "<SHA256>"}
}
```

Se configura su ruta en `paths.source_manifest` del pipeline. La selección de
fuentes no cambia los contratos Pharma00 ni las compuertas de serología as-of.

## Graph03 N3 exploratorio en Biowulf

```bash
/usr/local/Anaconda/envs/py3.10/bin/python /data/salazarda/data/obj1_sjd/src/studies/graph/03e_graph_n3_controlled_inference.py
```

Los agregados de uso interno quedan en
`outputs/tables/studies/graph/03_graph_n3_controlled_inference/`.
El archivo `03e_RESULTADOS.md` reúne cobertura, perfiles, modelos estimables
y motivos de exclusión. Las mediciones originales se leen en sus ubicaciones
de origen. Las filas individuales permanecen en dichas fuentes.
El protocolo está versionado en [n3_exploratory_defaults.yaml](configs/n3_exploratory_defaults.yaml).

## Qué ejecuta Graph02

* S0–S4 conservan las funciones históricas de selección, escalado y PCA. S3 y S4
  mantienen idénticas columnas; S4 balancea grupos LAB. R0 es un alias exacto de S3
  y la referencia fija; R1 corresponde a S4. No se elige referencia por outcomes.
* R2 realiza MFA jerárquico por subdominio y dominio clínico transversal. D01–D07
  son candidatos; D08/D09 son contexto. El YAML contiene **candidatos pendientes**,
  con nombres exactos y sin inferencia clínica por coincidencias de texto. Sin
  aprobación clínica, R2 puede ejecutar como `EXPLORATORY_PROVISIONAL` si pasa
  elegibilidad técnica, temporalidad basal y al menos dos dominios activos.
  `clinical_release` exige aprobación humana y unidades/temporalidad verificadas;
  una fuente técnicamente insegura se excluye con razón. Las 16 fuentes salivales,
  oculares y SF-36 están mapeadas exactamente; el join usa la fuente original,
  incluso en codificaciones `__binary`/one-hot. El linaje conocido y los pesos por
  constructo evitan contar derivados como ensayos independientes.
* R3 calcula Gower por variable fuente mixta y usa MDS métrico sólo como lente.
  DBSCAN recibe la distancia Gower precomputada. R4 usa FAMD cuando hay nominales;
  los niveles one-hot de una fuente cuentan como una variable mixta.
* D1–D6 separan lente y espacio de clustering; la rejilla Mapper predefinida es
  6/8/10 cubos × 0.30/0.40/0.50 de solapamiento. Cada celda tiene submuestras
  preliminares según YAML (10 en este protocolo), sin afirmar estabilidad final. UMAP es secundario y produce una
  limitación explícita si no está instalado.
* R0/R2/R3 elegibles tienen **200 submuestras al 80% sin reemplazo**. Los otros
  escenarios tienen el presupuesto secundario de 60. Bootstrap clásico, separado,
  usa **100 réplicas para R0 y R2 estimable**, refita con multiplicidades y
  representa cada paciente único una vez en Mapper. R3 usa 200 submuestras solo
  si pasa cobertura y estructura preespecificadas; de lo contrario usa 60.
  Los filtros, imputación, escalado, reducción y topología se refitan en cada réplica.
* El matching es Hungarian uno-a-uno sobre uniones de nodos, con auditoría de
  splits/merges, denominadores de ejecuciones válidas y comunidades evaluables,
  intervalos Wilson y coasignación. Los empates, pacientes no cubiertos y
  comunidades sin soporte se conservan; una comunidad no se denomina rama.
* Se ejecutan ablations por familia/dominio aplicable, panel con unidades válidas,
  permutaciones que conservan marginales y máscaras, controles de observación y
  estratos administrativos, y k-medoids/aglomerativo con K=2–5 y refit por submuestra.

* A5/context_excluded excluye D08/D09 y roles contextuales **antes de cada refit**
  de selección, redundancia, imputación, escalado, PCA y Mapper. No altera R0.
  A1/D01–D07, A2/familia y A6/constructos documentados tienen refits completos
  y presupuesto secundario. No se selecciona la rejilla por mayor cobertura.
* M0 fija las máscaras a fuentes/constructos R0, colapsando codificaciones y
  derivados documentados. Conserva estados explícitos, incluidos `unknown_detail`,
  sin interpretar un dato ausente como negativo. N2a usa solo máscaras y distancia
  Hamming balanceada por familia; N2b solo administración verificada y Gower;
  N2c combina distancias con pesos 1/2; N2d balancea dominios/instrumentos.
  Máscaras idénticas y totalmente observadas/ausentes se auditan y no se perturban
  artificialmente. Sin variabilidad se registra `NOT_ESTIMABLE`. El N2 histórico
  sigue etiquetado como mezcla de ausencia y administración.
* M1 combina valores R0 e indicadores con pesos de energía 1/2 y refits; N5
  permuta filas completas de máscaras; N6 permuta dentro de estratos amplios de
  protocolo/era cuando hay soporte. Se usan 199 permutaciones preespecificadas,
  p=(1+extremos)/(B+1), distribución nula e intersecciones auditadas. Son pruebas
  exploratorias múltiples: no identifican MCAR/MAR/MNAR ni mecanismos causales.
* Regresión logística regularizada diagnostica cobertura con cargas por
  familia/dominio y, por separado, administración. Imputer/scaler se entrenan en
  cada fold de CV estratificada repetida. Los percentiles son distribución de CV,
  no un IC de observaciones independientes. Cada contraste reporta n, soporte,
  universo completo y asignaciones duras comunes; los no cubiertos no son clase.
* La auditoría de cobertura usa los cubos exactos y los nodos emitidos de Mapper,
  distinguiendo cubos pequeños, ruido DBSCAN, nodos descartados y razón no resuelta.
  Registra distancias kNN, soporte local, bordes de cubos, carga observacional y
  efectos con bootstrap por paciente; no reasigna pacientes. N3 usa bandas de
  edad fijas, y protocolo solo usa aliases estructurales documentados.

El comando sin flags genera todos los CSV, figuras PNG/PDF aplicables, resumen
con cifras de ejecución, inventario de figuras no estimables, pesos y auditorías.
El descriptor `02_graph_release_manifest.json` registra la referencia, las
representaciones elegibles y el estado científico. Los archivos por paciente son
de uso interno restringido.

## Archivos vigentes

Cada etapa reemplaza sus propios archivos en su carpeta de siempre. Las etapas
calculan primero en un temporal y publican el resumen al completar sus archivos.
En caso de error, queda el último resultado completo y se registra un
`*_failure.json` con el estado del intento fallido.

| Etapa | Carpeta bajo `outputs/tables/studies/graph/` |
|---|---|
| Graph01 | `01_run_topological_phenotyping/` |
| Graph02 | `02_graph_representation_sensitivity/` |
| Graph03 | `03_graph_community_characterization/` |
| Graph03c | `03_graph_n3_clinical_adjudication/` |
| Graph03d | `03_graph_n3_resolution_preflight/` |
| Graph03e y auditoría 03f | `03_graph_n3_controlled_inference/` |
| Ejecutor integral | `run_graph_study/` |

Graph01 conserva la jerarquía habitual de figuras, QC y logs bajo `outputs/`
y de intermedios analíticos bajo `data/analytic/studies/`. Graph02 y Graph03
publican las figuras junto a las tablas. Los CSV que respaldan las figuras
llevan el sufijo `_source.csv`.

Graph03 comprueba la cohorte, las membresías y el QC de la salida actual de
Graph02. El ejecutor integral corre Graph02, Graph03 y Graph03e en ese orden y
publica `graph_study_results_summary.md` con enlaces a los resultados de cada etapa.

## Qué ejecuta Graph03

* Flujo de pacientes, denominadores, tabla basal y missingness por comunidad.
* Roles por representación: definitorias, derivadas de definitorias, anclas
  independientes y confusores. ESSPRI no valida sus componentes ni ESSDAI/overlap
  el constructo usado para definir Pop. Los constructos circulares no entran en la
  tabla de validación independiente.
* Anclas estimables: Cliff's delta o OR, IC por remuestreo, p exploratorio y FDR
  por familia de hipótesis. Una característica exige FDR y tamaño mínimo de efecto.
  Las variables definitorias se describen, sin validación independiente.
* Correspondencia descriptiva Pop, ARI/NMI y asociaciones incrementales
  `endpoint ~ Pop + comunidad + ajustes` para endpoints independientes estimables.
  Son asociaciones dentro de la cohorte; no se anuncian como predicción externa.
* Elegibilidad longitudinal explícita y modelos mixtos parsimoniosos con
  `tiempo × comunidad`, intercepto por paciente y tiempo real irregular posterior
  a la fecha basal congelada. Se comprueban soporte, rango y convergencia.
  Transiciones observadas de Pop son descriptivas, no movimiento por el grafo.
* Sensibilidad de efectos de anclas a los descubrimientos refitados y congelados
  en 02, con distribución empírica y consistencia de dirección. Los IC de los
  modelos siguen siendo condicionales a las etiquetas estimadas; no son intervalos
  confirmatorios que incorporen toda la selección.
* MGUS/linfoma/HLA sólo con fuentes y codificaciones exactas aprobadas. Sin ese
  contrato, quedan explícitamente no estimables. No se entrena proyección externa.

## Diagnóstico y compatibilidad

Estos parámetros son opcionales:

```bash
python src/studies/graph/run_graph_study.py --pipeline-config /ruta/pipeline.yaml
python src/studies/graph/run_graph_study.py --preflight-only
python src/studies/graph/03_graph_community_characterization.py --graph02-release /ruta/runs/ID/02_graph_release_manifest.json
```

`--output-dir` permite indicar el directorio de salida. El preflight del
ejecutor integral comprueba las fuentes de Graph02. El preflight de Graph03
comprueba las salidas actuales de Graph02. Los comandos de diagnóstico guardan
su estado de preflight sin reemplazar el resumen del análisis completo.

Graph01 conserva su interfaz. Las funciones históricas Graph02 viven ahora en
`stage02/s02_legacy.py`, con exports de compatibilidad desde el script numerado.
La CLI histórica, incluidos `--config`, `--canonical-config`, `--quick-exploratory`
y `--allow-without-lab`, sigue disponible explícitamente:

```bash
python src/studies/graph/02_graph_representation_sensitivity.py --legacy --preflight-only
```

Los CSV históricos no se sobrescriben. La ejecución completa conserva también los
nombres principales de membresía, estabilidad, feature sets, resumen y pesos S4;
los artefactos normalizados del manifiesto son la interfaz de Graph03. El estimador
nuevo añade matching uno-a-uno y distingue subsampling de bootstrap; no se debe
comparar una métrica legacy sin comprobar método y denominador.

## Validación y privacidad

```bash
python -m pip install -r src/studies/graph/requirements.txt
python -m pytest -q tests/test_graph_pipeline.py tests/test_graph_representation_sensitivity.py tests/test_graph_bootstrap_qc.py tests/test_graph_feature_catalog.py tests/test_graph_lab_contract_v2.py tests/test_graph_lab_ingestion.py tests/test_topological_phenotyping.py
```

La CI utiliza únicamente fuentes sintéticas. Los CSV de pacientes, nodos,
coasignación y réplicas permanecen bajo `outputs/` ignorado por Git, dentro de la
infraestructura clínica autorizada; no se publican en GitHub. La aprobación del
mapa clínico, la comprobación de fechas/unidades reales y la ejecución definitiva
en Biowulf siguen dependiendo de fuentes y revisión clínica reales.

Ver [inventario y límites de implementación](../../../docs/graph02_03_implementation.md).

## Sensibilidad transversal a datos faltantes (Graph02)

El comando sin flags ejecuta además `R0-I`, `R0-D`, `R0-M`, `R2-I`, `R2-D` y
`R2-M`, definidos en [configs/missingness_methods_02.yaml](configs/missingness_methods_02.yaml).
Son sensibilidades no definitorias: mantienen las fuentes y pacientes de su
referencia. R0/S3 y R2 conservan sus cálculos, presupuestos de 200 submuestras y
100 bootstrap, tablas y membresías. R1, R3, R4 y S0–S4 reciben auditoría de
missingness, familias/dominios y comparabilidad de pares. R0 frente a R2 mezcla
selección, ponderación y geometría; ese contraste no aísla missingness.

| Variante | Pregunta y método | Límite explícito |
| --- | --- | --- |
| `-I` | Reconstrucción de bajo rango regularizada con pesos singulares ajustados por familia R0 o constructo → subdominio → dominio R2; después se ejecuta la representación original | No equivale a `missMDA::imputeMFA`. Paneles con nominales no soportados quedan `NOT_ESTIMABLE`, sin excluirlos o imputar sus niveles continuamente |
| `-D` | Diferencias robustas de resultados compartidos, una contribución por fuente/constructo, ponderación jerárquica | Mínimo dos familias/dominios y 40% de peso compartido; matrices incompletas quedan `NOT_ESTIMABLE`, conservando NA y su auditoría; nunca se rellenan pares |
| `-M` | Distancia clínica más desacuerdo de estados de disponibilidad del propio panel | Peso principal 0.25, contrastes 0.10 y 0.50. Peso cero conserva exactamente el objeto de referencia. M1 histórico no es un alias |

La reconstrucción minimiza residuo cuadrático observado ponderado con penalización
nuclear y rango máximo preespecificado. Inicializa con medianas del entrenamiento,
ajusta normalizaciones singulares sólo en ese entrenamiento y las congela durante
la reconstrucción. Alterna SVD con contracción fija del 10% del primer singular
inicial y restauración exacta de celdas observadas. Evalúa rangos 2/3/4 con diez
particiones de pacientes; alterna enmascaramiento artificial por fuente e
instrumento, al 10%. Los factores, límites, escalas y pesos se ajustan sin los
pacientes de evaluación; éstos sólo ajustan sus propios scores mediante ridge
0.10 sobre sus resultados disponibles. El criterio es MAE normalizado ponderado
por celdas ocultadas; nunca outcomes ni comunidades. Tolerancia relativa 1e-5,
máximo 200 iteraciones: falta de soporte, exceso de memoria o no convergencia
producen `NOT_ESTIMABLE`, sin fallback silencioso. Las observaciones originales
se conservan, las binarias mantienen sus niveles y las ordinales su orden
previamente validado upstream. No se acepta una etiqueta nominal como continua.

Las distancias `-D` pueden incumplir la desigualdad triangular. MDS métrico es
sólo una lente sobre una matriz completa; DBSCAN recibe la disimilitud original,
no las coordenadas MDS. Se reportan constructos/dominios compartidos, fracción
de peso, pares no estimables y pacientes sin vecinos fiables. Los cortes 25% y
50% son sensibilidades fijas, sin optimizar cobertura. Los niveles one-hot
cuentan una sola vez. Las máscaras se construyen antes de imputar, deduplican
linaje de fuentes y conservan `observed`, `missing`, `partial`, `unknown_detail`
y razones upstream disponibles. No se infiere una ausencia administrativa o
estructural cuando no está documentada. D08/D09 permanecen fuera del panel R2;
los bloques administrativos no entran en las máscaras clínicas.

Cada variante principal solicita 60 submuestras; `-I` añade 20 bootstrap clásicos.
Cada sensibilidad de peso/corte solicita 20 submuestras sin cruzar la grilla
Mapper. Imputación, pesos, escalas, máscaras, reducción y Mapper se reajustan en
cada réplica sobre el panel congelado; no se submuestrea una matriz completada
previamente. Son cálculos CPU. El coste
crece especialmente con la validación de rangos dentro de cada refit. No requieren
GPU. El test sintético incluye un panel de 159 × 98 y control de memoria.

En la carpeta de Graph02, los CSV `02_graph_missingness_*` contienen registro
de métodos, paneles, estados agregados, reconstrucción, soporte de distancias,
comparaciones geométricas y comunitarias, cobertura, estabilidad y QC. Se separan
`FAILED` y `NOT_ESTIMABLE`; una variante secundaria fallida limita el release y
no reemplaza la referencia. Los CSV internos de membresías, componentes de
distancia y soporte por pares no son productos públicos. Las siete figuras
PNG/PDF incluyen CSV fuente, denominadores y supresión de nodos/celdas menores
a `minimum_publication_cell`. El resumen técnico no declara fenotipos validados,
causalidad del proceso de observación ni progresión temporal.

El descriptor registra la función de sensibilidad, la versión del estimador
y los estados de las variantes. Graph03 usa sus agregados para el contexto y
las representaciones elegibles de Graph02 para la caracterización clínica.

Pruebas sintéticas y regresión:

```bash
python -m pytest tests/test_graph_missingness_variants.py tests/test_graph_pipeline.py \
  tests/test_graph_representation_sensitivity.py tests/test_graph_missingness_v2.py \
  tests/test_graph_clinical_domain_mapping_v2.py tests/test_graph_coverage_diagnostics_v2.py \
  tests/test_graph_bootstrap_qc.py -q
```

La ejecución real sin flags se intentó en el entorno de implementación del
2026-10-29 y se detuvo antes del análisis: falta
`data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet`.
No se generó ni publicó un release clínico ni se sustituyeron fuentes privadas
por datos sintéticos. La verificación funcional usa fuentes sintéticas con el
contrato Pharma00, comprueba identidad SHA256 de ocho salidas históricas al
activar las variantes y compatibilidad con el lector Graph03. La corrida real
completa queda pendiente de disponer de los insumos privados autorizados.

Referencias: Josse y Husson, *missMDA* (2016), DOI
[10.18637/jss.v070.i01](https://www.jstatsoft.org/article/view/v070i01);
[`missMDA::imputeMFA`](https://search.r-project.org/CRAN/refmans/missMDA/html/imputeMFA.html).
La validación con enmascaramiento artificial no identifica el mecanismo real de
missingness; estabilidad técnica no establece validez clínica.

### Graph03 C/D: ejecución autónoma

```bash
python src/studies/graph/03c_graph_n3_clinical_adjudication.py
python src/studies/graph/03d_graph_n3_resolution_preflight.py
```

Graph03c usa los resultados actuales de Graph02 y Graph03. Graph03d incorpora
el diagnóstico actual de Graph03c. Ambas fases son opcionales para el análisis
exploratorio. `OBJ1_SJD_ROOT` permite indicar otra raíz para los diagnósticos.
Graph03d comprueba el soporte de la topología estable de R0 y documenta los
endpoints pendientes.

### Graph03 E: inferencia condicionada a decisiones humanas

El nuevo contrato `graph03_controlled_n3_v1` añade inventario de esquemas,
vista basal derivada, decisiones por representación/fuente verificadas por SHA,
preflight con bloqueos simultáneos y ejecución individualmente condicionada.
03c/03d conservan su carácter documental. La configuración de producción no
prescribe ni aprueba outcomes. Consulte
[`docs/graph03_controlled_inference.md`](../../../docs/graph03_controlled_inference.md)
para preparación de planes, evidencia institucional, comandos y límites de
privacidad. No se ha establecido viabilidad inferencial en datos clínicos reales.
