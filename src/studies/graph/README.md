# Graph 01, 02 y 03

Graph02 descubre y evalúa topología basal. Graph03 interpreta un release congelado y
verificado; no vuelve a seleccionar variables, imputar, estimar PCA ni ejecutar Mapper.
Los resultados por defecto son **exploratorios**, sin validación clínica confirmatoria.

## Ejecución completa sin flags

Desde la raíz del repositorio, con los insumos Step10/Pharma00 ya producidos:

```bash
python src/studies/graph/run_graph_study.py
```

También funcionan individualmente sin argumentos:

```bash
python src/studies/graph/02_graph_representation_sensitivity.py
python src/studies/graph/03_graph_community_characterization.py
```

La configuración principal es [configs/pipeline.yaml](configs/pipeline.yaml).
Incluye por referencia las configuraciones canónica e histórica, el protocolo de
experimentos, el mapa clínico y la validación de 03. Todos los análisis aplicables,
tablas y figuras PNG/PDF se generan en la misma llamada. No requiere generar
figuras o activar comparadores en un segundo comando.

Un comando sin flags sigue necesitando fuentes reales válidas. No crea datos de
ejemplo cuando falta Step10. El modo `clinical_release` requiere explícitamente las
aprobaciones vigentes; no se promueven ensayos pendientes. Las fuentes Pharma00 se
verifican por versión, hashes del master y hashes de salidas. Si hay más de un
bundle compatible distinto, se exige un manifiesto fuente anclado; no se elige por
fecha. Un manifiesto anclado tiene este esquema (rutas relativas a ese JSON):

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

## Qué ejecuta Graph02

* S0–S4 conservan las funciones históricas de selección, escalado y PCA. S3 y S4
  mantienen idénticas columnas; S4 balancea grupos LAB. R0 es un alias exacto de S3
  y la referencia fija; R1 corresponde a S4. No se elige referencia por outcomes.
* R2 realiza MFA jerárquico por subdominio y dominio clínico transversal. D01–D07
  son candidatos; D08/D09 son contexto. El YAML contiene **candidatos pendientes**,
  con nombres exactos y sin inferencia clínica por coincidencias de texto. Sin
  aprobación y temporalidad verificadas, R2 queda `NOT_ESTIMABLE`, sin bloquear R0.
* R3 calcula Gower por variable fuente mixta y usa MDS métrico sólo como lente.
  DBSCAN recibe la distancia Gower precomputada. R4 usa FAMD cuando hay nominales;
  los niveles one-hot de una fuente cuentan como una variable mixta.
* D1–D6 separan lente y espacio de clustering; la rejilla Mapper predefinida es
  6/8/10 cubos × 0.30/0.40/0.50 de solapamiento. Cada celda tiene 20 submuestras
  preliminares, sin afirmar estabilidad final. UMAP es secundario y produce una
  limitación explícita si no está instalado.
* R0/R2/R3 elegibles tienen **200 submuestras al 80% sin reemplazo**. Los otros
  escenarios tienen el presupuesto secundario de 60. Bootstrap clásico, separado,
  refita con multiplicidades y representa cada paciente único una vez en Mapper.
  Los filtros, imputación, escalado, reducción y topología se refitan en cada réplica.
* El matching es Hungarian uno-a-uno sobre uniones de nodos, con auditoría de
  splits/merges, denominadores de ejecuciones válidas y comunidades evaluables,
  intervalos Wilson y coasignación. Los empates, pacientes no cubiertos y
  comunidades sin soporte se conservan; una comunidad no se denomina rama.
* Se ejecutan ablations por familia/dominio aplicable, panel con unidades válidas,
  permutaciones que conservan marginales y máscaras, controles de observación y
  estratos administrativos, y k-medoids/aglomerativo con K=2–5 y refit por submuestra.

Las pruebas no aplicables tienen filas `NOT_ESTIMABLE` con motivos. Los fallos de
réplicas se registran y afectan QC; no se reintenta hasta obtener éxitos. Un grafo
sin estructura estable puede producir un release `NO_STABLE_STRUCTURE`: es un
resultado científico negativo, no un error computacional.

## Release inmutable 02 → 03

Cada ejecución crea un directorio nuevo `runs/<UTC_uuid>/`. Se rechazan IDs ya
usados. El manifiesto `02_graph_release_manifest.json` contiene esquema, SHA por
artefacto, fuentes exactas, configuración, versiones, referencia y representaciones
elegibles, población y QC. El puntero `latest_completed_02.json` sólo se publica
atómicamente tras finalizar y validar el release. Un run fallido no reemplaza al
último completado. La caché verificable de réplicas permite reanudar trabajo con
las mismas fuentes, configuración, experimento, semilla y muestra; cada publicación
sigue teniendo un directorio propio.

Graph03 valida hashes, esquema, estado, política, IDs y fuentes clínicas. Su modo
standalone usa el puntero; el runner conjunto pasa la ruta exacta recién generada.
No busca directorios `runs/*` ni importa los ejecutables numerados de descubrimiento.

El contrato diferencia membresía solapada de comunidad, asignación dura, nodos y
componentes conexas. Los scores Mapper son descriptivos, no probabilidades clínicas
calibradas. El CSV basal congelado conserva valores observados, sin imputación.

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

Las tablas de 03 incluyen `graph02_run_id` y `graph02_release_sha256`. El índice
maestro del estudio enlaza rutas relativas y hashes de tablas, figuras y logs de
ambas etapas. Los estados de ejecución son `COMPLETE`, `COMPLETE_WITH_LIMITATIONS`,
`PARTIAL`, `FAILED`; nunca se presenta una prueba no estimable como un éxito.

## Diagnóstico y compatibilidad

Estos parámetros son opcionales:

```bash
python src/studies/graph/run_graph_study.py --pipeline-config /ruta/pipeline.yaml
python src/studies/graph/run_graph_study.py --preflight-only
python src/studies/graph/03_graph_community_characterization.py --graph02-release /ruta/runs/ID/02_graph_release_manifest.json
```

`--output-dir` es la raíz de una colección de runs; `--run-id` fija un ID nuevo.
`--dry-run` equivale a preflight y nunca publica un release incompleto.
El preflight del runner comprueba las fuentes/contratos de 02; sin release nuevo
no ejecuta 03. Para comprobar un release ya publicado, usar preflight de 03.

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
