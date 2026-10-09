# longitudinal_graph: revisión de representación, topología y validación

Fecha: 2026-10-09. Base de `main`: `433ff9827938bd80dcd4e6aa11e6f9f27a628d27`.
SHA-256 inicial de Script 02:
`64092b0158953f6e2c28fc4fe1f2d21a9cb86e5234fc2949b6025eeb27de6d18`.
SHA-256 inicial de configuración:
`65f45745a162e6953316c6feea5a407bc6261191097f3b6e60663a9b4087788e`.

## Alcance y procedencia

La especificación suministrada describe resultados históricos de 159 pacientes,
497 visitas y 98 features. Los CSV/PNG históricos y los Parquet de Step 10 no están
en este entorno; esas cifras son referencias de la especificación, no resultados
recalculados. Todas las preguntas sobre la cohorte real quedan en
`not_run_missing_data`. No se ejecutaron 03 ni 04 sobre datos clínicos.

La revisión conserva la selección elegible de 01, el primario S3 con RobustScaler,
ponderación familiar, PCA con tope de 10 y clustering en los PCs retenidos. El
cubrimiento incluye ahora exactamente el máximo de cada eje, corrigiendo la
pérdida de bordes por redondeo. En una dimensión constante, las bandas coinciden;
la repetición se muestra en el QC, sin crear rangos artificiales nuevos.

No se reconstruyen episodios, fechas, ESSDAI, ESSPRI ni Pop. Las aristas siguen
siendo visitas compartidas; no son transiciones temporales. Las membresías siguen
siendo `equal_weight_per_supported_node`; `soft_membership_tau` es un parámetro
histórico que no modifica esos pesos. No se añadió una alternativa softmax sin
definición de ramas acordada.

## Cambios trazables

| Área | Antes en main | Revisión |
| --- | --- | --- |
| Valores originales | Auditoría hsCRP/urine desde integrado | 01 conserva raw y máscara de observabilidad; 02 alinea por keys y contrasta pares originales con imputados |
| Representación | Cinco sensibilidades; cargas puntuales | Registro exacto de features, transformaciones y parámetros; auditorías de cargas/familias, escala, scores y peso por paciente |
| Sensibilidades | S2, full80, urine, hsCRP log | Se conservan; se añaden sin hsCRP, sin PRO, peso por paciente, ε local, cubos 6/10 y rango robusto prespecificado |
| Subsets clínicos | No definidos | PRO-summary-only y features comparables omiten ejecución con motivo explícito hasta disponer de whitelist justificada |
| Cubrimiento | Rango min–max sin tabla de celdas | Tabla completa, celdas vacías, pacientes, ε, fallback, clusters, soporte, ruido y visitas fuera del rango |
| Estructuras | Comunidades por modularidad | Identidades distintas para componente y comunidad; `branch_id` vacío y `branch_status` pendiente de definición |
| Estabilidad | Comparaciones S2/S3 y sensibilidades | Matching por conjuntos, splits/merges/casos unmatched, ARI con denominadores y refits bootstrap de series completas de pacientes |
| Cobertura | Reconciliación agregada | Categorías exclusivas por visita y paciente, estratos y tabla de endpoints de intervalos originales |
| Confusión | V por visitas disponibles | Disponibilidad explícita y permutación de vectores de etiquetas por paciente, dentro de tamaños de serie iguales |
| Circularidad | Manifest previo al pruning | Features realmente usadas en primario y estado de independencia para 04 |
| Figuras | Misma figura bajo dos nombres | Todos los nodos en ambas figuras; componentes y comunidades con la misma geometría y leyendas diferentes |
| Publicación | Escritura directa | Bundle temporal comprobado, archivo externo de resultados anteriores y eliminación de sensibilidades obsoletas |
| dry-run | Argumento sin efecto en 01/02 | Valida inputs y describe escenarios; no ajusta Mapper ni escribe outputs canónicos |

### Bootstrap y significado de estabilidad

Cada réplica selecciona pacientes con reemplazo e incorpora todas sus visitas.
Las copias tienen keys sintéticas distintas; el soporte sigue contando pacientes
originales únicos. Se reajustan mediana, pruning, RobustScaler, balance familiar,
PCA, cubrimiento, DBSCAN, soporte y modularidad usando solo la réplica.
Los conjuntos de referencia y réplica se restringen a pacientes originales
muestreados comunes. Los matches usan Jaccard de pacientes y después visitas;
se permiten y registran merges/splits. Las copias discrepantes de una visita
quedan fuera del ARI duro. Las réplicas fallidas o sin nodos se cuentan como tales
y generan matches ausentes; no se eliminan del denominador silenciosamente.

Los criterios Jaccard ≥0.50 en ≥0.80 de réplicas elegibles están configurados como
criterios prespecificados. La tabla de componentes se guarda también bajo el
nombre solicitado `02_mapper_branch_stability.csv`, pero declara explícitamente
`connected_component_QC_only_not_defined_branches`. No prueba estabilidad de ramas.
La evaluación fuera de muestra y la definición clínica/geométrica de ramas quedan
pendientes, con estado explícito. Ningún resultado bootstrap autoaprueba 03.

La variante ponderada por paciente usa cuantiles ponderados de mediana/IQR y PCA
ponderado; cada paciente aporta masa total igual. Los empates de valores se agrupan
antes de interpolar cuantiles. El primario conserva el ajuste por visita.

### Administración, disponibilidad y unidades

El V por visita es descriptivo. La inferencia adicional permuta vectores completos
de etiquetas de pacientes con igual número de observaciones válidas, preservando
la dependencia dentro de esas series. Se informa falta de pacientes intercambiables
y no se publica un p-valor de χ² basado en visitas independientes. La asociación
de scores PC1/PC2 con variables administrativas se presenta descriptivamente.

Las unidades, límite de detección, técnica y era de hsCRP se marcan `unknown` si
no hay metadatos verificados. La configuración declara el estado de transformación
previa de Step 10; `unknown`, negativos, valores inválidos o fuente ausente impiden
log1p. La sensibilidad sin hsCRP no depende de esos metadatos.
La ausencia de una prueba no se interpreta como ausencia de enfermedad.
Los faltantes originales se cuentan como `missing_unknown`; no se inventa el
motivo `not_measured`. La auditoría de biomarcadores usa el manifest de 01 para
elegibilidad/scope y la lista real de 02 para el uso en construcción. Las anclas
retenidas fuera del Mapper todavía requieren revisión de dependencia por proxies.

## Reproducción y preservación de históricos

```bash
python -m pytest tests/test_longitudinal_graph.py -q
python -m pytest -q
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py --dry-run
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py --dry-run
```

Usar el entorno de dependencias del proyecto. En el entorno administrado de esta
revisión se utilizó `/workspace/.venvs/obj1_sjd/bin/python`.

Con datos reales, preparar un bundle aislado antes de actualizar el canónico:

```bash
python src/studies/longitudinal_graph/01_prepare_longitudinal_data.py --output-root work/longitudinal-review
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py \
  --state work/longitudinal-review/analytic/01_longitudinal_visit_state.parquet \
  --metadata work/longitudinal-review/analytic/01_longitudinal_visit_metadata.parquet \
  --intervals work/longitudinal-review/analytic/01_longitudinal_intervals.parquet \
  --feature-manifest work/longitudinal-review/tables/01_longitudinal_feature_manifest.csv \
  --output-root work/longitudinal-review
```

El manifest registra SHA de Git, script, configuración, entradas, semilla, fecha,
dependencias y checks. Para outputs canónicos, 02 archiva cada archivo previo
`02_*` bajo un directorio hermano `longitudinal_graph_history/<timestamp>/<kind>`
fuera de las carpetas canónicas; luego instala el bundle completo. No archiva ni
elimina 01/03/04. Una falla de cálculo o QC antes de publicación deja los outputs
anteriores intactos. La copia final no es una transacción atómica entre directorios;
un fallo de filesystem exige recuperar el bundle anterior desde el archivo.

Los CSV nuevos incluyen denominadores y estados explícitos. La figura principal
codifica componentes conectadas; la figura de macrostates codifica comunidades
y soporte. Ambas conservan posiciones por centroide de scores PC1/PC2, muestran
todos los nodos, tamaño por pacientes y ancho de arista por visitas compartidas.

## Validación y limitaciones

La suite longitudinal anterior tenía 21 tests; la revisión tiene 34, conservando
los anteriores y extendiendo las aserciones de escenarios para incluir los nuevos.
Todos pasan con fixtures sintéticos. Se comprueban keys, no fuga, PCA, exclusiones,
observabilidad, cover/ruido, soporte por paciente, pesos, empates, intervalos,
bootstrap, confusores ausentes, figuras distintas, gate y contrato 01→02→03→04.
Los dry-runs reales devuelven `not_run_missing_data` y `outputs_written: false`.

`python -m pytest -q` tiene dos errores de colección preexistentes: falta
`src/20_btris_visit_date_match_report.py` y falta
`src/studies/pharma/01_run_pharma.py`. La ejecución con
`--continue-on-collection-errors` permite comparar el resto con main: las fallas
ajenas a longitudinal_graph están presentes también en la base. No se modificaron
esos estudios para ocultar fallas. Los datos/figuras históricos tampoco se
revalidaron porque no fueron entregados como archivos legibles en este entorno.

Resultado completo con continuación tras errores de colección: **396 passed,
10 failed, 1 skipped, 2 errors**. Las diez fallas y los dos errores coinciden con
la base; un fallo adicional del checkout archivado de comparación desapareció al
restaurar su metadata de Git, y el test correspondiente pasó al repetirlo.
Los warnings son principalmente de dependencias y fixtures anteriores.

## Respuestas científicas de la cohorte real

| Pregunta solicitada | Estado y evidencia disponible |
| --- | --- |
| 1. ¿PC2 sigue dominado por hsCRP; cambia con log1p/exclusión? | `not_run_missing_data`; cargas y contribuciones por escenario están implementadas |
| 2. ¿PC1 sigue dominado por PRO; qué ocurre sin PRO/agregados? | `not_run_missing_data`; sin PRO implementado; agregados requieren whitelist clínica |
| 3. ¿Cuántos PCs necesita full80 y qué geometría cambia? | `not_run_missing_data`; full80 conserva target sin tope artificial y refita todo el Mapper |
| 4. ¿Persiste cube*_0 y cuál es su causa? | `not_run_missing_data`; ocupación, extremos, ε local y rango robusto permiten investigarlo; no se atribuye a un outlier sin evidencia |
| 5. ¿Qué ocurre con estructura principal, comunidades y componente aislada? | `not_run_missing_data`; matching independiente de IDs registra pérdida/fusión/división |
| 6. ¿Cuántas visitas/pacientes/empates/endpoints quedan? | `not_run_missing_data`; tablas exclusivas, soporte y endpoints con denominadores completos |
| 7. ¿Estabilidad de componentes/ramas y macrostates? | `not_run_missing_data`; bootstrap de pacientes implementado; definición de ramas y validación fuera de muestra pendientes |
| 8. ¿Asociación con protocolo/era/tipo/completitud? | `not_run_missing_data`; disponibilidad explícita y permutación de series por paciente; ausencia de variable no equivale a V=0 |
| 9. ¿Serologías de visita o anclas externas? | `not_run_missing_data`; se audita scope/elegibilidad sin incluir retrospectivamente variables basales o acumuladas |
| 10. ¿Estado del gate y blockers? | Corrida real `not_run`; el fixture nunca se aprueba. Definición/estabilidad de ramas, whitelist clínica, unidades/eras, validación de proxies y revisión investigadora siguen pendientes |

La ejecución técnica correcta permite revisar evidencia; no convierte comunidades
en subtipos clínicos ni autoriza automáticamente el flujo temporal.
