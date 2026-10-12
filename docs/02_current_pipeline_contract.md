# Contrato 01 → 02 → 03 y revisión metodológica

Fecha de revisión: 2026-10-12. Repositorio base:
`0c2130bd5523dab919211a5161008b1c7ef2cade`.

## Resultado y límites de esta implementación

Se añade un flujo de revisión S4 que **lee** una referencia S3-primary sellada,
genera corridas nuevas y permite revisar alternativas sin sustituir la referencia.
No hay selección automática. El análisis inferencial de Script 03 exige aprobación
científica documentada y vinculada a los hashes del run y sus membresías.

La lógica de ejecución sigue siendo **01 → 02 → 03 → 04**. Los módulos auxiliares
de revisión se agrupan en `src/studies/longitudinal_graph/_review/` y los importan
02/03 automáticamente; no son scripts ni pasos que el investigador deba ejecutar.
01 prepara, 02 construye Mapper y hace QC, 03 analiza flujo una vez aprobado y
04 caracteriza transiciones.

Este checkout no contiene los Parquet clínicos, CSV/JSON/PNG históricos ni
`longitudinal_graph_history/20261009T204529419045Z`. No se ha sellado ni reconstruido
la referencia clínica. Las cifras 159 pacientes / 497 visitas y 98 features son
datos del documento suministrado, no resultados recalculados. Ver
`02_reference_integrity.json` y `02_validation_test_report.md`.

Los SHA iniciales de **Script 02 y config.yaml sí coinciden** con el documento:

| Archivo | SHA-256 antes de editar |
| --- | --- |
| `02_run_longitudinal_mapper.py` | `328cd22348a0f279e58c5168b6e440add69695845dcb4830869fea74c2d5d1e6` |
| `config.yaml` | `352d8fc0e4be06a04a7a0535e3abda6549d1b2b9bd5e8725b9bb7edd5f02e841` |

El checkout inicial estaba limpio; el run descrito tenía `git_dirty=true`. La
coincidencia de estos dos archivos no prueba la reproducibilidad del resto del
estado histórico. Se conservó una copia del código inicial en el directorio de
trabajo de la sesión antes de editar. **config.yaml se conserva byte por byte**;
la política de candidatos está en `methodology_review.yaml`.

## Contratos efectivos

| Productor → consumidor | Artefactos / campos | Semántica y control |
| --- | --- | --- |
| Step 10 → Script 01 | `10_integrated_longitudinal_clinical_episode.parquet`, registro de variables | Episode clínico curado; no reconstruir fechas ni outcomes |
| 01 → 02 | `01_longitudinal_visit_state.parquet`: `patient_id`, `clinical_episode_id`, features finitas | Estado imputado con mediana histórica; exclusiones clínicas/temporales de 01 siguen vigentes |
| 01 → revisión S4 | Raw y observabilidad opcionales; metadata; feature manifest | Fuentes alineadas 1:1. Cero observado y faltante se distinguen. Carry-forward, fecha efectiva y ausencia estructural quedan `unknown` sin evidencia upstream |
| 01 → revisión de intervalos | `from_clinical_episode_id`, `to_clinical_episode_id`, `interval_days`, `interval_years` | Mismo paciente; extremos existentes; fechas, duración positiva y adyacencia comprobadas; sin saltar visitas no cubiertas |
| 02 histórico → 03 | `02_visit_mapper_membership.parquet`, prefijos `node_weight__`, `macrostate_weight__` | `mapper_covered` significa **macroestado soportado**, no cualquier nodo. Empates no se fuerzan a etiqueta dura |
| S4 → futuro 03 | `schema_version=longitudinal_mapper_v2`, `source_run_id`, contrato de intervalos y readiness | Membresías y cobertura separadas; datos exploratorios no autorizan inferencia |
| 03 → 04 | `03_patient_temporal_transitions.parquet`, feature manifest y representación efectiva | No se ejecutó 04. La adaptación de un candidato seleccionado exige revisión separada; no se cambiaron sus roles clínicos automáticamente |

Una visita entra al Mapper. Un paciente es la unidad independiente de remuestreo.
Un nodo contiene visitas; una arista comparte visitas y **no** es una transición.
Componentes conexas, comunidades y macroestados son objetos distintos. Ninguno se
convierte automáticamente en rama. `branch_id` permanece sin definir; H1 no está
establecida. La unidad paciente-basal y la agregación necesaria para T3 requieren
un estimando aprobado; el flujo no impone esa adaptación.

Las membresías históricas son `equal_weight_per_supported_node`. Pesos de nodo y
comunidad suman 1 en visitas cubiertas por macroestados soportados; fuera de ese
universo el peso largo es NA. `soft_membership_tau=1` sigue siendo un parámetro
histórico que no interviene en esta fórmula; no se afirma calibración probabilística.

## Seguridad y referencia

`reference.py` inventaría todos los archivos del archivo histórico, tamaños y
SHA-256. Escribe el sello **fuera** del archivo y no duplica ni modifica los
artefactos. Cuando existe el archivo histórico de 20261009, se apunta directamente
a él. El sello registra los hashes declarados de config, script e inputs y deja
pendiente la revisión de procedencia del código/entorno original.

G0 compara el inventario antes/después. Un archivo modificado, añadido o eliminado
detiene el flujo. Las rutas `history`, `longitudinal_graph_history`, `references`
y las rutas que solapan la referencia no admiten salida de runs. Se rechazan
symlinks en la referencia y IDs que contengan rutas.

Cada run se prepara en un directorio temporal del mismo filesystem y se publica
por rename tras completar el QC. Un lock exclusivo impide dos escritores del
mismo run. Si ya existe, falla. `--force` sólo admite un namespace llamado
`development`, nunca historia. El antiguo publicador también falla ante resultados
02 existentes: ya no archiva/elimina el canónico para sustituirlo.

```text
<output-root>/
  references/<reference-run-id>.json   # sello externo; archivo original intacto
  runs/<run-id>/
    run_manifest.json
    scenario_registry.json
    configuration_resolved.yaml
    02_temporal_readiness.json
    tables/ diagnostics/ analytic/ figures/ logs/
    scenarios/S4-.../
      02_graph_integrity_qc.json
      02_temporal_readiness.json
      02_interval_membership_contract.parquet
      ...
```

Este bundle contiene claves seudonimizadas de pacientes/visitas y fechas originales
y debe permanecer en almacenamiento institucional restringido. No subir datos
clínicos ni el archivo histórico al repositorio público. Los únicos resultados
de experimentos incluidos en `docs/` son agregados de fixtures sintéticos.

## Escenarios y preprocesamiento

| Escenario | Cambio aislado frente al comparador histórico |
| --- | --- |
| S4-visit-comparator | Ninguno; comparador recalculado con nombre nuevo, nunca S3-primary |
| S4-hscrp-log-robust | `log1p` de hsCRP; requiere unidades consistentes y transformación previa identity verificadas, observaciones no negativas y finitas |
| S4-hscrp-winsor | Winsorización p1/p99 aprendida sobre observaciones de entrenamiento |
| S4-no-hscrp | Ablación específica de hsCRP |
| S4-patient-mass | Masa total igual por paciente en escalador robusto ponderado y PCA ponderado; DBSCAN/cover y selección de epsilon siguen sin sample weights |
| S4-no-pro | Ablación PRO |
| S4-pro-summary | Lista blanca PRO explícita, disponible, con responsable y justificación; ausente → not_applicable |
| S4-common-features | Lista blanca clínica comparable, disponible y aprobada; ausente → not_applicable |
| S4-preimputation-pruning | Spearman sobre pares originales; mínimo predefinido de 20 visitas y 10 pacientes por par; prioridad cobertura/nombre, decisiones semánticas pendientes |
| S4-full80 | Meta PCA 80% aprendida en entrenamiento, sin tope histórico de 10 |

No hay candidato compuesto ni ganador preasignado. En el ajuste de cohorte se
mantiene fijo el estado imputado histórico de 01 y su poda, salvo el cambio
declarado. Los bootstraps vuelven a aprender transformación, mediana, selección,
escalador, PCA, cover, DBSCAN y comunidades **sólo en la muestra de pacientes**.
El cambio de pesos no añade una segunda regla de imputación: la mediana conserva
el contrato de 01. Cada copia de paciente remuestreada aporta una unidad de masa
en el fit ponderado; el soporte cuenta pacientes originales únicos.

La lista de variables usada, bounds, seed y parámetros quedan en el registro y
los manifiestos. La comparación ARI/NMI usa keys comunes con asignación dura
soportada sin empate en ambos escenarios e informa exclusiones y denominadores.
Una sola etiqueta produce `null/not_interpretable`. La tabla de todas las parejas
incluye matching óptimo uno-a-uno por Jaccard de pacientes y correspondencias
adicionales para revisar splits/merges; no reasigna IDs históricos.

## Observabilidad y validación

El export antiguo de aristas se construye después de filtrar nodos soportados en
`build_macrostates`; los grados se calculan sobre el grafo completo en
`topology_identity`. Ésta es la causa **en código** de los diferentes universos.
Se prueban 8 aristas completas / 5 soportadas con un fixture sintético que también
reproduce `{0:1,1:5,2:4,3:1}` y tres componentes. No se afirma haber reconciliado
el archivo clínico ausente. La carga de una referencia real valida membresías,
grados, componentes y endpoints/conteos de su CSV real antes de comparar.

Los nuevos exports declaran `graph_scope=full|supported`, motivo de filtrado,
componentes, articulaciones y peso. `n_shared_patients` cuenta pacientes de las
visitas efectivamente compartidas; `n_common_patients` es la intersección de los
conjuntos de pacientes de ambos nodos, que también define patient-Jaccard.
El CSV histórico conserva su significado anterior y no se reescribe.

PC1/PC2 se auditan con fracciones de loadings al cuadrado, familias y número
efectivo de features; esas fracciones **no** son varianza total del dataset. Las
distribuciones hsCRP, antes/después y bounds se exportan por escenario. No se
declara hsCRP errónea por su loading. Las observaciones originales no se imputan
en los informes descriptivos.

Protocolo se recupera de aliases documentados en `_shared.CONCEPTS` mediante
joins 1:1. Metadata tiene prioridad y la fuente integrada completa ausencias;
conflictos bloquean. Era no se inventa a partir de fechas. Variables ausentes
quedan `not_tested_missing_source`. V categórico es descriptivo; covariables
continuas usan eta² sin bins arbitrarios. Las permutaciones intercambian vectores
completos de etiquetas entre pacientes con igual tamaño de serie válida. Una
ausencia de flag o p no significativo no demuestra ausencia de confusión.

T3 permanece `not_tested`: faltan una agregación de visitas a pacientes, distancias
administrativas/clínicas aprobadas y una regla para componentes desconectadas.
No es justificable elegir estas decisiones después de ver asociaciones; no se
correlacionan distancias infinitas ni se presenta V como Mantel parcial.

Se exportan particiones exclusivas de cobertura, intervalos hard/soft, duración
y exclusiones; SMD de valores observados con IC por bootstrap de pacientes.
La caracterización usa la primera visita por paciente y comunidad, con denominadores
observados. Pacientes en varias comunidades se cuentan en cada una y no se suman
como partición. Roles y derivados bloquean la etiqueta de ancla independiente
cuando se usan en embedding. Los tests independientes y el valor condicional frente
a Pop requieren un plan clínico preespecificado, todavía ausente.

Los controles negativos intercambian trayectorias por **columna** entre pacientes
con igual número de visitas, respetando su orden cronológico, y repiten el fit
y bootstrap. Se reportan pacientes no intercambiables y controles sin cambio
geométrico. Están desactivados en la política de producción hasta fijar el presupuesto
de ese experimento; el test sintético sí ejecuta un control y un bootstrap por candidato.
Nunca se considera control negativo suficiente permutar sólo los IDs.

No hay mecanismo de proyección held-out. Su estado es `not_run`; el bootstrap
es in-sample. Las pruebas de fit/train-only no prueban generalización OOS.

## Reproducción

Usar el entorno del proyecto (en esta sesión: `/workspace/.venvs/obj1_sjd/bin/python`).

Script 02 admite el comando habitual sin flags:

```bash
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py
```

Genera un run ID único con timestamp UTC y busca primero un sello externo único
en `<output-root>/references/`. Si todavía no existe, reutiliza un archivo único
en `<output-root>/history/` o en el `longitudinal_graph_history/` contiguo a las
tablas estándar (incluida la ubicación del publicador anterior). La primera
ejecución completa crea el sello externo; ninguna escribe dentro del archivo.
El arranque muestra IDs, referencia elegida y destino antes del ajuste.

Si hay varias referencias no elige la más reciente: exige `--reference-run-id`
o `--reference-root`. Un ID explícito inexistente nunca se sustituye por otro.
Si no encuentra ninguna, imprime una explicación y las opciones necesarias,
termina con código 2 y conserva G0 bloqueado, sin traceback por IDs vacíos. Las
opciones explícitas de los ejemplos siguientes siguen siendo válidas.

Primero auditar/sellar el archivo existente y generar diagnósticos sin refit:

```bash
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py \
  --mode reference-only --run-id reference-audit-20261012 \
  --reference-run-id 20261009T204529419045Z \
  --reference-root /ruta/restringida/longitudinal_graph_history/20261009T204529419045Z \
  --freeze-reference --output-root /ruta/restringida/longitudinal_graph_review
```

La carpeta de referencia debe contener todos los outputs originales, incluida
la tabla Parquet de membresías por nodo. Si faltan, el run falla; no infiere
aristas clínicas desde una figura ni sustituye los outputs por cifras del documento.

Después ejecutar candidatos con los inputs originales de 01 y el Step 10:

```bash
python src/studies/longitudinal_graph/02_run_longitudinal_mapper.py \
  --mode candidate-qc --run-id candidates-20261012 \
  --reference-run-id 20261009T204529419045Z \
  --output-root /ruta/restringida/longitudinal_graph_review \
  --state /ruta/restringida/01_longitudinal_visit_state.parquet \
  --metadata /ruta/restringida/01_longitudinal_visit_metadata.parquet \
  --intervals /ruta/restringida/01_longitudinal_intervals.parquet \
  --feature-manifest /ruta/restringida/01_longitudinal_feature_manifest.csv \
  --integrated /ruta/restringida/10_integrated_longitudinal_clinical_episode.parquet
```

`--scenario S4-no-pro` puede repetirse para limitar el plan. `--dry-run` no crea
carpetas, sella ni ajusta Mapper; no requiere IDs y devuelve blocked si falta G0
o si el archivo descubierto todavía no está sellado.
La reproducción del algoritmo anterior usa `--mode legacy-reproduction`,
con un run nuevo y una referencia sellada (IDs y root admiten los mismos defaults
del arranque, o pueden fijarse explícitamente). Produce
`02_reproduction_diff.json`, crosswalk y universo comparado. No sobrescribe
outputs históricos y no constituye una reproducción certificada de un estado sucio.

```bash
python -m pytest tests/test_longitudinal_graph.py tests/test_longitudinal_methodology_review.py -q
```

## Gate y próximos pasos científicos

El flujo publica G0–G9 con estados/evidencia. G0/G2 pueden aprobarse técnicamente;
G1/G3/G5/G7 exigen revisar unidades, procedencia, dominancia, administración y
selección; G4 está bloqueado sin ramas definidas; G6/G8 no se han evaluado; G9
requiere la selección científica. Ningún pipeline crea automáticamente
`02_approval_record.json`.

Script 03 por defecto es `reference-only` y no calcula ni escribe flujo. El modo
`approved-temporal` verifica readiness, G0–G7, responsable, fecha, justificación,
escenario/run y hashes exactos de manifest, membresías, intervalos y config antes
de escribir. La aprobación ha de referirse al escenario seleccionado y su
inventario. Una inspección explícita `--allow-blocked-preview` se publica en
`previews/<run-id>` con marcador `EXPLORATORY_NOT_FOR_INFERENCE`, sin modificar
el gate oficial. Los resultados aprobados van a otro namespace create-only.

Pendientes para el investigador: estimando visita/paciente; vista integrada o
PRO/biología separadas; anclas reservadas y proxies; definición formal de ramas o
aceptación de continuo; recuperación protocol/era y observaciones originales;
separación selección-validación; método de T3 y proyección OOS. Estos pendientes
impiden declarar una topología fenotípica validada y mantienen Script 03 bloqueado.
