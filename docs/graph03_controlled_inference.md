# Graph03: contrato inferencial controlado v1

Este cambio implementa las herramientas del plan de resolución de bloqueos. No
afirma que exista una fuente clínicamente equivalente de protocolo, fechas de
instrumento verificadas o un contraste viable en la cohorte real. Esas decisiones
requieren evidencia institucional y revisión humana. La validación de desarrollo
utiliza exclusivamente pacientes, mediciones y firmas inventados para pruebas.

La entrada nueva es `src/studies/graph/03e_graph_n3_controlled_inference.py`.
No cambia 03c/03d, sus contratos documentales, los estimadores históricos ni el
Mapper. No activa `allow_model_estimation` en Phase D. Se puede aprobar un solo
par representación/outcome sin aprobar todos los candidatos. Una fuente que no
puede verificarse queda bloqueada; no se elimina protocolo ni se fuerza un ajuste
alternativo. Una enmienda del estimando necesita otro contrato revisado.

## 1. Inventario sin filas clínicas

En Biowulf, usando el intérprete institucional y las rutas reales:

```bash
BASE=/data/salazarda/data/obj1_sjd
PY=/usr/local/Anaconda/envs/py3.10/bin/python
G02="$BASE/outputs/tables/studies/graph/02_graph_representation_sensitivity/runs/20261103T141918515545Z_fd43fde0"

"$PY" "$BASE/graph03_schema_inventory.py" --base "$BASE" \
  --graph02-run "$G02" --output "$BASE/graph03_schema_inventory.json"
```

`--master` permite corregir una ruta sin modificar el dataset. Se leen esquemas
Arrow, el encabezado del baseline y una lista permitida de campos del registro:
nombre, rol, origen, destino, dtype, alias y alcance temporal. No se serializan
estadísticas parquet, metadata pandas, conteos de pacientes, valores ni fechas.
Un archivo ausente se registra como `file_exists: false`. El inventario no se
sobrescribe. Su salida requiere revisión institucional antes de compartirla.

La evidencia de código localiza candidatos en
`10_integrated_longitudinal_context.parquet`, escrito junto al master por
`src/block_A/10_build_integrated_longitudinal_dataset.py::_write_audits`:

| Campo candidato | Clasificación y significado documentado |
| --- | --- |
| `spine__source_protocol` | CONTEXT; protocolo de origen del episodio |
| `pop__protocol` | CONTEXT; contexto interpretativo/protocolo de Pop |
| `pro__parent_protocol` | CONTEXT; pertenencia/contexto de protocolo PRO |

Estos significados no prueban equivalencia con la covariable de protocolo de
investigación del paciente. El registro real debe confirmar los nombres públicos.
El código de recuperación rechaza explícitamente una revisión con alcance
`visit_origin`. No fusiona todo el contexto dentro del master o baseline original.

## 2. Insumos restringidos

El análisis consume rutas explícitas de Graph02 y Graph03 Phase A congelados;
el lector existente comprueba su integridad, finalización y correspondencia de
poblaciones. No descubre un "latest" que pueda cambiar un estimando aprobado.
Todos los artefactos de Graph03 inventariados también se verifican.

Además necesita dos parquet institucionales:

- **Contexto de asignación:** `patient_id`, `clinical_episode_id`, el campo de
  protocolo revisado y su fecha real de asignación. Puede ser un contexto derivado
  upstream, si la fuente original requiere adjudicación. No se inventa esa fecha.
- **Mediciones originales:** `patient_id`, `clinical_episode_id`, `source_column`,
  `value`, `measurement_date`, `assay_id`, `unit`, `sample_id`. El episodio es el
  episodio basal oficial. Se admiten observaciones repetidas bajo las reglas
  revisadas que se describen abajo; no se combinan ensayos diferentes.

El baseline congelado debe conservar `patient_id`, `clinical_episode_id`,
`is_clinical_baseline`, `clinical_baseline_date`, `clinical_anchor_date` y, cuando
exista, `clinical_baseline_episode_id`. Se verifican unicidad, bandera y
correspondencia de fechas/episodio. Que baseline y anchor coincidan nunca verifica
la fecha del instrumento.

El v1 exige fechas normalizadas a UTC y una revisión explícita de zona horaria.
El límite es el instante basal oficial; mediciones posteriores son inelegibles.
Si el upstream sólo tiene precisión de día, no se inventan horas ni se desplaza
el límite para incluir información posterior. Otro criterio temporal requiere
un contrato que lo documente expresamente.

## 3. Preparar un plan sin aprobarlo

Ejemplo con variables que deben fijarse a rutas institucionales existentes:

```bash
"$PY" "$BASE/src/studies/graph/03e_graph_n3_controlled_inference.py" \
  --graph02-release "$G02/02_graph_release_manifest.json" \
  --graph03-run "$G03" --context "$CONTEXT" --measurements "$MEASUREMENTS" \
  --plan "$PRIVATE/plan.yaml" --output-dir "$PRIVATE/unsigned_templates" \
  --prepare-plan
```

Se crea un plan vacío con SHA de los insumos, validación e implementación y
plantillas sin firma para las siete secciones de los 40 pares candidatos. No lee
filas para preparar ese borrador. No concede ninguna aprobación. Tanto el plan
como el directorio deben ser nuevos; el padre de `plan.yaml` debe existir.

Completar `outcomes` **antes** de inspeccionar asociaciones: representación,
outcome, `requested_models`, `n1_basis`, `use_pop`, `topology_scope`,
`model_family`, `hypothesis_family`. A es obligatorio; D requiere C. C necesita
una base N1 reducida y revisada que pertenezca exactamente al panel definitorio.
No hay selección por p-values ni adición automática de todas las fuentes N1.
Los binarios requieren además `minimum_events_per_parameter` positivo, fijado
por el revisor estadístico. Se conserva la familia completa de contrastes
solicitados, incluidos los bloqueados, para FDR-BH.

El hash del plan se calcula con
`s03_n3_decisions.plan_sha256(plan)`: JSON canónico de todo el contrato excepto
las ubicaciones de decisiones y la lista de revisores. Los revisores deben estar
en `authorized_reviewers`, configurada por el responsable institucional. Cambiar
fuentes, código, familias, cohortes, covariables, modelos o umbrales invalida las
decisiones anteriores. Preparar pins no autoriza el contenido científico.

## 4. Decisiones humanas verificadas

Cada documento YAML/JSON contiene los campos de la plantilla y `details`:

- `schema_version`, `section`, `representation_id`, `source_column` exactos;
- `plan_sha256` y `input_sha256` completos;
- `decision: APPROVED_FOR_THIS_ESTIMAND` o `EXCLUDED_FOR_THIS_ESTIMAND`;
- `reviewer`, `review_id`, `human_signoff: true` y
  `institutional_signature_verified: true`, aportados por humanos;
- `evidence`: lista no vacía de archivos relativos al directorio de decisiones,
  cada uno con `path` y `sha256` real, y sin enlaces simbólicos.

El plan referencia cada documento mediante `section`, `representation_id`,
`source_column`, `path`, `sha256`. No se aceptan URLs, rutas absolutas, traversal,
documentos alterados, otro scope, evidencia ausente ni revisores no autorizados.
Los documentos y evidencias originales se guardan dentro de la infraestructura
institucional; los informes sólo conservan estados controlados y hashes.

**Límite explícito:** se verifica la integridad de la atestación y la autorización
declarada, no una firma criptográfica ni la identidad/competencia del médico. La
autenticación institucional de la firma debe ocurrir externamente antes de que
un humano marque `institutional_signature_verified`. El código nunca lo rellena.

Campos `details` exigidos por sección:

| Sección | Contrato mínimo |
| --- | --- |
| `protocol` | `equivalent_to_research_protocol: true`, `canonical_target: demo__protocol`, `assignment_scope: patient_research_protocol`, `multiple_protocol_rule: reject`, `code_dictionary_verified: true`, `assignment_date_verified: true`, `dates_normalized_to_utc: true`, `source_column`, `assignment_date_column`, `code_dictionary` explícito |
| `baseline` | `official_baseline_verified`, `episode_anchor_verified`, `era_date_semantics_verified`, todos true |
| `measurement` | `instrument_date_verified: true`, `date_semantics: instrument_measurement`, `selection_rule: latest_prebaseline`, `duplicate_rule: identical_only`, `baseline_window_verified`, `source_semantics_verified`, `missingness_and_indication_reviewed`, `timezone_reviewed`, `dates_normalized_to_utc`, todos true; `lookback_days` entero no negativo, `type` numeric/binario (`binary`), `assay_id`, `unit` |
| `independence` | `clinically_independent`, `source_lineage_verified`, `proxy_reviewed`, `discovery_exclusion_verified`, `same_assay_dependencies_reviewed`, todos true |
| `pop` | Si se solicita: `outcome_dependency_reviewed`, `semantics_verified`, ambos true; `crosswalk` explícito de las cinco categorías. Pop1–3 permanecen iguales; las dos etiquetas no clasificables sólo pueden conservarse o reconciliarse entre sí |
| `topology` | Para full: `full_structure_stable: true`, además de estabilidad real en membresías congeladas. Para R0 stable-only: `selected_before_outcomes`, `unchanged_thresholds`, `exclusion_bias_reviewed`, `sensitivity_design_approved`, todos true |
| `design` | `requested_models`, `n1_basis`, `hypothesis_family`, `model_family` idénticos al plan; `multiplicity_prespecified`, `common_complete_case`, `covariate_semantics_verified`, todos true; con C, `n1_basis_source_semantics_verified` y `reduced_n1_basis_verified` true |

Para binarios, `measurement.category_convention` debe tener tokens explícitos y
disjuntos `positive`, `negative`, `unknown`, `not_tested`. Tokens no revisados
bloquean el análisis. Unknown/no realizado no son negativos. ESSDAI/ESSPRI y sus
derivados no pueden validar Pop; las dependencias definitorias se vuelven a
comprobar contra el registro/panel congelado, aun con decisión humana positiva.
`NO_DOCUMENTED_DEPENDENCE` no concede independencia.

El contexto debe tener una fila por paciente/episodio, sin protocolos múltiples
o cambiantes, códigos fuera del diccionario, asignaciones futuras ni discrepancias
con una covariable canónica ya observada. Múltiples protocolos requieren una
adjudicación upstream nueva; no se selecciona el primero ni se usa una constante.
La vista derivada conserva exactamente las filas y claves del baseline.

Para instrumentos se elige la última medición elegible anterior/al instante
basal dentro del lookback aprobado. Sólo se colapsan duplicados exactamente
iguales; empates restantes requieren adjudicación. ANA no se colapsa por nombre;
dos outcomes ANA con el mismo ensayo original se rechazan. RF texto/valor no
pueden generar dos hipótesis independientes en una representación. Se conserva
la incertidumbre por indicación de biopsia y faltantes mediante revisión explícita.

## 5. Preflight y ejecución

```bash
"$PY" "$BASE/src/studies/graph/03e_graph_n3_controlled_inference.py" \
  --graph02-release "$G02/02_graph_release_manifest.json" \
  --graph03-run "$G03" --context "$CONTEXT" --measurements "$MEASUREMENTS" \
  --plan "$PRIVATE/plan.yaml" --decision-root "$PRIVATE/decisions" \
  --output-dir "$PRIVATE/preflight_run"

# Sólo modelos individualmente elegibles; otro directorio nuevo.
"$PY" "$BASE/src/studies/graph/03e_graph_n3_controlled_inference.py" \
  --graph02-release "$G02/02_graph_release_manifest.json" \
  --graph03-run "$G03" --context "$CONTEXT" --measurements "$MEASUREMENTS" \
  --plan "$PRIVATE/plan.yaml" --decision-root "$PRIVATE/decisions" \
  --output-dir "$PRIVATE/inferential_run" --execute
```

El preflight no estima. `--execute` no salta decisiones, integridad o soporte.
Las compuertas se informan simultáneamente: protocolo, baseline/era, medición,
temporalidad, independencia, Pop, topología, diseño y soporte. Se distinguen
`FAILED_WITH_EVIDENCE`, `UNASSESSED_BECAUSE_UPSTREAM_BLOCKER`, `NOT_REQUESTED` y
`NOT_ESTIMABLE`. Un soporte no evaluado no establece falta de potencia definitiva.

A/C pueden avanzar sin estabilidad topológica para sus análisis base. B−A y
D−C requieren selección topológica aprobada y soporte comunitario; C−A requiere
la base N1 revisada. Los modelos solicitados de un par utilizan una única máscara
de casos completos con la unión de variables requeridas. Cuando los contrastes
de comunidad son elegibles, se usa la población congelada cubierta, sin ties y
con asignación soportada. Si no lo son, los modelos base usan la población basal
completa elegible; la columna `analysis_population` registra esa decisión. No se
compara un modelo base de otra cohorte para obtener un delta.

Se conservan mínimos de 30 pacientes/modelo, 8/grupo/comunidad, 10 grados de
libertad residuales, rango completo, variación de outcome, eventos/no eventos y
separación completa/cuasicompleta para binarios. R2 nunca habilita contrastes
confirmatorios con comunidad; todo el release se etiqueta exploratorio. R0
stable-only es sensibilidad preregistrada y no aprueba la estabilidad de todo R0.

Se reutilizan `_fit`, `_model_metrics`, `_design` y `_delta_interval` de N3
incremental y los diagnósticos de soporte de Phase D. OLS/GLM usan HC3; el cambio
de R² ajustado tiene bootstrap pareado de pacientes con membresías congeladas.
FDR-BH usa la familia prespecificada incluyendo contrastes bloqueados. No se
descartan resultados negativos. Fallos de ajuste producen estados controlados,
sin texto de excepciones que pueda contener valores privados.

## 6. Release, privacidad y reproducibilidad

Cada directorio nuevo es 0700 y cada archivo 0600. `preflight.csv`, `models.csv`,
`comparisons.csv` y `blockers.csv` no contienen identificadores, fechas clínicas,
valores categóricos libres, nombres o prosa de revisores. Se suprimen conteos y
grados de libertad en todo el reporte para evitar reconstrucción complementaria.
El registro único de bloqueos contiene causa, evidencia controlada, responsable,
acción, estado humano/técnico y versión de resolución.

Cuando existen contrastes completos, las figuras PNG restringidas muestran todos
los cambios disponibles de R² ajustado/Brier, incluidos los negativos, con los
intervalos condicionales cuando son estimables. Su fuente es `comparisons.csv`;
no incorporan datos individuales ni habilitan publicación automática.

**`private/` sí contiene datos individuales**: vistas basales derivadas con
protocolo/medición aprobados y listas exactas de cohortes para reproducibilidad.
Permanecen dentro de NIH; no hay ruta de exportación, figura pública ni ZIP.
Una vista aprobada en fuente no implica que el endpoint haya aprobado independencia
o que pueda estimarse. Los outputs de estado también son INTERNAL_RESTRICTED.

`release.json` registra plan, código, decisiones, inputs y SHA antes/después de
todos los archivos Graph02/03. `artifact_sha256.json` inventaría todos los outputs
excepto a sí mismo. Se rechazan solapamientos, sobrescrituras y cambios de inputs
durante la ejecución, incluso al fallar. Un error elimina el release incompleto.

Conservar una copia del contrato, los documentos/evidencias y los insumos
identificados por sus hashes permite reproducir el análisis con el commit
registrado. No se modifican releases previos para añadir aprobaciones posteriores.
El cierre científico real puede ser un contraste aprobado y viable o una
conclusión institucional documentada de no estimabilidad. Un release bloqueado
por fuentes no permite afirmar que la cohorte carece definitivamente de soporte.

Pruebas sintéticas y regresión:

```bash
python -m pytest -q tests/test_graph03_n3_controlled.py tests/test_graph03_n3_resolution.py \
  tests/test_graph03_n3_phasec_adjudication.py tests/test_graph03_run_resolution.py \
  tests/test_graph03_n3_candidate_audit.py tests/test_graph03_n3_incremental.py \
  tests/test_graph_characterization_tiers.py tests/test_graph_pipeline.py
```

Se cubren protocolo ausente/incompatible, diccionario, cardinalidad, múltiples
protocolos, asignación futura, medición posterior, ANA, RF, Pop, ties/no cubiertos,
R0 stable-only, R2, rango, soporte, decisiones alteradas, FDR, no divulgación,
release estimable sintético y hashes de los releases históricos antes/después.
