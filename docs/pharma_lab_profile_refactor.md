# Pharma 00: contrato de observaciones y gobernanza (2026-10-16)

Este cambio implementa las instrucciones suministradas para Pharma 00. Conserva
los nombres públicos de Step 10 y los campos/etiquetas que consumen Pharma 01 y
Graph 01. No cambia el productor, los estimadores, los consumidores ni el mapa S4.
La base de referencia consultada en GitHub fue
`ddac3c74a4cdd0dd2d3ed74667cc034fba6a5cfd` de `dasalazarb/obj1_sjd`.

## Cambios por archivo y función

- `src/studies/pharma/00_profile_labs.py`: `lab_stem()` descubre exclusivamente
  `lab__...__value`/`__text`. `extract_lab_representations()` y
  `join_lab_context()` separan inventario y metadatos; validan claves no nulas,
  únicas y unión uno a uno, conservan todas las filas, orden, índice y columnas
  originales. No usan resultados, QC ni procedencia del contexto.
- `assess_lab_quality_and_coverage()` cuenta observaciones por representación,
  pacientes, basales y repeticiones por episodio y por **fecha clínica distinta**.
  Dos horas distintas del mismo día no constituyen dos fechas. No certifica que
  dos observaciones sean comparables en ensayo/unidad.
- `parse_result()` reconoce códigos exactos, censura con operador/límite/dirección,
  desconocidos y texto libre. No transforma el maestro ni imputa la censura.
  `POSITIVE`/`NEGATIVE`/`NEG` son reconocimiento léxico, no aprobación clínica.
  El orden de `TRACE`/grados exige lista específica aprobada para el ensayo.
- `decide_downstream_eligibility()` aplica compuertas independientes de soporte,
  aprobación documentada, compatibilidad del texto **original** con consumidores
  y metadatos. Los resultados infinitos se excluyen de los conteos puntuales y
  bloquean la promoción de la columna original a consumidores numéricos.
- `expected_lab_audit()` usa preferencia exacta, después alias exactos aprobados;
  informa ensayos relacionados sin fusionarlos ni elegirlos por subcadenas.
- `serology_profile()` inventaría las derivadas por separado; `text_code_audit()`
  exporta frecuencias de códigos reconocidos y agrupa comentarios en una etiqueta
  redactada. No exporta identificadores, comentarios libres ni filas de pacientes.
- `downstream_impact()` compara snapshots reales entregados mediante
  `--previous-profile`. `run()` genera tablas y JSON estables con hashes,
  denominadores, cobertura, drift de familias y recomendación de liberación.
- `src/studies/pharma/expected_lab_concepts.yaml`: registro versionado de los 30
  conceptos, reglas por ensayo y mínimos prospectivos. **Todas las aprobaciones
  clínicas iniciales son pendientes.** La prioridad de nombre no es equivalencia.
- `tests/test_pharma_lab_profile.py`: reemplaza conscientemente los fixtures
  anteriores sin namespace/referencia como resultado, incompatibles con el
  contrato real. Comprueba nombres públicos, parser, soporte, QC, unidades,
  privacidad, decisiones y campos históricos.
- `tests/test_pharma_lab_profile_integration.py`: esquema sintético con las
  cantidades documentadas, contexto, CLI, hashes, reproducción y lectores reales
  de Pharma 01/Graph 01, sin ejecutar sus modelos.
- `tests/test_graph_representation_sensitivity.py`: únicamente fixtures de
  compatibilidad, con seguimientos sintéticos y decisiones expresamente de prueba.
  Conserva el input basal de Graph; no modifica su código ni parámetros.

## Uso y decisiones configurables

```bash
# Archivar primero el CSV anterior fuera del directorio de salida.
python src/studies/pharma/00_profile_labs.py \
  --previous-profile /ruta/al/snapshot_anterior.csv

# Contexto explícito; su ausencia es un error.
python src/studies/pharma/00_profile_labs.py \
  --context /ruta/10_integrated_longitudinal_context.parquet

# Modalidad inequívoca de maestro solamente.
python src/studies/pharma/00_profile_labs.py --no-context
```

Sin flags, busca el contexto compañero al lado de `--integrated`. El modo queda
registrado en consola y JSON. Solo se incorporan `__unit`, `__reference_status`,
`__measurement_date`, `__days_from_anchor` y `__n_measurements` de `lab__`.
Se conservan los metadatos del maestro si ya existen; el contexto no los reemplaza.
Las claves de maestro sin contexto y las claves extra del contexto se cuentan
por separado. Un archivo de contexto con claves duplicadas/nulas se rechaza.
`reference_status` nunca se usa como resultado positivo/negativo.

Mínimos iniciales por representación: 10 episodios, 5 pacientes, 2 pacientes
con resultados interpretables en al menos dos fechas distintas; fracción numérica
finita parseable de `__value` >= 0.90. Son parámetros del registro (`--registry`),
no se ajustan para reproducir los 110 candidatos anteriores. El conteo histórico
`n_patients_with_ge2_results` permanece documentativo; no selecciona numéricos.
Los porcentajes basales son fracciones sobre basales realmente presentes.

Una aprobación debe registrar `approval_status: approved`, `decision_id`,
`reviewer`, `evidence` y el permiso de representación (`numeric_approved` o
`binary_approved`). No hay decisiones clínicas inventadas en el registro.
Con los estados pendientes iniciales, ninguna variable se promociona a un modelo.
Una señal numérica con soporte puede conservarse provisionalmente sin unidades
solo con aprobación explícita `allow_unassessed_units: true` respaldada por esa
misma decisión. Seguirá marcada `not_assessed`, nunca armonizada. La falta de
contexto **por sí sola** no fuerza `review_before_graph`.

Múltiples unidades o metadatos incompletos bloquean la promoción numérica.
La ausencia de QC es `not_evaluable_without_upstream_qc`; `has_conflict` queda
vacío/NA, no falso. QC parcial tampoco demuestra ausencia de conflictos.
La presencia simultánea de valor y texto se cuenta por fila, separando censura,
código y comentario; no se infiere discrepancia clínica sin reglas específicas.

Los consumidores actuales leen las columnas originales. Por ello un parser
aprobado no basta para promocionar texto mixto a categorías: comentarios,
censura, variantes de código incompatibles y ordinales aprobados requieren una
segunda fase de diseño del consumidor. El perfil registra
`requires_consumer_encoding_review` sin cambiar esos modelos. El único contrato
categórico promocionable aquí es binario crudo compatible y aprobado por ensayo.
HLA permanece genético/descriptivo, incluso si algún alelo parece un número.

## Auditoría explícita de los 30 conceptos

Esta tabla documenta **prioridades del registro**, no disponibilidad clínica
verificada. El CSV de ejecución informa coincidencia, variantes, fuente y revisión.
Todos los ensayos requieren decisión clínica; las equivalencias están pendientes.

| Concepto | Stem preferido (prefijo `lab__`) | Relacionados no equivalentes |
|---|---|---|
| anti_ro_ssa | anti_ro_ssa | anti_ro52, anti_ro60 |
| anti_la_ssb | anti_la_ssb | — |
| ana | ana_status | ana_hep2_status, ana_hep2_titer, ana_hep2_pattern |
| rheumatoid_factor | rheumatoid_factor | — |
| cryoglobulin | cryoglobulins | — |
| complement_c3 | complement_c3 | — |
| complement_c4 | complement_c4 | — |
| igg | igg | igg_total |
| iga | iga | — |
| igm | igm | — |
| esr | esr | — |
| crp | crp_high_sensitivity | crp (método distinto) |
| wbc | wbc | urine_wbc |
| hemoglobin | hemoglobin | urine_hemoglobin |
| hematocrit | hematocrit | — |
| platelet_count | platelet_count | — |
| anc | anc | — |
| lymphocyte_count | lymphocyte_count | — |
| protein_total | protein_total | — |
| albumin | albumin | — |
| gamma_globulin | gamma_globulin | — |
| protein_total_electrophoresis | protein_total_electrophoresis | — |
| creatinine | creatinine | — |
| bun | bun | — |
| ast | ast | — |
| alt | alt | — |
| alkaline_phosphatase | alkaline_phosphatase | — |
| bilirubin_total | bilirubin_total | — |
| glucose | glucose | estimated_average_glucose |
| hemoglobin_a1c | hemoglobin_a1c | — |

IgG/IgA/IgM no coinciden con anticuerpos terminados en esas cadenas.
ANA/Hep2 y SSA/Ro52/Ro60 generan `multiple_distinct_assays_review` si hay variantes;
la fuente elegida sigue siendo exclusivamente la preferida exacta. Si solo hay
Ro52/Ro60, SSA queda sin fuente seleccionada. hsCRP se identifica expresamente
como método específico. La existencia de crioglobulinas y la interpretación de su
texto son preguntas separadas.

Las derivadas `sero__` se clasifican en basal, acumulativa as-of, genética as-of
y consenso de paciente. Ninguna se agrega a candidatos Graph. Un consenso puede
usar observaciones futuras y queda prohibido como predictor basal/temporal;
los demás tipos tampoco se aprueban automáticamente sin verificar su construcción.

## Salidas, comparación e impacto downstream

Se conservan los tres CSV históricos. Se añaden:

- `00_pharma_serology_profile.csv`: una fila por columna derivada y riesgo temporal.
- `00_pharma_text_code_audit.csv`: frecuencias de códigos y texto redactado.
- `00_pharma_downstream_impact.csv`: `lab`, `old_status`, `new_status`,
  `old_source_column`, `new_source_column`, `old_eligible`, `new_eligible`, `reason`,
  `impact_pharma01`, `impact_graph01`, `requires_manual_approval`, más conteos de
  soporte numérico/categórico. Distingue entradas, salidas y permanencias.
- `00_pharma_profile_summary.json`: hashes de inputs, parámetros, claves sin
  contexto, recuentos de clasificación/cobertura y drift explícito de familias.

El snapshot antiguo **comunicado por el investigador**, no verificado aquí,
contiene 177 labs: 99 numéricos, 11 categóricos, 41 descriptivos y 26 excluidos;
110 candidatos. No se aportaron esos CSV ni el encabezado completo junto a esta
solicitud: se recibió solamente el documento de instrucciones. No se fabrican
filas antiguas ni un diff clínico a partir de esos totales. Sin `--previous-profile`,
el impacto antiguo es `unknown/not_comparable_previous_snapshot_missing`.

La prueba sintética de familias tiene 485 columnas únicas, 177 stems, 100 valores,
177 textos y 36 serologías (7 basales, 15 acumulativas, 7 as-of y 7 consensos).
Es un esquema construido para pruebas, **no el encabezado real suministrado en
otra revisión**. En ese fixture hay 175 ensayos vacíos, un texto clínico pendiente
y una señal numérica pendiente: 175 `exclude`, 1 `descriptive_only` y 1
`review_required`. Son resultados de prueba, no una estimación del nuevo perfil
clínico. La concordancia JSON con la referencia compara cantidades por familia;
no certifica identidad de todos los nombres del encabezado real.

El fixture de escasez verifica 5 valores frente a 443 textos: 5 episodios numéricos,
3 pacientes numéricos, 2 pacientes con repetición numérica por fecha y 221 pacientes
con repetición de cualquier resultado. Queda `limited_for_longitudinal`; los textos
no completan los mínimos numéricos. En la integración aprobada exclusivamente para
pruebas, solo glucosa numérica y ANA binaria llegan a los lectores de consumidores;
una señal pendiente y las derivadas de consenso quedan fuera.

**Comparación real y lista clínica de entradas/salidas: pendientes por ausencia de
Parquet y snapshots reales.** Antes de cualquier rerun deben revisarse las filas de
anti-Ro52, anti-Ro60, anticardiolipina IgG, banda monoclonal y linfocitos variantes;
ANA, crioglobulinas, anti-CCP, proteinuria y hemoglobina urinaria permanecen visibles
sin aprobar texto mixto automáticamente. No se anticipan sus conteos nuevos.

## Validación ejecutada y límites

Se utilizó Python 3.12 del entorno `/workspace/.venvs/obj1_sjd/bin/python`.

```text
python -m pytest -q tests/test_pharma_lab_profile.py \
  tests/test_pharma_lab_profile_integration.py tests/test_graph_lab_ingestion.py \
  tests/test_topological_phenotyping.py tests/test_studies_shared.py
71 passed

python -m pytest -q tests/test_graph_representation_sensitivity.py \
  tests/test_graph_bootstrap_qc.py tests/test_pharma_main_plots.py
41 passed, 3 failed
```

Los tres fallos se reprodujeron con Pharma 00 anterior y los fixtures anteriores
sobre la misma base remota (3 failed):

1. `test_five_audited_mappings_preserve_selection_stages`: cinco filas del CSV S4
   actual tienen cuatro campos frente al encabezado expandido; `mapping_status`
   queda vacío/NaN.
2. `test_approved_ace_and_hepatitis_surface_antibody_mapping`: el mapa actual pone
   ACE en `INFLAMMATION_TISSUE`, mientras el test espera `GRANULOMATOUS_DISEASE`.
3. `test_domain_ordinal_delta_preserves_score_levels`: fixture de Pharma sin
   `interval_days`, que exige su contrato de salida.

No se corrigen mapas ni modelos fuera del alcance. Las pruebas de sensibilidad
restantes usan solo datos sintéticos. La batería completa del repositorio no se
certifica: por ejemplo, `test_pharma.py` ya importa el ausente `01_run_pharma.py`.
`py_compile` y `git diff --check` también se ejecutaron.

`python src/studies/pharma/00_profile_labs.py --no-context` no pudo ejecutar la
reconstrucción real: `FileNotFoundError` del Parquet integrado en la ruta por
defecto. No se escribieron resultados clínicos. Los tests sí verifican hashes
antes/después, claves y valores idénticos, siete outputs reproducibles byte a byte,
texto redactado y lectura de CSV vacíos con los consumidores existentes.

**Recomendación: publicar el código para revisión; no liberar todavía los nuevos
CSV como selección clínica downstream.** Bloqueos: ejecutar con maestro/contexto
reales y snapshot archivado; revisar el diff por ensayo y su soporte; documentar
aprobaciones, comparabilidad de unidades y parsers; auditar temporalidad de
serología. Resolver el mapa S4 en un trabajo separado antes de sus reruns.
No se ejecutan ni autorizan automáticamente modelos reales de Pharma/Graph.
Las necesidades upstream quedan separadas: armonización documentada de unidades,
QC auditado y temporalidad/procedencia del conocimiento serológico.
