# Instrucciones para el agente — corrección de `02_graph_representation_sensitivity.py`

**Proyecto:** `obj1_sjd` · **Módulo:** `src/studies/graph/`
**Fecha:** 2026-10-15 · **Prioridad:** alta · **Idioma de implementación/comentarios:** preferentemente inglés; conservar este documento como especificación en español.
**Objetivo:** reparar el análisis de sensibilidad de representaciones (en particular S4), preservar la integridad estadística del bootstrap, y hacer que la salida distinga errores computacionales de inestabilidad topológica real. **No optimizar Mapper para conseguir un resultado favorable.**

## 0. Contexto y diagnóstico comprobado

Comando de ejecución observado:

```bash
/usr/local/Anaconda/envs/py3.10/bin/python \
  /data/salazarda/data/obj1_sjd/src/studies/graph/02_graph_representation_sensitivity.py
```

Salida reciente: 159 pacientes; escenarios S0–S4 con 80, 76, 75, 75 y 75 features; comunidades estables S0=1, S1=1, S2=1, S3=1, S4=0. ARI S0–S1=0.935; S1–S2=0.225; S2–S3≈−0.001; S3–S4=0.522. Esos ARI se computan sobre intersecciones filtradas de pacientes y **no** prueban estabilidad de las 159 asignaciones.

**Warnings relevantes:**

```text
Skipped S4 bootstrap replicate: S4 LAB feature has no mapped clinical group:
  feature=lab__angiotensin_converting_enzyme__value, lab=angiotensin_converting_enzyme
Skipped S4 bootstrap replicate: S4 LAB feature has no mapped clinical group:
  feature=lab__hepatitis_b_surface_antibody__cat_negative, lab=hepatitis_b_surface_antibody
Skipped S4 bootstrap replicate:
  LAB_GROUP:INFECTIOUS_SEROLOGY block has no positive first singular value
```

Otros avisos: `age_dx` no disponible en baseline (exclusión explícita, no sustituir por edad de visita); `numexpr` obsoleto para pandas; `FutureWarning` en concatenación de `redundancy_rows`. Los últimos dos son de mantenimiento, no la causa principal de S4.

**Lectura verificada del código actual en GitHub:**

- `SCENARIOS`: S0 base; S1 filtro de cobertura; S2 + reducción de redundancia; S3 + balance por familias; S4 + balance jerárquico LAB por grupos clínicos. S3 y S4 comparten el mismo conjunto de features.
- `bootstrap_stability()` remuestrea pacientes, repite selección de features, imputación/escala, PCA y Mapper; captura excepciones y hace `continue`, pero devuelve `scores / replicates`, contando los intentos fallidos en el denominador.
- `prepare_matrix()` exige un grupo mapeado por feature LAB, y `scale_mfa_block()` lanza `ValueError` si el primer valor singular es cero.
- `interpretation_check()` puede imprimir `YES` si al menos dos escenarios depurados tienen un componente conectado y una comunidad estable, **sin requerir reproducibilidad entre escenarios**.
- El mapa fijo `data/raw/graph_s4_lab_group_map.csv` no incluye los dos labs señalados arriba, aunque el bootstrap pueda volver a seleccionarlos.
- `config_sensitivity.yaml`: 60 réplicas, fracción de pacientes 0.80, Jaccard para match 0.50, umbral de estabilidad 0.80, PCA objetivo 0.80, Mapper `n_cubes=8`, `perc_overlap=0.40`.

**Consecuencia:** `S4 stable=0` no se puede interpretar biológicamente hasta saber cuántas réplicas fueron válidas. Si fallaran 33 de 60, incluso un match en todas las 27 válidas produciría como máximo 27/60=0.45 en la implementación actual. Esta cifra es un **ejemplo condicional**, no un recuento auditado de réplicas únicas.

## 1. Archivos a intervenir

| Archivo | Cambio esperado |
|---|---|
| `src/studies/graph/02_graph_representation_sensitivity.py` | Validación previa del mapeo; tratamiento de bloques sin variación; bootstrap con auditoría detallada; resultados y criterio interpretativo explícitos; warning de `concat` si procede. |
| `data/raw/graph_s4_lab_group_map.csv` | Completar grupos clínicos para **todo** laboratorio candidato elegible para S4 y no solo los seleccionados en el ajuste completo. |
| `src/studies/graph/config_sensitivity.yaml` | Agregar únicamente opciones de QC/auditoría necesarias, documentadas y con valores por defecto conservadores; no retocar parámetros de Mapper/PCA ni criterios científicos para mejorar cifras. |
| `tests/test_graph_representation_sensitivity.py` | Pruebas unitarias e integración para los fallos observados y para el cálculo correcto de estabilidad. |
| `src/studies/graph/README.md` o documentación metodológica existente | Añadir descripción breve de denominadores, tasa de éxito, mapeo y política de bloques constantes si el repositorio ya centraliza allí la documentación. |

No modificar `src/studies/pharma/`, `src/studies/economic/`, `src/studies/graph/01_run_topological_phenotyping.py`, los insumos `blockA`, ni los algoritmos de detección de comunidades salvo que una prueba muestre un defecto independiente y esté documentado. No reetiquetar ni excluir pacientes con el fin de aumentar ARI.

## 2. Tarea A — garantizar el mapeo completo antes del bootstrap (bloqueante)

1. Cargar el universo de features que puede entrar en **cualquier réplica**, no solo el `raw` final de S4. El código dispone de `full_manifest`, `full_values`, `feature_to_lab` y `lab_group_lookup()` en `run()`.
2. Extraer **todos los nombres canónicos LAB** de ese universo mediante el mismo contrato `feature_to_lab` / `canonical_lab_name()`. Mantener intactos stems compuestos (`acute_care_panel__...`, `immunofixation_...`).
3. Validar el mapa contra ese universo **antes** de iniciar S0–S4. Si falta algún lab, interrumpir de inmediato con la lista exacta y un mensaje para completar `graph_s4_lab_group_map.csv`. No dejar que el fallo se repita decenas de veces dentro del bootstrap.
4. Añadir explícitamente y justificar clínicamente estas filas pendientes:

```csv
lab,s4_group,s4_subgroup,mapping_status
hepatitis_b_surface_antibody,INFECTIOUS_SEROLOGY,HEPATITIS,MAPPED
```

Para `angiotensin_converting_enzyme`, **no inventar** silenciosamente un grupo por conveniencia numérica. Seleccionar y documentar con el investigador responsable una categoría clínica adecuada (p. ej., una categoría específica de biomarcador enzimático/diagnóstico diferencial si está justificada). Añadir la fila con `mapping_status=MAPPED` solo después de decidirla. Si no hay decisión clínica válida, el pipeline debe fallar tempranamente mostrando el lab pendiente.
5. Revisar **todos** los candidatos faltantes, no solo estos dos. Rechazar duplicados después de normalizar los stems. Conservar `lab`, `s4_group`, `s4_subgroup`, `mapping_status` como esquema mínimo.
6. Conservar los laboratorios fuera del modelo principal como **excluidos por las reglas de selección**; no introducirlos artificialmente en el conjunto S4 solo para que aparezcan en el mapa.
7. Generar/expandir `02_graph_S4_lab_group_mapping_audit.csv` con: `feature`, `lab`, `s4_group`, `s4_subgroup`, `mapping_status`, `mapped`, `eligible_for_bootstrap`, `included_S4` y `mapping_validation_result` (sin borrar columnas previas). Incluir auditoría de laboratorios excluidos del ajuste de S4 que puedan entrar en remuestreos.

**Invariante:** todo LAB seleccionable por el bootstrap tiene un grupo fijo, aprobado y trazable antes de iniciar las réplicas; ningún bootstrap falla por `unmapped clinical group`.

## 3. Tarea B — controlar grupos MFA sin variabilidad, sin sesgar la representación

Origen en `scale_mfa_block()` y el bloque `family == "LAB" and lab_group_balance` de `prepare_matrix()`:

- Actualmente `RobustScaler` transforma features imputadas; si un grupo completo es constante en una réplica, `sigma1 == 0` y el código aborta esa réplica.
- **No** usar `1 / epsilon` para reemplazar un valor singular nulo: produciría pesos arbitrariamente grandes y distorsión del espacio.
- **No** eliminar manualmente `INFECTIOUS_SEROLOGY` de todo el estudio, ni bajar los umbrales de estabilidad para solucionar el warning.

**Política recomendada (determinista y auditada):**

1. Identificar al calcular cada subgrupo `LAB_GROUP` si su matriz escalada tiene norma/singular principal numéricamente nula, usando tolerancia de precisión definida y documentada; no confundir variación diminuta real con nulo por una tolerancia excesiva.
2. Si **solo ese grupo** carece de variación, conservar sus columnas y su posición en la matriz, pero asignar a ese grupo un bloque transformado de ceros del mismo tamaño, con **peso efectivo 0** y `status=zero_variance`. No dividir por cero. No agregar señal inexistente. El grupo se excluye del cálculo de normalización entre **grupos activos**.
3. Combinar los bloques LAB y aplicar la normalización a nivel LAB **solo si** quedan grupos activos y el bloque LAB combinado tiene primer valor singular positivo.
4. Si **todos** los grupos LAB quedan degenerados, o si una familia completa no se puede construir, clasificar el remuestreo como **no estimable**; registrar motivo explícito, no falsificar un embedding.
5. Si la muestra completa (no bootstrap) fuera degenerada de forma equivalente, abortar el escenario con error claro. La política debe documentar cuándo un grupo constante se conserva como cero y cuándo el análisis se considera inviable.
6. Auditar por réplica: `LAB_GROUP`, número de features, número de pacientes, sigma1, estado (`active`, `zero_variance`, `invalid`), peso nominal/efectivo y número de grupos activos.
7. Preservar el orden, nombres y dimensiones del conjunto de características; las representaciones S3 y S4 deben seguir teniendo las **mismas features y missingness de origen**, aunque pesos y PCA difieran.
8. Verificar que valores finitos, peso positivo de grupos activos y ausencia de división por cero se cumplen en los bloques principales y remuestreados.

**Nota científica:** la ausencia de variación en una réplica es una propiedad de la muestra; su contribución es cero y debe quedar documentada. Cualquier cambio en las ponderaciones MFA constituye una decisión de representación y debe estar descrita en Methods y en la auditoría.

## 4. Tarea C — rehacer el contrato de bootstrap y el cálculo de estabilidad (bloqueante)

Modificar `bootstrap_stability()` para que retorne **estadísticas y estado de ejecución**, no solo `np.ndarray`. Mantener retrocompatibilidad interna mediante adaptación explícita de `run()` y de las pruebas.

### 4.1 Contadores y trazabilidad

Para cada escenario registrar:

- `n_requested`, `n_attempted`, `n_success`, `n_failed`.
- `success_fraction = n_success / n_requested`.
- Una fila por intento con `scenario`, `replicate_id`, `seed` o identificador reproducible de selección, `n_sampled`, `status`, `failure_stage`, `failure_type`, `failure_message`, `n_features_selected`, `n_communities`, `n_supported_communities`.
- Al menos categorías de error: `MAPPING_ERROR` (debe ser fallo previo bloqueante), `ZERO_VARIANCE_BLOCK`, `FEATURE_SELECTION`, `PCA`, `MAPPER`, `OTHER`.
- Fallos inesperados (`KeyError`, bugs de esquema/programación, etc.) **no deben ser silenciados** por un `except Exception` genérico.

### 4.2 Reportar DOS estimadores, no ocultar fallos

Para cada comunidad de referencia `c`:

```text
matched_successes[c] = # réplicas técnicamente válidas con
                       mejor Jaccard >= branch_match_jaccard
stability_conditional[c] = matched_successes[c] / n_success
stability_conservative[c] = matched_successes[c] / n_requested
```

Si `n_success == 0`, ambos valores derivados del éxito **no son evidencia de estabilidad**: `stability_conditional = NaN`, estado `INVALID_BOOTSTRAP` (aunque `stability_conservative` pueda registrarse como 0, no usarlo como estimador científico).

`stability_conservative` se conserva para transparencia y comparabilidad con el código previo, **no** como estimación directa de la estabilidad topológica. `stability_conditional` está condicionada a remuestreos computacionalmente válidos y puede estar sesgada si los fallos dependen de la composición muestral.

### 4.3 Puerta QC obligatoria

- Definir una política explícita, reproducible y documentada de tasa mínima de éxito (p. ej. **95% como umbral operativo de QC**, no como umbral de validez biológica ni parte retroactiva del pre-registro); un valor distinto requiere justificación.
- Si la tasa cae por debajo del mínimo, emitir `BOOTSTRAP_QC_FAIL` y reportar valores solo como diagnósticos; **no declarar ninguna comunidad estable ni inestable** a partir de ese bootstrap.
- Si se supera la puerta QC, aplicar `stable_community_threshold=0.80` al estimador primario de estabilidad **preespecificado y documentado** (`stability_conditional` solo bajo tasa de éxito adecuada). Reportar también el conservador.
- Registrar `n_requested`, `n_success`, puerta QC y denominador junto a cada comunidad.
- No «reponer» réplicas fallidas de forma no documentada hasta obtener una cifra deseada; si se implementan reintentos, fijar un presupuesto de antemano, semillas auditables y el criterio exacto.
- Añadir `bootstrap_qc_status` al resumen. Si falla el QC, `n_stable_communities` debe ser `NA` (no `0`), y el mensaje `Stable: INVALID_BOOTSTRAP` o equivalente.

**Conservar:** remuestreo de pacientes sin reemplazo, fracción de 0.80, refit completo del pipeline en cada réplica, Jaccard≥0.50, umbral 0.80 y las reglas de asignación de la implementación principal. No convertir el bootstrap en uno de features fijos por comodidad.

### 4.4 Nuevos outputs mínimos

- `02_graph_bootstrap_replicate_audit.csv`: una fila por intento y escenario, éxito/error y metadatos.
- `02_graph_bootstrap_qc_summary.csv`: éxito, fallos por causa, tasa y estado QC por escenario.
- Extender `02_graph_scenario_community_stability.csv` con `matched_successes`, `n_bootstrap_requested`, `n_bootstrap_success`, `bootstrap_success_fraction`, `stability_conditional`, `stability_conservative`, `bootstrap_qc_status`; preservar `bootstrap_stability` aclarando su definición.
- Extender `02_graph_representation_sensitivity_summary.csv` con `bootstrap_n_success`, `bootstrap_n_failed`, `bootstrap_success_fraction`, `bootstrap_qc_status` y permitir `NA` semántico para estabilidad no evaluable.

## 5. Tarea D — reparar `interpretation_check()` y separar tres preguntas científicas

Actualmente devuelve `YES` si hay conectividad + al menos una comunidad estable en al menos dos escenarios depurados; esa condición **no verifica reproducibilidad de particiones**. Separar en campos distintos:

**(i) Robustez computacional:** ¿los cinco escenarios corrieron y aprobaron el QC bootstrap?

**(ii) Persistencia de núcleo:** ¿existe al menos una región/agrupación reproducible entre escenarios depurados, expresada en Jaccard entre conjuntos de pacientes, tamaño de conjuntos y cobertura? Reportar **qué parejas** la apoyan y cuáles no; no basta con llamar `C0` al núcleo en cada escenario porque los identificadores de comunidad no se alinean semánticamente.

**(iii) Robustez de estructura global / múltiples ramas:** ¿cuántas comunidades superan el umbral de estabilidad en cada escenario? ¿se conservan las particiones al cambiar representación? Reportar ARI entre escenarios, n común y proporción cubierta; el hallazgo «un núcleo estable» **no equivale** a ≥2 ramas reproducibles.

La función debe devolver resultados interpretables como `COMPUTATIONAL_QC_PASS/FAIL`, `CORE_EVIDENCE=SUPPORTED/PARTIAL/UNSUPPORTED/NOT_EVALUABLE` y `GLOBAL_PARTITION_ROBUSTNESS=SUPPORTED/PARTIAL/UNSUPPORTED/NOT_EVALUABLE`, o estructura equivalente. Una salida `YES` sin estas precisiones **no es aceptable**.

No establecer umbrales de similitud retrospectivamente con los resultados conocidos. Si se necesita criterio de decisión adicional para Jaccard/ARI, declararlo antes de interpretar la nueva corrida e incluir justificación. Evitar «CONFIRMED» si S4 falla QC o si hay discordancia entre S1–S2, S2–S3, etc.

**Distinción metodológica obligatoria:** `bootstrap_stability()` usa la unión de pacientes de los nodos Mapper por comunidad (`community_sets`, con posible solapamiento), mientras que `cross_scenario_tables()` calcula el Jaccard del núcleo sobre **asignaciones duras sin empates** (`eligible_community_sets`). Identificar expresamente ambos denominadores/objetos; idealmente reportar Jaccard para ambas definiciones con nombres diferentes, sin mezclarlas ni cambiar retrospectivamente el estimador primario.

## 6. Tarea E — auditar la comparabilidad entre escenarios

1. Mantener los ARI **pairwise** existentes y reportar `n_patients_used` más porcentaje del total (159), sin extender artificialmente etiquetas de pacientes no cubiertos.
2. Añadir análisis descriptivo secundario con la **misma intersección válida de pacientes en todos los escenarios** cuando el tamaño sea suficiente. Si no lo es, informar `not_estimable`; no atribuir significado a un ARI calculado con n ínfimo.
3. Explicar en salida/README que ARI = 0 es concordancia aproximadamente esperada por azar, valores negativos son posibles, y un valor alto en una pareja no prueba robustez global.
4. Confirmar que S1→S2 modifica redundancia y S2→S3 modifica ponderación, aunque features estén casi o totalmente iguales. Conservar auditoría `02_graph_scenario_feature_sets.csv` y `02_graph_S4_balance_weights.csv`.
5. No usar `baseline_pop` para optimizar características, Mapper o el criterio de selección de escenarios: continúa siendo una etiqueta externa de interpretación.

## 7. Tarea F — resolver avisos secundarios sin introducir cambios científicos

**Edad al diagnóstico**: mantener la regla de que `age_dx` no se sustituye por edad en visita o baseline. Verificar si el insumo `11_integrated_baseline_patient_level.csv` posee una columna **real** con edad al diagnóstico bajo otro nombre autorizado; si no existe, continuar excluyéndola y registrar la carencia. No inferir ni inventar fechas/edades.

**`numexpr`**: emitir recomendación de usar `numexpr >= 2.8.4` en el entorno Python 3.10 que ejecuta pandas, verificando compatibilidad de dependencias; no modificar silenciosamente el entorno NIH o paquetes de usuario sin aprobación.

**`FutureWarning` de pandas**: antes de `pd.concat(redundancy_rows, ignore_index=True)` confirmar schema/dtypes, ignorar DataFrames realmente vacíos de forma segura y, si todos están vacíos, construir DataFrame vacío con esquema explícito. Conservar `scenario` y columnas esperadas y no perder filas de auditoría.

## 8. Pruebas de regresión obligatorias

Agregar a `tests/test_graph_representation_sensitivity.py` (o archivo de test complementario) como mínimo:

- **Mapa completo:** cada LAB potencialmente elegible tiene mapeo; `angiotensin_converting_enzyme` y `hepatitis_b_surface_antibody` se consideran correctamente una vez validada su clasificación; duplicados tras normalización generan error.
- **Fail fast:** laboratorio elegible sin mapa falla antes de arrancar bootstrap y muestra el nombre exacto.
- **Grupo constante:** un bloque `INFECTIOUS_SEROLOGY` con todas las filas iguales no provoca división por cero, no cambia dimensiones ni posición de columnas, registra `zero_variance` y peso efectivo 0.
- **Todos LAB degenerados:** réplica marcada `not_estimable` con razón explícita, sin crear embedding espurio.
- **Denominadores:** mock con 60 solicitadas, 27 exitosas, 27 matches y 33 fallidas devuelve `conditional=1.0`, `conservative=0.45`, `QC_FAIL`; **NO declara estabilidad**. Este es un test sintético, no el recuento real.
- **Bootstrap sano:** 60/60 exitosas, 50 matches → ambos 50/60≈0.833; con puerta QC aprobada puede declarar estable.
- **Sin éxitos:** 0/60 lleva `INVALID_BOOTSTRAP` y estabilidad primaria no evaluable (`NaN`), nunca cero como resultado biológico.
- **Errores inesperados:** no se silencian problemas de esquema/código.
- **Reproducibilidad:** mismas semillas y mismos insumos producen los mismos remuestreos, auditorías y métricas.
- **Invariancia S3/S4:** feature set y missingness crudos idénticos después del cambio.
- **Interpretación:** `interpretation_check()` no puede declarar robustez global por encontrar una sola comunidad estable en dos escenarios; un fallo QC en S4 se refleja como `NOT_EVALUABLE`.
- **Compatibilidad:** siguen generándose los CSV y tres PNG preexistentes; outputs nuevos son adicionales; `patient_id` y orden de asignación permanecen intactos.

**Ejecutar:**

```bash
cd /data/salazarda/data/obj1_sjd
/usr/local/Anaconda/envs/py3.10/bin/python -m pytest -q \
  tests/test_graph_representation_sensitivity.py

/usr/local/Anaconda/envs/py3.10/bin/python \
  src/studies/graph/02_graph_representation_sensitivity.py
```

Si la corrida completa tarda demasiado para integración continua, realizar tests sintéticos rápidos y una prueba de humo real con réplicas reducidas **solo para depuración**; el resultado científico formal debe conservar el presupuesto configurado.

## 9. Criterios de aceptación de la entrega

- [ ] El mapeo S4 cubre **todo** el universo elegible y no hay warnings de features sin grupo durante bootstrap.
- [ ] Los grupos constantes tienen tratamiento matemáticamente definido, auditado y sin pesos infinitos.
- [ ] Existe un registro **por réplica** para S0–S4, incluyendo casos fallidos y causas.
- [ ] `n_requested = n_success + n_failed` por escenario y las métricas usan denominadores declarados.
- [ ] La puerta QC impide describir fallos de ejecución como comunidades biológicamente inestables.
- [ ] Se conserva el refit completo del pipeline por bootstrap.
- [ ] `interpretation_check()` distingue continuidad de núcleo, reproducibilidad global y QC computacional.
- [ ] Se conservan escenarios, parámetros, cohortes y características científicas originales salvo las correcciones estrictamente justificadas.
- [ ] S3 y S4 siguen usando exactamente las mismas variables crudas, pero distinta ponderación.
- [ ] Las pruebas nuevas y existentes pasan, y el script genera salidas/figuras esperadas.
- [ ] El resumen de PR/documentación declara qué cambió, cómo se probó, cuál era el fallo original y si cambió la conclusión S4 tras reparar QC.

## 10. Qué NO hacer

- **No** aumentar estabilidad cambiando `stable_community_threshold`, `branch_match_jaccard`, `n_cubes`, `perc_overlap`, `min_samples` o las definiciones de comunidades.
- **No** escoger retrospectivamente S0, S1, S2, S3 o S4 según ARI, coincidencia con Pop o visualización más atractiva.
- **No** imputar edad de diagnóstico con edad basal.
- **No** eliminar laboratorios clínicamente importantes como solución a un error técnico de mapping.
- **No** cambiar la definición de la cohorte ni los CSV de Block A.
- **No** interpretar `Stable: 0` si la tasa de éxito del bootstrap no permite estimar estabilidad.
- **No** convertir ausencia de concordancia con Pop1–3 en fracaso del método: las etiquetas Pop son referentes externos, no la verdad de terreno topológica.

## 11. Entregables del agente

1. Diff/PR con cambios focalizados y pruebas.
2. Tabla de cobertura de grupos clínicos LAB, con decisiones justificadas para ACE y hepatitis B surface antibody.
3. Outputs de auditoría del bootstrap y estabilidad, más resultados resumidos para S0–S4.
4. Nota antes/después: réplicas solicitadas, exitosas, fallidas, causas; estado QC; estabilidad condicional y conservadora; comunidades estables/evaluables; comparaciones Jaccard/ARI con denominadores.
5. Explicación de si la evidencia apoya **un núcleo reproducible**, **múltiples ramas reproducibles**, **estructura global sensible a representación**, o si algo permanece **no evaluable**.

**Criterio científico final:** el objetivo de esta corrección es que los resultados reflejen la señal de la cohorte y las incertidumbres metodológicas, **no** que S4 obligatoriamente produzca comunidades estables ni que `INTERPRETATION CHECK` imprima `YES`.

---

**Referencias internas verificadas (rama principal al redactar la especificación):**

- `src/studies/graph/02_graph_representation_sensitivity.py`
- `src/studies/graph/config_sensitivity.yaml`
- `data/raw/graph_s4_lab_group_map.csv`
- `tests/test_graph_representation_sensitivity.py`
- Documento de diseño del estudio: objetivo de múltiples ramas estables y análisis bootstrap como parte de la validación; cualquier discrepancia entre el protocolo preespecificado y el código debe transparentarse, no resolverse con cambios silenciosos.
