# Pharma 05 · Clinical and Translational Evidence

Ejecuta ambos comandos desde la raíz de `obj1_sjd` en el entorno de análisis de Biowulf.

```bash
python src/studies/pharma/05_pharma_deck_evidence.py
bash collect_pharma_deck.sh
```

La ejecución usa `config_stage05.yaml` y `05_pharma_mapping.json`. Hereda semilla, mínimo de eventos y definición de conversión sostenida de `config.yaml`. Requiere las dependencias existentes de Pharma y de la publicación semántica. El recolector requiere Bash, Python 3.9 o posterior y un sistema POSIX.

## Entradas canónicas

- Master efectivo mediante `common.resolve_effective_longitudinal_input(common.EFFECTIVE_LONGITUDINAL_PARQUET)`. La selección del master es la publicada por `common.py`. Stage 05 no tiene su propio selector.
- Registro `10_variable_registry.csv` junto al Step 10 de origen, según `common.step10_sidecar_base`. El contexto adyacente aporta fechas y unidades cuando se encuentra disponible.
- `01_pharma_transition_intervals.parquet` en `data/analytic/studies/pharma/01_run_pharma_main/`.
- Perfil y resumen de Pharma 00, disponibilidad clínica y de laboratorio de Pharma 01, resumen de Pharma 02 y candidatos, soporte, modelos y resumen de Pharma 03 en sus carpetas fijas debajo de `outputs/tables/studies/pharma/`.
- Step 11 baseline y Step 09 PRO aportan reconciliación y FDR suplementario cuando están disponibles. Los resúmenes de Pharma 03B y 04 son opcionales. Se registra su presencia sin reestimar sus modelos.

Si los productores solo publicaron en subdirectorios semánticos, el responsable de upstream debe publicar los archivos canónicos coherentes antes de ejecutar Pharma 05. Un archivo con el nombre esperado puede seguir siendo incompatible. Preflight exige que el hash de Pharma 00 identifique el master efectivo, verifica los hashes disponibles, comprueba las claves de pacientes y episodios, y recalcula disponibilidad, soporte y escalado de Pharma 01–03. El refit de riesgo coteja los coeficientes publicados con la misma muestra y diseño. Las fuentes legadas sin manifiesto quedan identificadas en QC.

Step 10 ordena episodios por fecha y número de visita canónico. Se auditan los episodios de igual fecha y las adyacencias que tocan visitas no clasificables. Se conservan únicamente intervalos canónicos de duración positiva cuyas dos visitas son clasificables. La cohorte basal usa exclusivamente la bandera clínica. Las diferencias con el primer Pop clasificable y con Step 11 se cuantifican por claves internas cuando hay referencia. La discrepancia histórica 148/159 requiere verificar el universo y las claves de ambos conteos en Biowulf.

## Mapeos y alcance

| Conceptos | Fuente y condición |
| --- | --- |
| ESSDAI, ESSPRI, edad, sexo, protocolo, componentes ESSPRI | Conceptos y alias oficiales de `_shared.py`. Solo ESSDAI y ESSPRI son obligatorios en el master. |
| Doce dominios ESSDAI | Ordinales `essdai__*\_ordinal_score` / `eg_*\_ordinal_score` según Step 13 y pesos canónicos de Step 14. La actividad complementaria corresponde a ordinal >0. |
| ESR, MCV, proteína total, IgG, C3, C4 y candidatos revisados | Tallo exacto de Pharma 00 respaldado por Step 10, representación numérica, unidad única verificable y fecha de toma hasta el ancla. Los candidatos adicionales requieren selección de Pharma 03 y representación longitudinal aprobada por el productor. |
| SSA, SSB, RF, ANA, crioglobulinemia | Campos de historia hasta el ancla del Step 10. Una observación desconocida no equivale a ausencia. SSA “ever” abarca la historia documentada en el conjunto efectivo y puede incorporar seguimiento. |
| Duración, histología y comorbilidad | Duración usa solo los alias del productor Pharma 01. El focus score fechado as-of y el ever retrospectivo requieren una fuente de histología verificable. Comorbilidades requieren una selección explícita con temporalidad confirmada. |
| Tratamiento | Step 08/10 publica nombres documentados, sin confirmación de cobertura de exposición en la fecha del ancla. Se reportan menciones CTDB y órdenes anteriores/posteriores por separado. Los recuentos de exposición y el ajuste MCV quedan no estimables cuando falta el estado activo/pasado/no expuesto verificado. |
| Muerte y cutoff administrativo | Desactivados en la configuración recibida. Se necesita una fuente confirmada con fecha para activar la incidencia acumulada con Aalen–Johansen. |
| Inventario CTDB | Opcional y restringido. Ruta `data/restricted/pharma/ctdb_specimen_inventory.parquet`, reservada para un inventario autorizado. Requiere clave de episodio y de espécimen, fecha de toma, tipo, preservación, fecha de inventario y disponibilidad observada en esa fecha. `aliquot_id` habilita el conteo de alícuotas. El QC registra ausencia sin generar ceros. El denominador HV requiere un universo autorizado aparte. |

Se documentan mapeos presentes, ausentes, unidades y exclusiones temporales en el preflight y el inventario agregado. La verificación del código revisa los productores y contratos. La verificación de qué columnas opcionales están presentes y llenas en la cohorte requiere los insumos de Biowulf.

## Estimandos y esquemas de tablas

Los CSV usan el prefijo `05_pharma_`. El manifiesto declara las columnas de cada artefacto y el SHA-256. `n_patients`, `n_intervals` y `n_evaluable` corresponden a sus unidades indicadas. Las celdas sin estimación tienen `status` y `reason`. Las proporciones se expresan de 0 a 1 y los deltas de riesgo de E en puntos porcentuales.

| CSV (sufijo) | Filas y medidas principales |
| --- | --- |
| `ever_pop1_by_baseline_pop` | Pop al baseline, pacientes basales, seguimiento ESSDAI, eventos basales y posteriores, fracción ever observada e IC binomial exacto. |
| `pop1_cumulative_risk`, `pop1_risk_curve_source` | Pop, índice, sensibilidad, estimador y tiempo. KM usa fecha de detección, último ESSDAI conocido y la tabla de riesgo anterior a retiros empatados. IC log-log Greenwood queda vacío con riesgo 0 o 1. Horizontes sin pacientes en riesgo se marcan no estimables. Se incluyen reverse KM, midpoint, primer Pop clasificable y ESSDAI basal bajo con ESSPRI faltante sin reasignar Pop. CIF puntual si se habilita muerte verificada. |
| `patient_profiles_by_transition`, `serology_profiles_by_transition`, `pop2_converter_profiles` | Las nueve rutas, marcador, nivel y esquema de intervalos. Mediana, cuartiles, media/SD o prevalencia con denominador conocido. `pop2_converter_profiles` es una vista de intervalos de origen Pop2. |
| `domain_drivers_by_transition`, `domain_cooccurrence` | Ruta, dominio o par de dominios. Pares observados, activación/resolución, distribución de cambios ordinales y empeoramiento simultáneo con denominador pareado. |
| `pro_changes_by_transition` | Cambio origen–destino de total ESSPRI y componentes. La magnitud máxima absoluta define el componente dominante. Empate y todos los cambios cero se reportan por separado. |
| `essdai_decomposition`, `leave_domain_out_sensitivity` | Cambios del panel ordinal ponderado. La descomposición requiere panel completo y concordancia con el total en ambos extremos. Se auditan residuos e incompatibilidad. La exclusión biológica/hematológica usa umbral modificado ≥5 y delta continuo en la misma muestra completa. |
| `biomarker_correlations_by_pop`, `pop2_biomarker_correlations`, `longitudinal_biomarker_correlations` | Pop, par, esquema, N, Spearman, IC por remuestreo de pacientes. Índice fijo antes de aplicar disponibilidad pareada. BH por separado para pares de biomarcadores y cruces clínicos, Pop2 y las otras Pop, índice y medias. Dentro del paciente se usan pares centrados y bootstrap agrupado, sin p nominal que suponga visitas independientes. |
| `absolute_risk_by_biomarker`, `absolute_risk_support` | Marcador, Pop de origen, horizonte, método y soporte de rutas; probabilidades en P25/P75 con edad mediana fija y diferencia en puntos porcentuales. La primera llegada tiene Pop1 absorbente y requiere cuatro rutas, la ocupación requiere seis. Se exige soporte de seguimiento en el horizonte. Bootstrap de todos los intervalos del paciente, reajuste completo y cuantiles de referencia fijos. |
| `medication_by_transition`, `treatment_confounding` | Medicamento y ruta o contraste MCV; documentación, estado y proporción entre conocidos. El modelo, si hay exposiciones verificables y soporte, compara Poisson con offset y SE agrupado en la misma muestra completa. La confusión por indicación puede persistir. |
| `trial_eligibility_proxy` | Pop basal, tiempo de información SSA, positivos, negativos, indeterminados y sin medición. Es un proxy observacional. |
| `patient_trajectory_summary`, `sustained_conversion` | Pop basal y primer Pop2 con o sin Pop1 previo. La clasificación reconoce primero conversiones recurrentes, sostenidas y otras conversiones observadas. Si no hay evento, exige el mínimo de visitas y tiempo antes de llamar persistencia o cambio observado. Pop1 basal tiene categorías propias de persistencia, salida y retorno. Visitas desconocidas rompen series de confirmación y la confirmación exige fechas distintas. Se publican tiempos agregados a primera entrada y confirmación. |
| `patient_converter_profiles`, `pretransition_biomarker_trends` | Perfiles al landmark Pop2 por horizonte común y comparación descriptiva preevento con visita a tiempo comparable. Los controles requieren observación al horizonte. Las pendientes requieren ≥3 fechas anteriores al evento o pseudoíndice, dentro de la ventana fija. El pseudoíndice usa el tiempo mediano de evento sin seleccionar por valores del marcador. |
| `biomarker_coverage`, `baseline_reconciliation`, `transition_evidence_summary` | Cobertura y denominadores por rutas, Pop, visitas y protocolo observado; intersecciones de cohortes y síntesis de evidencia por las nueve rutas. La cobertura diferencial se interpreta descriptivamente. |
| `specimen_availability`, `pro_slopes_fdr` | Muestras y alícuotas por cohorte, tipo, preservación y relación temporal con índice. Step 09 conserva p y q originales y coteja la familia completa de coeficientes de tiempo crudos y ajustados, incluyendo no estimables. |

Pop1 corresponde a actividad sistémica *moderate-or-higher*, ESSDAI ≥5. Las proporciones ever, probabilidades, intensidades y resultados de regresión tienen denominadores e interpretaciones distintos. Las proyecciones de Pharma 03 son escenarios exploratorios de intensidades homogéneas a covariables fijas. La fecha detectada en una visita no identifica el inicio biológico.

## Publicación y ZIP

`outputs/{tables,figures,qc,logs}/studies/pharma/05_pharma_deck_evidence/` contiene el conjunto fijo. `05_pharma_evidence_summary.json` resume módulos y limitaciones. `05_pharma_preflight.json` registra entradas, procedencia, configuraciones y controles. `05_pharma_risk_bootstrap_qc.json` conserva los fallos por estimando. `05_pharma_output_manifest.json` registra los archivos y sus hashes. `05_pharma_figure_manifest.csv` enlaza figuras en inglés a sus tablas. La preparación temporal precede a la sustitución y se revierte una publicación fallida. Los análisis pasan un conjunto de tablas agregado al publicador.

El ZIP reemplaza `pharma_deck_outputs.zip` en la raíz después de comprobar su integridad. Usa una lista cerrada de rutas canónicas. `MANIFEST.csv` registra estado, hash, cobertura por diapositiva y menor conteo positivo por tabla para revisión. `PENDING.md` explica faltantes, análisis no estimables, conflictos y discrepancias de integridad. La distribución basal agregada de Pharma 05 sustituye al visual histórico. El ZIP es interno y las tablas y figuras requieren revisión de divulgación institucional antes de compartirlas.

## Verificación

```bash
python -m compileall src/studies/pharma/stage05 src/studies/pharma/05_pharma_deck_evidence.py
python -m pytest -q tests/test_pharma_stage05_*.py
python -m pytest -q tests/test_pharma_lab_profile*.py tests/test_pharma_main_plots.py \
  tests/test_pharma_multistate_biomarkers.py tests/test_10*.py \
  tests/test_integrated_longitudinal_dataset.py tests/test_pop_transitions*.py \
  tests/test_risk_factors_progression.py tests/test_08_medication_episode_contract.py
bash -n collect_pharma_deck.sh
```

Las pruebas sintéticas no verifican resultados clínicos reales. El checkout de desarrollo no trae el master efectivo ni los outputs canónicos de Pharma 00–03. El preflight termina con diagnóstico en ese checkout.

En la batería heredada de referencia, `tests/test_pharma.py` intenta importar un entrypoint retirado. La fixture de `test_domain_ordinal_delta_preserves_score_levels` en `tests/test_pharma_main_plots.py` omite `interval_days`, que Pharma 01 ya exige. Se registran como incompatibilidades de las pruebas previas.
