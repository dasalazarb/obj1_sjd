# Graph03g: auditoría de independencia y estimabilidad N3

Graph03g explica las causas de no estimación de la ruta exploratoria que se
ejecuta al invocar Graph03e sin argumentos. Conserva las comunidades congeladas
y cruza tres fuentes de evidencia por representación y desenlace. La primera es
el linaje de las variables incluidas y el contrato de revisión humana. La
segunda es la selección de mediciones originales de colección. La tercera es el
soporte de los diseños A y B en la misma población que usa Graph03e.

## Ejecución

Desde la raíz del repositorio con resultados compatibles de Graph02, Graph03 y
Graph03e:

```bash
python src/studies/graph/03g_graph_n3_model_readiness_audit.py \
  --mode summaries --representation R0 --strict
```

En el entorno institucional con acceso autorizado a las fuentes originales:

```bash
python src/studies/graph/03g_graph_n3_model_readiness_audit.py \
  --mode secure-recompute --representation R0 --strict
```

La ubicación predeterminada sigue la configuración de Graph03e. Para una
instantánea archivada compatible con el contrato de agregados de Graph03e,
indique explícitamente `--graph02-release` (el manifiesto), `--graph03-run` y
`--graph03e-dir`. Se comprueban las huellas publicadas de Graph02 y la política
temporal incluso en modo normal. `--strict` exige además el vínculo documentado
a Graph03 en el resumen de Graph03e, verifica los pins disponibles y rechaza
discrepancias entre la cobertura publicada y el recálculo. `--representation all`
incluye las sensibilidades R1–R4. R2 conserva su alcance exploratorio.

## Interpretación

- `graph03e_reason` conserva el motivo efectivo de la ejecución publicada.
  `structural_independence_classification` evalúa la representación congelada
  y `approval_gate_status` comprueba el contrato exploratorio vigente. Una
  clasificación estructural favorable con revisión pendiente sigue bloqueada.
- Un cero de cobertura se limita a la definición primaria, procedencia original
  verificada, ventana y membresía del panel. `SUPPRESSED` deja el conteo
  indeterminado. `NOT_ASSESSABLE_FROM_PUBLISHED_AGGREGATES` señala qué evidencia
  queda pendiente en el modo de agregados.
- El recálculo usa las conciliaciones de paciente y episodio de Graph03e, el día
  calendario original, la ventana cerrada `[t0, t0 + 6 meses calendario]`, las
  definiciones comparables y la selección original de la primera medición.
  El embudo distingue fallos de fuente, mapeo, vínculo, ventana, validez y
  comparabilidad cuando la supresión permite publicar la causa.
- A incluye edad, sexo y era basal. B añade comunidad congelada. Los dos usan
  el filtro compartido de casos completos y comunidades de Graph03e. El soporte
  de rango, variación, balance, eventos y grados de libertad se informa como
  estados o motivos permitidos por la política. La codificación cuantitativa
  de RF tiene prioridad sobre la cualitativa. La prioridad definitiva depende
  de la estimabilidad en Graph03e.
- `READY_FOR_ESTIMATION` indica que pasaron los controles de preparación
  evaluados en Graph03g. Graph03e conserva el ajuste y sus diagnósticos, que
  pueden bloquear un desenlace después del preflight.
  `STATISTICALLY_PLAUSIBLE_PENDING_REVIEW` requiere soporte comprobado con
  datos protegidos y sigue sujeto a adjudicación.

La revisión acreditada se incorpora mediante los registros de
`characterization.independence_reviews` y
`n3_validation.prespecified_outcomes`, con la misma representación, fuente y
versión de política. Graph03g indica los campos ausentes sin publicar texto
libre de los registros. La ruta controlada de Graph03e sigue sus propios
contratos, firmas y decisiones institucionales.
[Flujo de inferencia controlada](graph03_controlled_inference.md)

## Archivos y privacidad

Las salidas se escriben en
`outputs/tables/studies/graph/03_graph_n3_model_readiness_audit/` dentro del
entorno autorizado. `--output-dir` permite indicar otro directorio privado y
separado de los insumos. La escritura se prepara en un directorio privado y
publica el resumen de finalización al final.

| Archivo | Contenido |
|---|---|
| `03g_summary.json` | Política, modo, alcance, huellas de entrada y bloqueos. |
| `03g_readiness_matrix.csv` | Una fila por representación y desenlace, con bloqueos y decisión. |
| `03g_independence_lineage.csv` | Linaje estructural y requisitos concretos de revisión. |
| `03g_measurement_funnel.csv` | Indicadores de fuente, transiciones, cobertura y códigos originales divulgables. |
| `03g_statistical_support.csv` | Una fila por diseño A o B con motivos y estados de soporte. |
| `03g_privacy_qc.json` | Política de divulgación y verificación de las salidas. |
| `03g_RESULTADOS.md` | Interpretación por representación y desenlace. |

La clasificación es `INTERNAL_RESTRICTED` y `public_export=false`. El
diagnóstico publica evidencia categórica y omite conteos nuevos, valores y
subgrupos que podrían reconstruirse al cruzar tablas. Se aplican los controles
de celda mínima y supresión complementaria al publicar las causas. Los
archivos quedan con permisos `0600` y el directorio con `0700`. Mantenga los
resultados institucionales fuera de Git y de destinos compartidos.
