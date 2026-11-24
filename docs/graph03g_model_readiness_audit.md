# Graph03g. Auditoría integral de preparación de N3

En Biowulf, ejecute el comando con los resultados canónicos de Graph02, Graph03 y Graph03e disponibles.

```bash
/usr/local/Anaconda/envs/py3.10/bin/python \
  /data/salazarda/data/obj1_sjd/src/studies/graph/03g_graph_n3_model_readiness_audit.py
```

La ejecución usa `secure-recompute` con integridad estricta y audita R0–R4 y los
cinco desenlaces del protocolo. Descubre las fuentes a través de los contratos
del repositorio. El proceso devuelve un resumen seguro y un código de salida
distinto de cero si falla una comprobación de integridad.

## Verificaciones

- Compara el release y la política temporal con las publicaciones previas
  y verifica las huellas de los insumos antes y después de la auditoría.
- Reutiliza el linaje y las membresías congeladas de cada representación.
  Distingue variables definitorias N1, variables relacionadas N2 y candidatos
  N3. R2 conserva su alcance exploratorio.
- Reconstruye la primera medición comparable por paciente desde el día basal
  hasta seis meses calendario después, inclusive. Audita en memoria las causas
  de exclusión, las definiciones de ensayo, los empates y las sensibilidades
  previstas. Concilia las definiciones y los agregados disponibles en Graph03e.
  La publicación de Graph03e no permite cotejar pares paciente–espécimen.
  La ausencia de un mapeo reconocido deja pendiente la investigación de
  variantes que quedan fuera del catálogo de EDA.
- Compara fechas de selección basal con Step01 y documenta la cobertura
  longitudinal del contrato de Graph03. La coincidencia por analito y fecha
  indica posible reutilización. La identidad exacta de la muestra requiere
  un identificador verificable en ambos contratos.
- Evalúa en memoria los modelos A (edad, sexo y era basal) y B (con comunidad
  congelada), con la máscara de casos completos de Graph03e. Calcula soporte,
  rango, variación, eventos y grados de libertad por modelo. Ejecuta cero
  estimadores y usa las familias de hipótesis predefinidas. RF cuantitativo
  tiene prioridad y RF cualitativo es su codificación alternativa. ANA y HEp-2
  pertenecen a una familia relacionada.

## Salidas

El directorio canónico es
`outputs/tables/studies/graph/03_graph_n3_model_readiness_audit/`.

| Archivo | Contenido |
|---|---|
| `03g_summary.json` | Estado, alcance solicitado y política aplicable |
| `03g_pipeline_checks.csv` | Integridad, disponibilidad y conciliación |
| `03g_RESULTADOS.md` | Matriz y explicación de bloqueos |
| `03g_readiness_matrix.csv` | Disposición por representación y desenlace |
| `03g_independence_lineage.csv` | Linaje estructural y requisitos documentales |
| `03g_measurement_funnel.csv` | Embudos, procedencia y cobertura divulgable |
| `03g_source_assay_audit.csv` | Categorías permitidas y alcance temporal y analítico |
| `03g_statistical_support.csv` | Disposición de A y B y acción de revisión |
| `03g_review_queue.csv` y `03g_REVIEW_PACKET.md` | Decisiones pendientes y referencias reproducibles |
| `03g_privacy_qc.json` | Política de publicación aplicada |

## Estados y revisión

- `AUDIT_COMPLETE` indica que se completaron las verificaciones previstas.
  `AUDIT_COMPLETE_WITH_LIMITATIONS` remite a las limitaciones documentadas,
  incluidas fuentes ausentes y conclusiones reservadas para revisión autorizada.
- `AUDIT_FAILED` indica que una dependencia indispensable o una comprobación
  de integridad bloqueó el uso de los resultados actuales. Consulte el control
  correspondiente. `PUBLISHED_SELECTION_OR_COVERAGE_DRIFT` señala una
  discrepancia con Graph03e. Se conserva el diagnóstico estructural disponible.
- `DATA_UNAVAILABLE` indica que una etapa no dispone de evidencia suficiente
  para pronunciarse. `ZERO_ELIGIBLE_MEASUREMENTS` se limita a los filtros de
  la cobertura publicada. `SUPPRESSED` deja la magnitud o el estado indeterminado.
- `N3_REVIEW_PENDING` requiere adjudicación clínica. Un registro exploratorio
  compatible se documenta como `CONTRACT_RECORD_PRESENT`. Si existe evidencia
  firmada de la ruta controlada, Graph03g verifica su alcance y huellas como
  antecedente documental de ese plan. El equipo habilitado registra las
  decisiones por el procedimiento institucional vigente.
- `AUTHORIZED_STATISTICAL_REVIEW_REQUIRED` indica que la factibilidad
  recalculada de A o B requiere acceso autorizado. El contrato vigente permite
  trasladar categorías de los agregados publicados y mantiene protegidos los
  nuevos diagnósticos, incluidos los indicadores de aprobación o fallo.
  `READY_FOR_03E` queda reservado para una disposición verificable y
  autorizada. Las fichas requieren una decisión clínica por par candidato.

Las salidas son `INTERNAL_RESTRICTED`, con directorio `0700` y archivos `0600`.
La escritura es atómica y se realiza bajo bloqueo. Los resultados describen
una ventana retrospectiva condicionada a comunidades congeladas. Las
mediciones pueden haber ocurrido después de t0.
