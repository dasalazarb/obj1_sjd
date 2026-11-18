# Graph03 N3 con recolección por día calendario

`calendar_day_prebaseline` es el contrato de la ejecución automática de 03e para asociaciones exploratorias con membresías congeladas. Su alcance temporal es el día de recolección. La precedencia intradía, la identidad del espécimen y de la plataforma y la causalidad quedan fuera del alcance. Las invocaciones técnicas explícitas conservan el contrato v1 de timestamps con offset, ensayo y muestra verificados.

## Selección y procedencia

- Cotejar `Collected Date Time` en BTRIS raw e intermedio para cada protocolo 11D y 15D con `lab_date` en el parquet EDA de Step20. Inspeccionar los encabezados originales y documentar en el expediente dónde Step20 reduce la precisión a día. El código compara los días escritos en las fuentes sin conversiones de zona. Una fecha ausente, inválida o discordante se excluye. Un día interpretable de `specimen_datetime` que contradiga el cotejo también excluye la fila.
- La unión de los tres insumos usa los identificadores y nombres originales, con correspondencia literal. `Order ID` es parte de la clave de procedencia. Si una clave se repite en registros distintos del mismo CSV o de varios archivos, añadir un discriminador original documentado cuando exista. Si el vínculo sigue ambiguo, el registro queda bloqueado. La existencia de nombres o etiquetas de ensayo solo sirve para delimitar la definición comparable del crosswalk.
- Reconciliar el paciente con el baseline oficial del release Graph02 y con el episodio de origen que consta en EDA y Step10. Las recolecciones elegibles de otro episodio del mismo paciente conservan su `source_episode_id`. Step01 y contexto Step10 auditan la selección derivada por anchor por separado.
- Fijar la ventana por outcome antes de analizar asociaciones. La regla principal usa `1 <= baseline_day - collection_day <= lookback_days` y toma el último día elegible. El mismo día, días posteriores y días fuera de ventana se excluyen. Una sensibilidad con diferencia de al menos dos días requiere su propia configuración, expediente, artefacto y plan. En el último día se colapsan duplicados comprobablemente idénticos. Discordancia en resultado, unidad, definición o identidad lógica genera `AMBIGUOUS_SAME_DAY` y excluye al paciente para ese outcome sin recurrir a un día anterior.
- El crosswalk firmado restringe el outcome a combinaciones comparables de nombre original de orden, nombre original de observación y etiqueta EDA `assay` (puede ser nula), con una unidad exacta. La definición describe la prueba clínica comparable dentro de ese alcance. `assay_id` y `sample_id` quedan nulos. Los tokens desconocidos o no realizados se tratan como faltantes. Los valores censurados, tokens ajenos a la convención y unidades fuera de alcance se excluyen. Incluir ANA general y HEp-2 juntos exige evidencia revisada de que son hipótesis distintas y definiciones de origen disjuntas. Las dos codificaciones de RF representan la misma hipótesis por representación.

## Operación habitual

```bash
/usr/local/Anaconda/envs/py3.10/bin/python /data/salazarda/data/obj1_sjd/src/studies/graph/03e_graph_n3_controlled_inference.py
```

El comando localiza Graph03 Fase A y su release Graph02 mediante el selector de compatibilidad existente y verifica sus hashes. Después valida la prespecificación, ejecuta la auditoría y construye el candidato v2. Con el expediente de fuente firmado genera el Parquet revisado, prepara el borrador de 03e y busca el plan firmado. El preflight evalúa los gates por representación, outcome y modelo. La inferencia se ejecuta exclusivamente para los modelos `READY`.

Las rutas de instalación están en [n3_auto.yaml](../src/studies/graph/configs/n3_auto.yaml). Por defecto, los directorios `obj1_sjd` y `eda_sjd` comparten el padre `/data/salazarda/data`. El primero contiene Step01, Step10 y las colecciones Graph02/03. El segundo contiene Step20 y BTRIS raw e intermedio, cada uno con 11D y 15D. En otra instalación se puede editar una vez `n3_auto.yaml`. Las rutas de los releases se leen de `pipeline.yaml` y la selección usa la relación de procedencia entre los releases.

El directorio operativo predeterminado es `/data/salazarda/data/graph03_n3_private`. Su propietario debe ser la cuenta que ejecuta 03e. El directorio y los runs usan modo `0700` y sus archivos modo `0600`. El comando crea el directorio si su padre existe y permite la escritura. Un lock de sistema protege cada ejecución y se libera cuando termina el proceso. Cada etapa reutiliza resultados con los mismos hashes si su inventario de artefactos está íntegro. Los cambios de código, configuración, insumos, archivos congelados, evidencia, revisiones o firmas invalidan las etapas correspondientes. Los informes llevan su propio inventario y las ejecuciones previas quedan disponibles en el espacio privado.

## Depósitos institucionales

La siguiente tabla usa rutas relativas a `/data/salazarda/data/graph03_n3_private`. El equipo custodio deposita cada paquete una vez por alcance aprobado. Si cambian los hashes vinculados, el informe identifica la revisión que necesita actualización. Se deben copiar los borradores a las ubicaciones de depósito, completarlos y firmar sus bytes definitivos. El comando conserva sus borradores separados de los depósitos.

| Etapa | Ubicación exacta | Qué deposita el equipo custodio |
| --- | --- | --- |
| Claves de confianza | `trust/prespecification_public.pem`, `trust/source_public.pem`, `trust/pair_public.pem` | Claves públicas institucionales, instaladas por el administrador mediante un canal autorizado. El directorio y las claves deben impedir escritura de grupo y de terceros. |
| Prespecificación | `institution/prespecification.yaml`, `institution/prespecification.sig`, `institution/prespecification_evidence/` | Contrato v2, ventanas principales por outcome, campos de enlace, formatos de fecha y anexos con SHA-256. Usar `APPROVED_CALENDAR_DAY_PRESPECIFICATION` en el expediente revisado y firmarlo con la clave correspondiente. |
| Revisión de fuente | `institution/source/review.yaml`, `institution/source/review.sig`, anexos relativos a `institution/source/` | Expediente `graph03_n3_measurement_review_v2` con `APPROVED_SOURCE_PROVENANCE`, revisor, ID de revisión, hashes `review_binding` obtenidos de la auditoría y definiciones comparables. |
| Plan y decisiones | `institution/inference/active.yaml`, `institution/inference/active.sig`, `institution/inference/decisions/` | Plan con hashes del borrador, representaciones/outcomes, modelos y revisores autorizados. Adjuntar decisiones por par y sus evidencias. El plan firmado fija los hashes de las decisiones. Cada decisión de medición Calendar Day lleva también su propia firma separada. |

Las firmas separadas usan el formato binario de `openssl dgst -sha256`. Los nombres de anexos en `evidence` son relativos al expediente y cada anexo tiene `path` y `sha256`. El plan apunta a cada decisión con `path` y `sha256` relativos a `institution/inference/decisions/`. La firma de la decisión de medición se indica en `signature_path`, en ese mismo directorio. Los borradores se encuentran bajo `artifacts` en el informe privado.

Cada decisión 03e fija `plan_sha256` del plan con sus outcomes y modelos definitivos. El equipo puede calcularlo con `plan_sha256()` de `s03_n3_decisions.py`. Los borradores generados tras el preflight ya incluyen ese hash. La lista de decisiones y la de revisores quedan cubiertas por la firma de `active.yaml`.

### Prespecificación de la ventana y del enlace

El borrador `unsigned_prespecification.yaml` usa `schema_version: graph03_n3_measurement_source_v2` y `calendar_day.contract: calendar_day_prebaseline`. En `calendar_day.windows`, registrar por outcome los enteros `minimum_days_before_baseline: 1` y `lookback_days` preespecificado. En `calendar_day.linkage_fields`, registrar la columna de EDA y la original de cada nivel BTRIS para el paciente, orden, nombre de orden y nombre de observación. Agregar `record_id` solo si el diccionario acredita un discriminador original presente en los tres insumos.

La tabla `COLUMN_ALIASES` de Step20 acepta `Observation Name` como fuente de `cluster_name_original` y prioriza `Cluster Name` o `cluster_name` si están presentes. El custodio confirma el encabezado efectivo de los 11D/15D y registra la evidencia institucional en `prespecification_evidence/`. Los archivos con encabezados que contradicen este enlace bloquean el flujo. `calendar_day.date_formats` exige un formato para raw y otro para intermedio. Se puede usar `ISO8601` o un formato explícito con año de cuatro dígitos. 03f comprueba los formatos en los CSV y excluye fechas ausentes, inválidas o discordantes. [Diccionario de columnas y normalización de Step20](https://github.com/dasalazarb/eda_sjd/blob/ffe59ecc5711b9e2e0bf4e611e32e8feffc5e2b4/src/20_btris_visit_date_match_report.py#L527)

### Expediente de fuente y decisiones por par

Para el expediente de fuente, `CALENDAR_AFFIRMATIONS` enumera las afirmaciones requeridas por outcome. El crosswalk `comparable_original_tests` lista tuplas exactas de `original_order_name`, `original_observation_name` y `assay`. El expediente fija `date_field: lab_date`, `unit_field: unit`, el campo de resultado original y un `measurement_review` con el tipo, definición, unidad, ventana, reglas de selección y convención de categorías. Debe documentar consentimiento y uso institucional, truncamiento temporal, comparabilidad y riesgo residual. Ver la estructura sintética de [las pruebas de Calendar Day](../tests/test_graph03_n3_calendar_day.py).

La decisión `measurement` de 03e reproduce en `details` el `measurement_review` firmado y agrega `source_review_sha256`, el hash del expediente de 03f. El preflight conserva los gates de protocolo, baseline, medición, temporalidad, independencia de N1/N2, Pop cuando aplica, topología, diseño y soporte. `demo__protocol` requiere acreditación semántica y fecha efectiva. ESSDAI, ESSPRI y biopsia requieren sus propias fechas instrumentales y revisión de fuente. Las comparaciones A/B/C/D y el ajuste por multiplicidad conservan las reglas de 03e.

## Lectura del resultado

El comando imprime una sola línea JSON. `n_ready` cuenta modelos que superaron los gates y `estimators_run` cuenta los intentos de ajuste de la ejecución, incluidos los recuperados de una corrida íntegra. `COMPLETE_WITH_LIMITATIONS` identifica, por ejemplo, un modelo autorizado junto con otro bloqueado o una comparación no estimable. El informe de `report_path` contiene el historial de estados, la procedencia y los bloqueos por modelo y gate. Los Parquet y ledgers quedan restringidos. Las tablas de 03f suprimen celdas pequeñas y 03e retiene los recuentos de pacientes y grados de libertad.

### Ejemplo sintético de revisión pendiente (`PENDING_REVIEW`)

```json
{
  "mode": "calendar_day_v2",
  "status": "SOURCE_REVIEW_PENDING",
  "graph02_run": "SYNTHETIC_GRAPH02",
  "graph03_run": "SYNTHETIC_GRAPH03",
  "source_status": "SOURCE_CANDIDATE_UNAPPROVED",
  "source_run": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_review_status": "SOURCE_REVIEW_PENDING",
  "plan_status": "NOT_STARTED",
  "preflight_status": "NOT_STARTED",
  "n_ready": 0,
  "estimators_run": 0,
  "blockers": ["SOURCE_REVIEW_SIGNATURE_MISSING"],
  "next_required_action": "Deposite el expediente de fuente y su firma en /data/salazarda/data/graph03_n3_private/institution/source.",
  "report_path": "/data/salazarda/data/graph03_n3_private/runs/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/report.json"
}
```

### Ejemplo sintético con un modelo A aprobado y completado

```json
{
  "mode": "calendar_day_v2",
  "status": "COMPLETE",
  "graph02_run": "SYNTHETIC_GRAPH02",
  "graph03_run": "SYNTHETIC_GRAPH03",
  "source_status": "SOURCE_REVIEWED_PENDING_03E_DECISIONS",
  "source_run": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "source_review_status": "SOURCE_REVIEWED_PENDING_03E_DECISIONS",
  "plan_status": "PLAN_REVIEWED",
  "preflight_status": "READY",
  "n_ready": 1,
  "estimators_run": 1,
  "blockers": [],
  "next_required_action": "Consulte la salida privada de inferencia.",
  "report_path": "/data/salazarda/data/graph03_n3_private/runs/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/report.json"
}
```

La validación de la implementación usa datos sintéticos. La instalación institucional debe aportar los CSV raw/intermedios, Step20, los releases congelados y la evidencia local de columnas y fechas.
