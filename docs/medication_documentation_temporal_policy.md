# Política temporal de documentación farmacológica — Step 08 → Step 10 → estudios

**Fecha:** 2026-10-23. **Estado:** decisión científica para futuras implementaciones; no implica tratamiento administrado ni modelos de efecto causal.

## Regla central

Conservar dos representaciones independientes:
1. **Documentación durante el intervalo clínico:** Step 10 meddoc, relaciones principales many-to-many, descriptiva y CONTEXT únicamente. La orden puede ser posterior al anchor. No es predictor histórico.
2. **Documentación previa a fecha índice:** nuevo producto temporal calculado directamente del catálogo original de registros BTRIS/CTDB, con paciente vinculado y fecha de documentación estrictamente ANTERIOR al origen temporal de la pregunta. No es exposición activa.

Las órdenes BTRIS (11D/15D) indican orden/documentación, no despacho/administración. CTDB es medicamento reportado como actual SOLO en la visita original, no continuidad longitudinal. Cero documentos no significa paciente sin medicación. Los bloqueos de MRN y fechas de QC permanecen.

## Contrato as-of

- Para un momento de decisión t0, admitir un registro únicamente si patient linkage/QC es válido, fecha exacta válida y t_doc < t0. Con solo fecha calendario, excluir el mismo día del análisis histórico principal; contabilizarlo en QC.
- Para transiciones Pop, usar t0 = from_clinical_anchor_date, pero verificar que el anchor represente el tiempo real de medición de las variables clínicas. Una evaluación construida con datos posteriores al anchor requiere revisión de procedencia antes de declarar predictores pre-outcome.
- NO obtener datos históricos de la suma de documentos enlazados al episodio: el intervalo puede incluir eventos posteriores a t0. Construirlos desde el catálogo de una fila por documento y fechas de fuente, independientemente de cuántos episodios contengan la orden.
- Para un mismo paciente, cada registro documental se cuenta una sola vez dentro de una ventana histórica t0. Los 50 órdenes asociados con dos episodios se mantienen en los datos descriptivos de episodios, sin crear dos órdenes independientes.
- Nunca inferir de órdenes previas uso actual, nuevo usuario, discontinuación, duración, dosis consumida, administración o adherencia.
- Mantener CTDB prior reported-current at earlier visit separado de BTRIS prior order, y separar categorías farmacológicas candidatas de indicaciones verificadas.
- Describir años de cobertura de BTRIS/CTDB, presencia de documentación, disponibilidad de días/meses previos y censura izquierda. Registrar 15 casos QC de identidad, 6 alejados, ausencia de fechas y observación fuera de captura. Nunca equiparar ausencia de datos con grupo no tratado.
- Sensibilidad opcional: un margen de 7 o 30 días para separar una orden de t0 si existe riesgo de causalidad inversa; ventanas retrospectivas de 90 y 365 días por fuente con suficiencia de historial; nunca incluir relaciones de tolerancia futura ±1 mes en principal.

## Decisión por estudio

**Pharma:** código actual analiza biomarcadores/Pop/ESSDAI/ESSPRI y transiciones, NO efecto farmacológico. Conservar modelos principales. Añadir perfil descriptivo de órdenes/reportes y, solo secundariamente con DAG y soporte, variables de documentación pre-origen en modelos de transición. No usar medicación documentada entre origen y destino para explicar un cambio ya ocurrido. SE clustered-by-patient no resuelve causalidad inversa ni confusión por indicación.

**Graph/Mapper:** excluir documentación farmacológica del embedding basal principal. Caracterizar comunidades a posteriori por fuente y fecha de documentación, sin reclasificar fenotipos como exposición. Sensibilidad separada y explícita con historia pre-index puede investigar influencia de práctica clínica en las comunidades, pero nunca reemplazar el embedding clínico principal.

**Economic (planeado):** no existe aún un directorio de implementación economic en la rama main consultada. No estimar costos, consumo, utilización ni gasto a partir de órdenes; exigir datos de dispensación/administración/facturación y perspectiva económica antes de esos análisis.

## Implementación propuesta

1. Preservar output de Step 10: 497 episodios, 159 pacientes, master analítico intacto y meddoc en contexto científico.
2. Crear un adaptador temporal versionado desde el catálogo consolidado Step 08, con fecha índice explícita y productos agregados pre-t0 por fuente. No modificar Step 00 ni el enlace many-to-many validado.
3. Separar columnas y registro: meddoc (dentro del episodio) versus meddoc_prior (documentación anterior) versus QC (exclusiones). Todo nuevo predictor requiere opt-in del estudio, reglas de temporización y revisión científica.
4. Pruebas obligatorias: no post-index, no mismo día, cero joins de pacientes distintos, conservación exacta del episodio, deduplicación por ID original en ventana, identificadores QC excluidos, fuentes separadas y semántica cero=no documentación.
5. Solo si se investiga eficacia: diseño independiente target-trial con validación de tratamiento, estrategia comparadora, time zero, confusión por indicación y positividad. Marginal structural models/g-formula cuando exista exposición longitudinal validada y confusores dependientes del tiempo afectados por tratamiento.

## Fuentes

- Hernán et al., 2016, J Clin Epidemiol: https://pmc.ncbi.nlm.nih.gov/articles/PMC5124536/
- BMJ, 2022, Target trial emulation: https://www.bmj.com/content/378/bmj-2022-071108
- Measurement error/misclassification in EMR: https://pmc.ncbi.nlm.nih.gov/articles/PMC9141310/
- OHDSI OMOP drug source-type: https://ohdsi.github.io/CommonDataModel/cdm54.html
- RECORD-PE: https://pmc.ncbi.nlm.nih.gov/articles/PMC6234471/
- Active comparator/new user: https://pmc.ncbi.nlm.nih.gov/articles/PMC4778958/
- Time-varying confounding/g-methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC5710813/
- Health-care ML label leakage: https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/
- EULAR Sjögren systemic and topical therapy: https://pubmed.ncbi.nlm.nih.gov/31672775/
