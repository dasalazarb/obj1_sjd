# Caracterización basal de Graph03 entre día 0 y seis meses

La fecha índice es `clinical_baseline_date` de Graph02/Step10. Graph02 R0 conserva la cohorte, membresías, coordenadas, features y valores de su release. Su uso retrospectivo de laboratorios cercanos y posteriores al índice fue intencional. El programa de Graph03 lee esos artefactos y compara sus huellas SHA-256 antes y después de analizar las mediciones.

| Estimando | Datos y límites | Interpretación |
|---|---|---|
| `FROZEN_GRAPH02_BASELINE_SNAPSHOT` | Variables observadas del release de Graph02, incluida la selección histórica de laboratorios | N1 describe la estructura interna. N2 caracteriza constructos relacionados. |
| `GRAPH03_BASELINE_0_TO_6_CALENDAR_MONTHS` | Nuevas mediciones verificadas desde el índice, inclusive, hasta seis meses calendario después, inclusive | Caracterización retrospectiva condicionada a las comunidades congeladas. Las mediciones pueden no estar disponibles el día 0. |
| `LONGITUDINAL_POST_INDEX` | Visitas verificadas estrictamente posteriores al índice, según la política de cada endpoint | Trayectorias repetidas con tiempo medido desde la fecha índice original. |
| `LONGITUDINAL_POST_CHARACTERIZATION_WINDOW` | Visitas verificadas después del último día calendario de caracterización | Sensibilidad para reducir el solapamiento. Sigue usando el índice original como origen del tiempo. |

La suma de meses conserva el día del mes si existe. Si el mes de destino es más corto, usa su último día. El 31 de agosto de 2024 termina el 28 de febrero de 2025. El 31 de agosto de 2023 termina el 29 de febrero de 2024. Se conserva el día escrito por la fuente, incluso cuando un offset llevaría el instante al día siguiente en UTC. La fecha sin hora no establece orden intradía.

`PRE`, `DAY0`, `DAY1_30`, `DAY31_90`, `DAY91_TO_6M`, `AFTER_6M` y `UNKNOWN` forman una partición por observación. La unión de los cuatro intervalos centrales constituye el análisis basal primario. Un mismo paciente puede aparecer en varios intervalos del inventario de observaciones. La tabla de demora usa solo la observación primaria y agrega la categoría sin medición para reconciliar el denominador por comunidad.

03e conserva sensibilidades con la última medición válida, solo día 0, días 0–30, episodios exactos y procedencia EDA de menor certeza. Las primeras tres se publican como perfiles descriptivos. El enlace exacto y la procedencia EDA producen diagnósticos de estimabilidad aparte, con su propia corrección BH. Los perfiles prebasales y posteriores al límite quedan fuera de la inferencia basal.

La disponibilidad y la demora clínica reflejan indicación, frecuencia de pruebas y selección de pacientes con datos completos. Las mediciones compartidas con longitudinal aportan evidencia dependiente. La auditoría de solapamiento compara paciente, outcome y día de colección verificado. Sus coincidencias indican posible reutilización y requieren revisar la identidad exacta de la muestra. Los modelos de trayectoria requieren políticas aprobadas para los endpoints. Las definiciones Pop por primera visita válida y por todas las visitas conservan sus ventanas.

Los resultados son exploratorios en la cohorte que descubrió las comunidades. El análisis condiciona en las etiquetas observadas y sus intervalos omiten la selección e incertidumbre de Mapper. R1–R4 se interpretan como sensibilidades de representación. La estabilidad publicada determina la elegibilidad de inferencia.

## Evidencia de partida

Las auditorías institucionales informaron 159 pacientes basales, 497 episodios Step10 y 177 analitos Step01. R0 incluye 98 features, con 71 analitos N1 identificados por nombre. En 70 de 159 pacientes se detectó alguna medición N1 posterior al día índice con coincidencia en una feature R0 no nula. Entre los 4.452 pares paciente–analito identificados, 3.468 correspondieron a días 1–30, 617 a días 31–180, 142 a días 181–365 y 225 a días posteriores. Esos intervalos eran los de la auditoría histórica. La concordancia Step01–Context acredita consistencia de fechas seleccionadas. La fecha de colección y la identidad original en BTRIS requieren evidencia propia.

El release auditado fue `graph02_current_v1`, referencia R0, estado `PARTIAL_OR_EXPLORATORY`. El inventario `02_graph_feature_role_manifest.csv` documenta roles de descubrimiento. Las listas de revisión independiente, outcomes prespecificados y políticas longitudinales siguen pendientes de revisión clínica.
