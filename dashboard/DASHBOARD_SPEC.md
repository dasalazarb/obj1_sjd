# DASHBOARD_SPEC.md — SjD Research Explorer (Streamlit)

> Repository integration: the user subsequently requested hosting this app under
> `obj1_sjd/dashboard/`. Runtime defaults therefore read `../outputs` and use `..`
> as `SJD_REPO_ROOT`; the local `AGENTS.md` permits dashboard-local edits while
> protecting analytical code and upstream data/results. The original specification
> below is retained as the scientific and editorial contract.

**Destinatario:** Codex (agente de implementación)
**Versión:** 1.0 · **Fecha:** 2026-10-07
**Fuente editorial:** `pres_po1.pptx` (19 slides, versión editada por el investigador) + deck "Objective 1" generado en esta sesión
**Fuente científica:** repositorio `dasalazarb/obj1_sjd` (carpeta `src/`), commit `1dd1662` en el momento de redactar
**Idioma de la UI y del contenido científico:** inglés. Este documento está en español; los identificadores, rutas y textos de UI se escriben tal cual deben aparecer.

---

## 0. Léeme primero (resumen ejecutable)

1. Construyes una **capa de presentación**. No eres dueño de ningún resultado científico (ver §8, regla inviolable).
2. Cada slide del PPTX se convierte en una **sección** de `objective_01`. El mapa exacto está en §1.
3. Cada número que se ve en pantalla se resuelve por **referencia** a una celda de un CSV/JSON de `obj1_sjd` (§1 y §5). Cero números escritos a mano.
4. Los ~170 plots **no son páginas**: viven en un **Variable Explorer** manifest-driven (§6).
5. Si falta un dato, un archivo o una definición: **no lo inventes ni lo infieras**. Muestra `MissingOutput` y anótalo en `OPEN_QUESTIONS.md` (§8.5).
6. Primero crea `AGENTS.md` en la raíz del repo del dashboard con el contenido de §8.7 (Codex lo lee en cada sesión).
7. Orden de trabajo: §10 (fases). Empieza por la Fase 0 (auditoría), no por la UI.
8. **Rutas de cada archivo fuente** (repo en GitHub, script productor, raíz de outputs, qué falta): **Anexo C**.

### Hallazgos de la auditoría que cambian cómo debes construir

| # | Hallazgo | Consecuencia para el dashboard |
|---|---|---|
| H1 | Los CSV de Step 11 y Step 12 **sí** existen en el código con rutas deterministas (§5.2). | Se registran en `output_registry.yml` con ruta exacta. |
| H2 | El script que genera las figuras longitudinales de laboratorio (`... longitudinal laboratory profile`, `... categorical laboratory profile`) **no está en `obj1_sjd` en `main`**. Sus nombres de archivo no son verificables. | **No adivines nombres de archivo.** `figures.csv` se construye con la auditoría de la Fase 0 + curación humana (§6.4). |
| H3 | La imagen del swimmer plot (slide 4) muestra "Phase 1: Initial Full Evaluation", eje en **días** y `n=162`. El código actual (`01_pop_distribution.py`) dibuja un swimmer en **años**, coloreado por Pop, con N basal 159. | La imagen de la slide 4 es probablemente de una versión anterior. En el dashboard se muestra el **output actual** del script, no el PNG del PPTX (§1, slide 4). Pregunta abierta Q1. |
| H4 | La slide 1 cita `scr/blockA/01_table1_baseline.py`. Ese script **no existe** en el repo. Los productores reales son `11_integrated_baseline_characterization.py` y `12_followup_characterization.py`. | El pie de procedencia se genera desde `source_script` del registro, nunca copiando el texto del PPTX. |
| H5 | La slide 3 (Day 1 / Day 6 / Day 7 / Day 9) es un **diagrama conceptual**. Su "Source: 12_followup_summary.csv" solo respalda el conteo de episodios. | Marcar la sección `illustrative: true`; el pie dice "Conceptual diagram — no data source". |
| H6 | En `11_table1_by_pop.csv` las celdas vacías significan **cero pacientes** (ej. Pop1 × `overlap_status: glandular_only`; Pop3 × `ethnicity: LATINO OR HISPANIC`), pero el CSV no lo dice explícitamente. | Renderizar `—` con tooltip "empty in source". La slide 12 lo trata como 0 %. Pedir a upstream escribir `0/32 (0.0%)` (Q3). |
| H7 | `11_table1_overall.csv` y `11_table1_by_pop.csv` guardan estadísticos como **texto** (`"7.5 (6.0–13.2); n=32"`, `"49/62 (79.0%)"`). | Parser estricto y testeado (§5.4). Si no parsea: mostrar el texto crudo y un warning, nunca adivinar. |
| H8 | Las figuras de laboratorio de las slides 15–19 se agrupan por **historial de `SJOGRENS_CLASS`** (paneles A "Class 1 history (no class 2)", B "Ever class 2", C "Class 4 only"), **no** por Pop. Los N de cada panel solo están dentro del PNG. | El dashboard no puede mostrar esos N como texto. Pedir sidecar JSON por figura (Q2). No mezclar leyendas Pop con estos paneles. |
| H9 | En varios PNG de laboratorio la fila `n=` se superpone con el eje `Clinical visit`. | Es defecto upstream. **No recortar, redibujar ni reescalar con filtros**: mostrar el archivo tal cual. |
| H10 | `12_followup_summary.csv` (ancho) y `12_followup_summary_long.csv` (largo) son el mismo contenido. El largo tiene columna `Cohort`. | Consumir el **largo** como canónico (permitirá estratos 11D/15D sin cambiar el loader). El ancho queda registrado como redundante. |
| H11 | `11_baseline_full_variable_inventory.csv` y `11_baseline_variable_availability.csv` se escriben desde el mismo DataFrame (líneas 506–507 de `11_...py`). | Usar solo `availability`. |
| H12 | Step 12 no emite un `provenance.json`; Step 11 sí (`run_date`, `git_commit`, `n_patients`, `script`). | `RunStatus` usa el modo "run sintetizado" (§5.6) hasta que upstream publique `run_manifest.json`. |

---

## 1. Mapa exacto: slide → sección → procedencia por celda

### 1.1 Convenciones de este capítulo

- **Slide N** = número en `pres_po1.pptx` (19 slides).
- **Archivo** usa los IDs del registro (§5.2): `t11_overall`, `t11_by_pop`, `t11_avail`, `t11_dict`, `t12_long`, `t12_ret`, `prov11`.
- **Selector normativo = clave, no número de línea.** Fila se identifica por (`Section`, `Variable`) o `variable` o `Indicator`+`Cohort`. El número de línea (`L##`) es solo ayuda de verificación para la exportación del 2026-10-03; **no se debe codificar**.
- **Campo** indica qué parte de la celda se usa tras el parser de §5.4: `median`, `q1`, `q3`, `n`, `k`, `N`, `pct`, `raw`.
- **Formato**: reglas de visualización (redondeo, unidades). El redondeo es solo de display y usa `ROUND_HALF_UP` (`decimal.Decimal`), no `round()` de Python.
- Un texto con números en `narrative.md` o en `claims.yml` **no** es fuente: usa `{{ref:...}}` (§5.5).

### 1.2 Tabla resumen

| Slide | Título (PPTX) | Sección del dashboard | Componente principal | Datos |
|---|---|---|---|---|
| 1 | Primary Objective 1 / Baseline Characteristics | `objective_01/page.py` → hero | `PageHeader(variant="cover")` | estático |
| 2 | A diagnosis-anchored view of overlap | `01_study_design` | `TimelineDiagram` | estático |
| 3 | A date is not always a visit | `01_study_design` | `EpisodeDiagram` | estático (conceptual) |
| 4 | (swimmer plot, sin título) | `01_study_design` | `FigureViewer` | figura `fig_swimmer` |
| 5 | What data we start with | `02_cohort` | `KpiRow` + `StepFlow` | `t11_overall`, `t12_long`, `t11_dict` |
| 6 | Who is in the cohort | `03_baseline` | `KpiCard` + `BarList` | `t11_overall` |
| 7 | What data are available at baseline | `03_baseline` | `BarList(thresholds)` | `t11_avail` |
| 8 | Disease activity, full cohort | `04_phenotype` | `KpiGrid` | `t11_overall` |
| 9 | Activity and patient-reported outcomes | `04_phenotype` | `StatTable(by_pop)` | `t11_by_pop` |
| 10 | Glandular phenotype and serology | `05_glandular_serology` | `KpiGrid` | `t11_overall`, `t11_avail` |
| 11 | Glandular, serology and laboratory | `05_glandular_serology` | `StatTable(by_pop)` | `t11_by_pop` |
| 12 | Glandular and extraglandular overlap | `06_organ_involvement` | `StackedBarRows` | `t11_by_pop` |
| 13 | How much follow-up we have | `07_followup` | `KpiGrid` + `BarList` | `t12_long` |
| 14 | Retention from the baseline episode | `07_followup` | `ColumnChart` | `t12_ret`, `t12_long` |
| 15–19 | (5 figuras de laboratorio) | `08_variable_explorer` | `VariableExplorer` | `figures.csv` + PNG |

**Slides retiradas por el investigador** (existían en el deck generado y no están en `pres_po1.pptx`): "Four groups by ESSDAI and ESSPRI" (definiciones de Pop) y "Limitations and next steps". Recomendación, no obligación: conservar las definiciones de Pop como cajón `MethodsDetails` en `04_phenotype` (sin ellas las columnas Pop1/2/3 no se interpretan) y llevar las limitaciones a los `Caution` por sección. Contenido de referencia en §1.16.

### 1.3 Slide 1 — Cover

| Elemento | Origen |
|---|---|
| Eyebrow `DESCRIPTIVE / NATURAL HISTORY ANALYSIS`, título `Primary Objective 1`, subtítulo `Baseline Characteristics` | `objectives/objective_01/config.yml` (`eyebrow`, `title`, `subtitle`) — texto editorial |
| Logo (icono de eslabón) | `assets/logos/` — extraído de `ppt/media/image*.png` de la slide 1 |
| Pie `Source: scr/blockA/01_table1_baseline.py` | **No copiar** (H4). Mostrar `RunBadge` con `prov11.script`, `prov11.run_date`, `prov11.git_commit` |

### 1.4 Slide 2 — A diagnosis-anchored view of overlap

- Contenido: línea de tiempo `day 0 → visit → visit → ★ → ○`, hito "Clinical episode", etiquetas "Follow-up". Subtítulo `Baseline = clinical episode`.
- Sección: `01_study_design` (definición de baseline).
- Implementación: componente `TimelineDiagram` en SVG/HTML parametrizado por `config.yml` (`nodes: [...]`). **No usar el PNG del PPTX.** Colores: ver §2 (hito rojo `#FF0000` en el original → usar `--color-danger-text` `#9B2C2C`; estrella `#D9682A`).
- Sin datos. Pie: "Conceptual diagram — no data source".
- Nota: el título habla de "overlap" pero el diagrama define baseline/seguimiento. Se respeta el título del investigador (Q7 si desea cambiarlo).

### 1.5 Slide 3 — A date is not always a visit

- Eyebrow `UNIT OF ANALYSIS`; título `A date is not always a visit`; intro `Components of one evaluation were recorded on different days; they are grouped into one clinical episode.`
- Cuatro chips: `Day 1 · ESSDAI / physician`, `Day 6 · Research component`, `Day 7 · ESSPRI`, `Day 9 · Eye exam`.
- Caja roja `Wrong: every date is a visit` / caja azul `Correct: one clinical episode` (textos en `narrative.md`).
- Notas del orador (a mostrar en `MethodsDetails`): `A research activity (e.g. genomics, CCGO) between visits is not a clinical visit: it could simulate Pop → Unclassifiable → Pop.`
- Componente: `EpisodeDiagram` (HTML estático). `illustrative: true` (H5).
- Dato real asociado (opcional, recomendado): chip "497 clinical episodes · 159 patients" vía `t12_long` (§1.12).

### 1.6 Slide 4 — Swimmer plot

| Elemento | Origen |
|---|---|
| Figura | Producida por `src/block_A/01_pop_distribution.py` → `outputs/figures/blockA/01_pop_distribution/02_pop_distribution_plot.pdf` (figura principal) y paneles por Pop `02_pop_distribution_plot_pop1.pdf`, `..._pop2.pdf`, `..._pop3.pdf`, `..._unclassifiable.pdf` (el sufijo es `baseline_pop.lower()`, línea 579 del script) |
| Leyenda del panel | Está en la propia figura: `Pop1 = ESSDAI ≥5; Pop2 = ESSDAI <5 and ESSPRI ≥5; Pop3 = ESSDAI <5 and ESSPRI <5; grey = insufficient data.` |
| Colores de grupo | `POP_COLORS` del script: Pop1 `#d95f02`, Pop2 `#7570b3`, Pop3 `#1b9e77`, Unclassifiable `#9e9e9e` (ver §2.6) |

Reglas:
- Mostrar el PDF renderizado a PNG (cache por `run_key`, §5.3) con enlace de descarga del PDF original.
- Si los PDF no existen en la ejecución seleccionada, **no usar la imagen del PPTX**: mostrar `MissingOutput` ("Swimmer plot not available for this run"). Registrar Q1.
- Título de sección sugerido (editorial, el PPTX no lo tenía): `Visit timeline per patient`.

### 1.7 Slide 5 — What data we start with

| Elemento | Archivo | Fila (selector) | Columna | Campo | Formato |
|---|---|---|---|---|---|
| `159` "patients in the cohort" | `t11_overall` | `Section=Cohort / demographics`, `Variable=N patients` | `Summary` | `raw` (entero) | entero (L2) |
| `497` "clinical episodes in total" | `t12_long` | `Indicator=Clinical episodes`, `Cohort=Overall` | `Value` | `raw` | entero (L2) |
| `26 / 952` "canonical variables published in Table 1" | `t11_dict` | todas las filas | `include_in_table1` y recuento de filas | ver nota | `{n_true} / {n_rows}` |
| Paso 2: "159 rows, one per patient" | `t11_overall` | `N patients` | `Summary` | `raw` | entero |
| Textos de los 3 pasos | `config.yml` | — | — | — | editorial |

Nota `26 / 952`: es un **recuento de metadatos** (filas con `include_in_table1 == True` sobre filas totales del diccionario). Es una de las operaciones permitidas en §8.2 (`COUNT_METADATA_ROWS`). Cuando upstream publique `n_variables_total` / `n_variables_table1` en un summary JSON, cambiar a referencia directa (Q4). El valor en la exportación del 2026-10-03: 26 verdaderos y 952 filas; `is_canonical` también suma 26.

### 1.8 Slide 6 — Who is in the cohort

Archivo `t11_overall`, columnas `Variable` (clave) y `Summary` (valor, formato `k/N (pct%)`).

| Elemento | Selector (`Section` · `Variable`) | Campo |
|---|---|---|
| `159` enrolled patients | Cohort / demographics · `N patients` | `raw` |
| `95.0%` female `(151/159)` | Cohort / demographics · `sex: FEMALE` | `pct`, `k`, `N` |
| `8 male` | Cohort / demographics · `sex: MALE` | `k` |
| Race — White `70.4% (112)` | `race: WHITE` | `pct`, `k` |
| Race — Black or African American `15.7% (25)` | `race: BLACK OR AFRICAN AMERICAN` | idem |
| Race — Asian `5.0% (8)` | `race: ASIAN` | idem |
| Race — Multiple races `3.1% (5)` | `race: MULTIPLE RACES` | idem |
| Race — Other `0.6% (1)` | `race: OTHER` | idem |
| Race — Unknown `5.0% (8)` | `race: UNKNOWN` | idem |
| Ethnicity line | `ethnicity: LATINO OR HISPANIC` (15.1%, 24), `ethnicity: NOT LATINO OR HISPANIC` (83.6%, 133), `ethnicity: UNKNOWN` (1.3%, 2) | `pct`, `k` |

Reglas:
- Ancho de barra = `pct` parseado (sin normalizar a 100).
- Orden y etiquetas (`White`, `Black or African American`, …) en `config/labels.yml` (§4.3); el orden de las barras es editorial y no depende del orden del CSV (que es alfabético por sección).
- `Unknown` usa color `--color-steel-light` (`#9DB4D6`), el resto `--color-steel` (primera barra `--color-navy`).
- No mostrar edad: `age_at_baseline` está en el contrato de Table 1 de `11_...py` pero **no aparece** en las exportaciones recibidas (la slide antigua mostraba 47.1 años; esa cifra no proviene de estos CSV y no debe reaparecer).

### 1.9 Slide 7 — What data are available at baseline

Archivo `t11_avail`; columnas `variable` (clave), `n_available_for_analysis` (n), `pct_available_for_analysis` (pct, con decimales completos). Display: `pct` a 1 decimal, `n` entero → `93.1% (148)`.

| Etiqueta en slide | `variable` (clave) | n | pct (display) |
|---|---|---|---|
| Demographics | `sex` (también `race`, `ethnicity`) | 159 | 100% |
| Population (Pop) | `pop_status` | 159 | 100% |
| ESSDAI total | `essdai_total` | 148 | 93.1% |
| Extraglandular | `n_extraglandular_domains_active` | 148 | 93.1% |
| PROFAD total | `profad_total` | 118 | 74.2% |
| ESSPRI total | `esspri_total` | 117 | 73.6% |
| SF-36 (PCS, MCS) | `sf36_pcs`, `sf36_mcs` | 116 | 73.0% |
| MDAFS global | `mdafs_global` | 116 | 73.0% |
| Salivary flow | `salivary_flow_unstimulated` | 153 | 96.2% |
| C3, IgG, ESR | `complement_c3__value`, `igg__value`, `esr__value` | 152 | 95.6% |
| C4 | `complement_c4__value` | 151 | 95.0% |
| Minimum Schirmer | `ocular_schirmer_min` | 147 | 92.5% |
| Sicca symptoms | `sicca_any_symptom` | 107 | 67.3% |
| Ocular staining | `ocular_staining_positive` | 104 | 65.4% |
| Focus score | `biopsy_focus_score` | 102 | 64.2% |
| Anti-Ro / Anti-La | `anti_ro_ssa__ever_positive_through_episode`, `anti_la_ssb__ever_positive_through_episode` | 62 | 39.0% |

Reglas:
- Una fila agrupada (varias `variable`) se define en `config.yml` (`members: [...]`). El validador exige que **todos los miembros tengan el mismo `n_available_for_analysis`**; si difieren, mostrar una fila por miembro (no promediar, no tomar mínimo).
- Umbrales de color (presentación): `>= 90` navy; `60 <= x < 90` steel; `< 60` amber. Están en `config.yml` (`bar_thresholds`) y se declaran en el subtítulo: `Navy bar ≥ 90%, blue 60–90%, amber < 60%`. Es una convención visual, no una clasificación científica.
- Alternativa más precisa a mostrar al hacer hover: `n_baseline_total` y `n_nonmissing` de la misma fila.

### 1.10 Slide 8 — Disease activity, full cohort

Archivo `t11_overall`, columna `Summary` (formato `median (q1–q3); n=N`) y `N available`.

| Tarjeta | Selector (`Section` · `Variable`) | Campos | Ejemplo exportación 2026-10-03 |
|---|---|---|---|
| ESSDAI total | Disease activity · `essdai_total` | `median`, `q1`, `q3`, `n` | 2.0 (0.0–4.0) · n=148 |
| ESSPRI total | Disease activity · `esspri_total` | idem | 5.3 (3.7–7.0) · n=117 |
| ESSPRI dryness | Disease activity · `esspri_dryness` | idem | 6.0 (4.0–8.0) · n=117 |
| ESSPRI fatigue | Disease activity · `esspri_fatigue` | idem | 6.0 (3.0–8.0) · n=120 |
| ESSPRI pain | Disease activity · `esspri_pain` | idem | 4.0 (1.8–7.0) · n=120 |
| Active extraglandular domains | Organ involvement · `n_extraglandular_domains_active` | idem | 1.0 (0.0–2.0) · n=148 |

`n` se toma de la columna `N available` (validador: coincide con el `n=` del texto).
Formato de tarjeta: `{median}` grande, `({q1}–{q3}) · n = {n}` pequeño, un decimal.

### 1.11 Slides 9, 11 — Tablas por población

Archivo `t11_by_pop`; columnas `Overall`, `Pop1`, `Pop2`, `Pop3`, `Unclassifiable` (la UI muestra `Overall / Pop1 / Pop2 / Pop3 / Unclassifiable`; en slide 11 abreviado `Unclass.`). Cada fila = (`Section`, `Variable`).

**Slide 9 — "Activity and patient-reported outcomes"**

| Fila de la tabla en la UI | Selector (`Section` · `Variable`) | Formato |
|---|---|---|
| N patients | Cohort / demographics · `N patients` | entero |
| ESSDAI total | Disease activity · `essdai_total` | `median (q1–q3)` 1 decimal |
| ESSPRI total | Disease activity · `esspri_total` | idem; `NA` literal se muestra `NA` |
| Fatigue, MDAFS global | PROs · `mdafs_global` | idem |
| Fatigue, PROFAD total | PROs · `profad_total` | idem |
| SF-36 physical component (PCS) | PROs · `sf36_pcs` | idem |
| SF-36 mental component (MCS) | PROs · `sf36_mcs` | idem |
| n with SF-36 | PROs · `sf36_pcs` → campo `n` por columna | entero. Validador: `n(sf36_pcs) == n(sf36_mcs)` en cada columna; si no, mostrar dos filas |

Valores esperados con la exportación 2026-10-03: N `159 / 32 / 54 / 38 / 35`; n SF-36 `116 / 24 / 52 / 37 / 3`. `ESSPRI total` en `Unclassifiable` = `NA` (L18).

- `Caution` fijo (texto editorial, id `claim: c.pop_pro.definitional`): `ESSDAI and ESSPRI define the groups, so differences in them are expected by construction; in Unclassifiable, PROs have n = 2–3.` El "2–3" **no** es literal: es `min..max` de `n` de `mdafs_global`/`profad_total`/`sf36_*` en la columna `Unclassifiable` → debe enlazarse con `{{ref}}` (`n` = 2 en MDAFS y PROFAD, 3 en SF-36).
- Intro (claim `c.pop_pro.pop2_burden`): `Among Pop1–3, Pop2 shows the highest fatigue and the lowest SF-36.` Es una **afirmación interpretativa**; va en `claims.yml` con `reviewed_hash` (§5.5). Si cambian los valores citados, aparece "interpretation pending review".

**Slide 11 — "Glandular, serology and laboratory"**

| Fila UI | Selector | Formato |
|---|---|---|
| Biopsy focus score | Glandular / extended phenotype · `biopsy_focus_score` | `median (q1–q3)` 1 decimal |
| Ocular staining positive | Glandular / extended phenotype · `Ocular staining positive, n/N (%)` | `k/N (pct%)` |
| Anti-Ro/SSA positive | Serology · `Anti-Ro/SSA positive, n/N (%)` | `k/N (pct%)` |
| Anti-La/SSB positive | Serology · `Anti-La/SSB positive, n/N (%)` | idem |
| Complement C3 | Laboratories · `complement_c3__value` | 1 decimal |
| Complement C4 | Laboratories · `complement_c4__value` | 1 decimal |
| IgG | Laboratories · `igg__value` | **entero**, `ROUND_HALF_UP` (ej. `1996.5 → 1997`); declarado en el pie: `IgG rounded to integer` |
| ESR | Laboratories · `esr__value` | 1 decimal |

- Caution: `Serology has n = 10–20 per group; descriptive comparisons, unadjusted.` Los límites 10–20 salen del denominador `N` de las filas de serología en columnas Pop1–Pop3 (10, 14, 18, 20 en la exportación; el 20 corresponde a `Unclassifiable`). **Enlazar con `{{ref}}`**; no escribir "10–20".
- Intro (claim `c.pop_lab.pop1_igg_esr`): `Pop1 shows the highest IgG and ESR, and more positive ocular staining.` → `claims.yml`.
- Abreviatura `ESR = erythrocyte sedimentation rate` en el pie. Nota de doble redondeo: la fuente ya trae 1 decimal; el entero de IgG es un redondeo del valor ya redondeado. Si upstream publica más decimales (Q5), usarlos.
- `Unclassifiable` se abrevia `Unclass.` solo si el ancho lo exige (decisión de layout, no de contenido).

### 1.12 Slide 10 — Glandular phenotype and serology

Archivo `t11_overall` salvo donde se indica.

| Tarjeta | Selector | Campos | Exportación 2026-10-03 |
|---|---|---|---|
| Unstimulated salivary flow | Glandular / extended phenotype · `salivary_flow_unstimulated` | `median`, `q1`, `q3`, `n` | 1.2 (0.3–2.5) · n=153 |
| Minimum Schirmer | Glandular / extended phenotype · `ocular_schirmer_min` | idem | 5.0 (2.0–13.0) · n=147 |
| Biopsy focus score | Glandular / extended phenotype · `biopsy_focus_score` | idem | 1.0 (1.0–3.0) · n=102 |
| Ocular staining positive | Glandular / extended phenotype · `Ocular staining positive, n/N (%)` | `pct`, `k`, `N` | 27.9% · 29/104 |
| Anti-Ro/SSA positive | Serology · `Anti-Ro/SSA positive, n/N (%)` | idem | 79.0% · 49/62 |
| Anti-La/SSB positive | Serology · `Anti-La/SSB positive, n/N (%)` | idem | 35.5% · 22/62 |

Caution (mixto, con refs):
- `Serology is available in only 62 of 159 patients (39.0%)` → `62` = `N available` de `Anti-Ro/SSA positive, n/N (%)` (`t11_overall`), `159` = `N patients`, `39.0%` = `pct_available_for_analysis` de `anti_ro_ssa__ever_positive_through_episode` en `t11_avail`. **El dashboard no calcula 62/159**; toma el porcentaje ya publicado.
- `Sicca symptoms: 107/107 with documented data (100%)` → `Any sicca symptom present, n/N (%)` (Glandular / extended phenotype), campos `k`, `N`, `pct`. Mantener `107/107`: el denominador **no** es 159.

### 1.13 Slide 12 — Glandular and extraglandular overlap

Archivo `t11_by_pop`, filas Section=`Organ involvement`, columnas `Overall`, `Pop1`, `Pop2`, `Pop3`, `Unclassifiable`.

| Segmento | Variable | Campos |
|---|---|---|
| Glandular only | `overlap_status: glandular_only` | `pct` (y `k/N`) |
| Overlap | `overlap_status: overlap` | idem |
| Unclassifiable | `overlap_status: unclassifiable` | idem |
| Etiqueta de fila `Overall (N = 159)`, `Pop1 (n = 32)`, … | `N patients` por columna | entero |
| Línea "Active extraglandular domains, median: overall 1.0 · Pop1 2.0 · Pop2, Pop3 1.0" | Organ involvement · `n_extraglandular_domains_active`, campo `median` por columna | 1 decimal |

Reglas:
- Celda vacía (Pop1 × `glandular_only`) → segmento de ancho 0 y leyenda `0 patients (empty in source)` (H6). No reportar "0.0 %" como si viniera del CSV hasta que upstream lo escriba.
- Ancho de segmento = `pct` parseado; **no renormalizar**. (La slide generada usó 33.4 % de ancho para una etiqueta 33.3 %; el dashboard no debe hacerlo: si la suma ≠ 100 por redondeo, aceptar la diferencia.)
- Definición (editorial, `MethodsDetails`): `Overlap = active glandular manifestation plus at least one active extraglandular domain, at baseline.` Proviene de `src/derivations/overlap_flags.py` (`derive_overlap_flags`): `overlap` = glandular activo y extraglandular activo; `glandular_only` = glandular activo y extraglandular no activo; `unclassifiable` = no evaluable. Las categorías `extraglandular_only` y `neither` existen en el código pero **no aparecen** en la exportación: mostrar solo las categorías presentes en el CSV.
- La mediana de dominios en `Unclassifiable` (1.0, n=24) existe en el CSV y la slide la omite; el dashboard puede mostrarla.

### 1.14 Slide 13 — How much follow-up we have

Archivo `t12_long` (columnas `Indicator`, `Cohort`, `Value`); usar `Cohort = Overall`.

| Elemento | `Indicator` (clave) | `Value` | Display |
|---|---|---|---|
| `497 clinical episodes in 159 patients` | `Clinical episodes`; `Unique patients` | enteros | `497`, `159` |
| `4.0 years` follow-up `(1.9–6.1)`, `maximum 13.5` | `Follow-up, median (IQR), years`; `Maximum follow-up, years` | `4.0 (1.9–6.1)`; `13.5` | 1 decimal |
| `3.0` episodes per patient `(2.0–4.0)`, `mean 3.1`, `maximum 6` | `Clinical episodes per patient, median (IQR)`; `Clinical episodes per patient, mean`; `Maximum clinical episodes per patient` | `3.0 (2.0–4.0)`; `3.1`; `6` | idem |
| `707 days` inter-visit gap `(394.2–765.0)`, `P90 1006.6` | `Median inter-visit gap, days`; `IQR inter-visit gap, days`; `P90 inter-visit gap, days` | `707.0`; `394.2–765.0`; `1006.6` | mediana a entero (`707`); IQR y P90 tal cual |
| `0.6` episodes per follow-up year `(0.5–0.8)` | `Clinical episodes per follow-up year, median (IQR)` | `0.6 (0.5–0.8)` | 1 decimal |
| Barra `≥ 2` `97.5% (155)` | `Patients with >=2 clinical episodes` | `155 (97.5%)` | `k (pct%)` |
| Barra `≥ 3` `59.7% (95)` | `Patients with >=3 clinical episodes` | `95 (59.7%)` | idem |
| Barra `≥ 5` `13.8% (22)` | `Patients with >=5 clinical episodes` | `22 (13.8%)` | idem |
| Barra `≥ 10` `0.0% (0)` | `Patients with >=10 clinical episodes` | `0 (0.0%)` | idem |
| `Only 4 patients (2.5%) have a single episode.` | `Patients with exactly 1 clinical episode` | `4 (2.5%)` | enlazar con `{{ref}}` |

Nota de unidades: `IQR inter-visit gap, days` es el único valor con rango "a–b" **sin paréntesis**; el parser lo trata como `range` (§5.4).

### 1.15 Slide 14 — Retention

Archivo `t12_ret`; columnas `cohort`, `time`, `days`, `n_retained`, `denominator`, `pct_retained`. Seis filas (`time` ∈ `6 months`, `1 year`, `2 years`, `3 years`, `5 years`, `10 years`; `cohort = Overall`).

| Elemento | Columna |
|---|---|
| Etiqueta de eje X | `time` |
| Altura de la barra | `pct_retained` (eje fijo 0–100; altura = valor publicado, sin recalcular) |
| Etiqueta sobre la barra `96.2%` | `pct_retained`, 1 decimal |
| Etiqueta bajo la barra `153/159` | `n_retained` / `denominator` |
| Subtítulo `denominator 159` | `denominator` (validador: constante en las 6 filas) |
| Panel "Patients with at least one inter-visit gap": `> 180 days: 96.2% (153)`, `> 365 days: 82.4% (131)`, `> 730 days: 54.1% (86)` | `t12_long`: `Patients with at least one gap >180 days`, `... >365 days`, `... >730 days` (`Value` = `k (pct%)`) |

- Colores de barra: `6 months` y `1 year` navy; `2–5 years` steel; `10 years` steel claro (decisión visual; se puede reemplazar por regla `pct >= 80 → navy`, no por significado científico).
- Caution fijo: `Descriptive curve, not a Kaplan–Meier estimator.` (regla de proyecto, no cambia con los datos).
- Cuando `t12_long`/`t12_ret` incluyan `Cohort` ≠ `Overall` (11D, 15D), mostrar un selector de cohorte (`st.segmented_control`/`st.radio`) poblado por los valores presentes. No crear estratos que no existan.

### 1.16 Slides 15–19 — Figuras de laboratorio (semillas del explorer)

Las 5 imágenes del PPTX son ejemplos del conjunto de ~170. Se usan como **casos dorados** del Variable Explorer (§6.9):

| Slide | `lab_id` | Vista | Título dentro de la figura |
|---|---|---|---|
| 15 | `anti_ro_ssa` | categórica longitudinal (barras apiladas por visita) | `anti_ro_ssa: categorical laboratory profile` |
| 16 | `complement_c4` | longitudinal numérica | `complement_c4: longitudinal laboratory profile` |
| 17 | `igg` | longitudinal numérica | `igg: longitudinal laboratory profile` |
| 18 | `rheumatoid_factor` | longitudinal numérica | `rheumatoid_factor: longitudinal laboratory profile` |
| 19 | `wbc` | longitudinal numérica | `wbc: longitudinal laboratory profile` |

Rasgos observados (solo lectura visual; no definir semántica a partir de ellos, ver Q2):
- Tres paneles: `A. Class 1 history (no class 2)`, `B. Ever class 2`, `C. Class 4 only`; eje X `Clinical visit` (1–6); fila `n=` por visita; leyenda `Observed numeric value`, `Patient later transitioned to class 4`, `Observed group mean`, `95% bootstrap CI`.
- La figura categórica remite a `01_labs_categorical_summary_by_visit.csv` ("Full category text is preserved in …").

### 1.17 Slides retiradas — contenido de referencia (opcional)

**Pop definitions** (verificables en el código, `config.py` líneas 177/193 y `01_pop_distribution.py` `classify_pop`): Pop1 = `ESSDAI >= 5`; Pop2 = `ESSDAI < 5` y `ESSPRI >= 5`; Pop3 = `ESSDAI < 5` y `ESSPRI < 5`; `Unclassifiable` = datos insuficientes de ESSDAI/ESSPRI para asignar. Conteos y porcentajes: `t11_overall` filas `Disease activity · pop_status: Pop1/Pop2/Pop3/Unclassifiable` (`32/159 (20.1%)`, `54/159 (34.0%)`, `38/159 (23.9%)`, `35/159 (22.0%)`).
**Limitations** (por sección, como `Caution`): ESSPRI faltante (N missing 42) y ESSDAI faltante (11) en `t11_overall`; serología 62/159; biopsia 102/159; tinción ocular 104/159 (las tres de `t11_avail`); comparaciones por Pop descriptivas y sin ajuste; retención descriptiva.

---

## 2. Sistema visual (extraído del deck)

Fuente: slides 3–14 de `pres_po1.pptx` (el contenido generado) y la slide 1/2 originales del investigador. Escala del PPTX: 13.333 in × 7.5 in. **Conversión a web:** 1 pt = 1.333 px.

### 2.1 Principios (no negociables)

- Austero, claro, "data-forward": una idea por bloque, mucho espacio en blanco, figura protagonista.
- **No** estilo dashboard BI: sin mosaicos de widgets, sin tarjetas decorativas, sin sombras fuertes ni gradientes.
- Ámbar `#B8873B` / fondo `#FBF3E4` **solo para caveats**. Nunca como acento decorativo ni como color de categoría (excepción: la barra < 60 % de disponibilidad, que *es* un aviso).
- Cada bloque termina con pie de procedencia en monoespaciada.

### 2.2 Colores (tokens)

| Token CSS | Hex | Uso | Fuente |
|---|---|---|---|
| `--color-navy` | `#1B2A4A` | texto principal, tarjetas oscuras, cabecera de tabla, barras ≥ umbral | slides 3–14 |
| `--color-steel` | `#4A6FA5` | eyebrow, subtítulos itálicos, barras intermedias, etiquetas de tarjetas claras | slides 3–14 |
| `--color-steel-light` | `#9DB4D6` | etiquetas sobre navy, barra "Unknown", segmento `Glandular only` | slides 3, 5, 12 |
| `--color-steel-pale` | `#C9D6EA` | texto secundario sobre navy | slides 5–10 |
| `--color-tint` | `#EEF2F7` | tarjetas claras, filas alternas de tabla | slides 3–14 |
| `--color-track` | `#E3E8EF` | pista de barras, segmento `Unclassifiable` | slides 6, 7, 12 |
| `--color-bg` | `#FAFBFC` | fondo de página | slides 3–14 |
| `--color-cover-bg` | `#0E2841` | portada (valor real de la slide 1 de `pres_po1.pptx`; `dk1` del tema) | slide 1 |
| `--color-text-muted` | `#5B6573` | pie de procedencia, `n` secundarios | slides 3–14 |
| `--color-text-body-on-tint` | `#2F3E5C` | texto de apoyo sobre tarjeta clara | slides 5, 8, 10 |
| `--color-amber` | `#B8873B` | borde izquierdo de `Caution`; barra < 60 % | slides 9–11, 14 |
| `--color-amber-bg` | `#FBF3E4` | fondo de `Caution` | idem |
| `--color-amber-text` | `#5E4210` | texto de `Caution` | idem |
| `--color-danger` | `#9B2C2C` | borde/título "Wrong", hito de la slide 2 | slide 3 |
| `--color-danger-bg` | `#FBEFEF` | fondo "Wrong" | slide 3 |
| `--color-danger-text` | `#3A2A2A` | texto en caja "Wrong" | slide 3 |
| `--color-on-dark` | `#FAFBFC` / `#DCE6F5` | texto sobre navy (primario / secundario) | slides 3–10 |

Colores **científicos de grupo** (§2.6) se declaran aparte y no se mezclan con esta paleta de interfaz.

### 2.3 Tipografía

| Rol | Familia | Tamaño PPTX → web | Peso/estilo | Notas |
|---|---|---|---|---|
| Eyebrow | Arial | 12 pt → 16 px | 700, MAYÚSCULAS, `letter-spacing: 0.12em`, color steel | `Starting point`, `Baseline`… |
| Título de sección (h2) | Georgia | 30 pt → 40 px | 700, `line-height: 1.1`, navy | |
| Frase introductoria | Arial | 15 pt → 20 px | *italic*, steel, `line-height: 1.35` | "Introductory italicized framing sentences" |
| Cifra KPI grande | Georgia | 44–48 pt → 59–64 px (tarjeta principal); 28–32 pt → 37–43 px (secundaria) | 700 | |
| Etiqueta de tarjeta | Arial | 14–16 pt → 19–21 px | 700 | |
| Cuerpo / tabla / barras | Arial | 13 pt → 17 px | 400 | `font-variant-numeric: tabular-nums` en tablas |
| `Caution` | Arial | 13 pt → 17 px | `Caution:` en 700 | |
| Procedencia (`Source:`) | **Courier New** | 12 pt → 16 px | 400, `--color-text-muted` | en todas las secciones |
| Cover título | Georgia (el original usa Cambria 44 pt) | → 59 px | 700 | |

Fallbacks: `Georgia, 'Times New Roman', serif` y `Arial, Helvetica, sans-serif`. La slide 1 y 2 del investigador usan **Cambria/Calibri**; el resto del deck usa Georgia/Arial. **Decisión:** estandarizar en Georgia/Arial (mayoría del deck, fuentes universales, sin licencias). Registrar en `config/theme.py` (`HEADING_FONT`, `BODY_FONT`) para cambiar en un solo lugar.

### 2.4 Geometría y componentes visuales

| Elemento | Especificación |
|---|---|
| Ancho máximo de lectura | 1100–1200 px, centrado; márgenes laterales ≥ 24 px |
| Radio de esquina | tarjetas 16 px (12 px en `Caution`), pistas de barra 8 px |
| Tarjeta oscura | fondo navy, padding 28–40 px, etiqueta `--color-steel-light`, cifra `--color-on-dark`, detalle `--color-steel-pale` |
| Tarjeta clara | fondo `--color-tint`, etiqueta steel, cifra navy, detalle `--color-text-body-on-tint` |
| Tabla | cabecera navy con texto blanco 700; filas alternas `--color-tint`; sin bordes verticales; 17 px; primera columna ancho ≥ 24 %; alineación izquierda; `tabular-nums` |
| Barra horizontal | pista `--color-track`, alto 24 px, radio 8 px; etiqueta izquierda ancho fijo; valor a la derecha, 700, alineado a la derecha |
| Barra apilada | alto 56 px, segmentos sin separación; texto 700 centrado dentro (≥ 16 px); `Glandular only` `#9DB4D6` (texto navy), `Overlap` navy (texto blanco), `Unclassifiable` `#E3E8EF` (texto navy) |
| Columna (retención) | barra navy/steel, esquinas superiores redondeadas 8 px, valor 700 encima, etiqueta de tiempo y `n/N` debajo |
| `Caution` | `border-left: 8px solid --color-amber`, fondo `--color-amber-bg`, texto `--color-amber-text`, padding 16–22 px / 24–32 px |
| Caja "Wrong/Correct" (slide 3) | 2 columnas; "Wrong": `border: 2px solid --color-danger`, fondo `--color-danger-bg`; "Correct": fondo navy, texto claro |
| Flecha entre pasos | `x-shape arrow-right` 64×32 px steel; en web: SVG inline |
| Iconos | en el deck original, insignias circulares navy con iconos; en la versión actual solo el logo de portada. Usar SVG inline; no depender de fuentes de iconos |

### 2.5 Reglas de contraste y accesibilidad

- Texto normal ≥ 4.5 : 1, ≥ 3 : 1 para ≥ 24 px. Tokens arriba cumplen sobre su fondo declarado.
- **El color nunca es el único portador de significado:** barras de disponibilidad muestran el % y n; `Caution` lleva la palabra "Caution:"; "Wrong/Correct" lleva texto.
- No usar rojo/verde como par categórico. Los colores de grupo Pop vienen de la figura (§2.6) y se acompañan siempre de etiqueta textual.

### 2.6 Colores científicos de grupo (no cambiar entre capítulos)

| Grupo | Hex | Origen |
|---|---|---|
| Pop1 | `#d95f02` | `POP_COLORS` en `src/block_A/01_pop_distribution.py` |
| Pop2 | `#7570b3` | idem |
| Pop3 | `#1b9e77` | idem |
| Unclassifiable | `#9e9e9e` | idem |
| Laboratorio, paneles A/B/C (azul / naranja / verde) | los de cada PNG | **no recolorear** |

Uso: chips de leyenda y gráficos Plotly que se creen en el futuro. Las tablas por Pop conservan la cabecera navy.

### 2.7 `theme.css`, `config.toml` (arranque)

```css
:root{
  --color-navy:#1B2A4A; --color-steel:#4A6FA5; --color-steel-light:#9DB4D6; --color-steel-pale:#C9D6EA;
  --color-tint:#EEF2F7; --color-track:#E3E8EF; --color-bg:#FAFBFC; --color-cover-bg:#0E2841;
  --color-text-muted:#5B6573; --color-text-body-on-tint:#2F3E5C; --color-on-dark:#FAFBFC; --color-on-dark-2:#DCE6F5;
  --color-amber:#B8873B; --color-amber-bg:#FBF3E4; --color-amber-text:#5E4210;
  --color-danger:#9B2C2C; --color-danger-bg:#FBEFEF; --color-danger-text:#3A2A2A;
  --font-heading:Georgia,'Times New Roman',serif; --font-body:Arial,Helvetica,sans-serif; --font-mono:'Courier New',monospace;
  --radius-card:16px; --radius-caution:12px; --radius-bar:8px;
}
```

```toml
# .streamlit/config.toml
[theme]
base = "light"
primaryColor = "#4A6FA5"
backgroundColor = "#FAFBFC"
secondaryBackgroundColor = "#EEF2F7"
textColor = "#1B2A4A"
font = "sans serif"
[server]
runOnSave = false
```

Reglas CSS: toda personalización en `shared/styles/theme.css`, inyectada **una vez**; clases propias con prefijo `sjd-`; no apuntar a selectores internos de Streamlit más allá de contenedor principal y sidebar (frágiles entre versiones).

---

## 3. Componentes reutilizables (`shared/components/`)

Todos reciben **referencias** (`ref`) y nunca valores numéricos. Cada componente devuelve HTML con clases `sjd-*` (vía `st.html`/`st.markdown(..., unsafe_allow_html=True)`), salvo `FigureViewer` y `VariableExplorer`, que usan widgets nativos.

### 3.1 Resolución de referencias

```python
# shared/loaders/refs.py
Ref = str  # "t11_overall:Disease activity|essdai_total@Summary#median"
resolve(ref: Ref, run: Run) -> Value   # Value(raw:str, number:Decimal|None, display:str, source:Provenance)
```

Gramática: `<registry_id>:<row-key>@<column>#<field>`.
- `row-key` = `Section|Variable` (Table 1), `variable` (availability), `Indicator|Cohort` (follow-up long), `time` (retention).
- `field` ∈ `raw | median | q1 | q3 | n | k | N | pct | range_lo | range_hi`.
- Cada `Value` conserva `source = {file, row_key, column, line_hint, run_key}`. Esa estructura alimenta `EvidenceFooter` y el modo "inspect".

### 3.2 Catálogo

| Componente | Props | Slides |
|---|---|---|
| `page_header(eyebrow, title, intro, variant="section"\|"cover")` | textos (editoriales) | 1 y todas |
| `section_header(id, eyebrow, title, intro)` | idem; ancla `#id` | 2–14 |
| `kpi_card(label, value_ref, detail_refs, tone="dark"\|"light", size="lg"\|"md"\|"sm")` | `detail_refs` = lista formateada (p. ej. `({q1}–{q3}) · n = {n}`) | 5, 6, 8, 10, 13 |
| `kpi_row(cards)` / `kpi_grid(cards, cols)` | layout | idem |
| `bar_list(rows, thresholds=None, value_fmt)` | `rows[i] = {label, pct_ref, n_ref?, group_members?}` | 6, 7, 13 |
| `stat_table(columns, rows, header_tone="navy")` | `columns` = columnas del CSV; `rows[i] = {label, row_key, field, fmt}` | 9, 11 |
| `stacked_bar_rows(rows, segments)` | segmentos con `pct_ref`; reglas de §1.13 | 12 |
| `column_chart(rows)` | `label_ref, pct_ref, n_ref, denom_ref` | 14 |
| `step_flow(steps)` | 3 pasos con flechas | 5 |
| `timeline_diagram(config)` / `episode_diagram(config)` | solo texto/estructura | 2, 3 |
| `caution(text_md, claim_id=None)` | `text_md` con `{{ref:...}}` | 9–12, 14 |
| `interpretation(claim_id)` | muestra el párrafo y el estado `reviewed`/`pending` | 9, 11 |
| `methodology(md_path, title)` | `st.expander` con Markdown | 3, 12, def. Pop |
| `source(refs)` / `evidence_footer(refs)` | genera línea Courier: `Source: <archivos> · Generated by: <script> · Run: <fecha>` | todas |
| `run_badge(run)` | `run_key`, `run_date`, `git_commit`, estado | cabecera |
| `figure_viewer(figure_id)` | carga PNG/SVG/PDF→PNG; zoom; descarga | 4, 15–19 |
| `variable_explorer(objective_id)` | §6 | 15–19 |
| `missing_output(what, expected_path)` | caja neutra: `Not available for this run` + ruta esperada | cualquiera |
| `pending_review_banner(claim_id)` | `Interpretation pending review` | claims |
| `slide_nav()` | modo presentación: anterior/siguiente | fase 3 |

### 3.3 Comportamiento común

- Si `resolve()` falla (archivo ausente, fila ausente, parser) → el componente renderiza `missing_output` **en su lugar**, no rompe la página y registra el motivo en `st.session_state["sjd_diagnostics"]`.
- Cada valor renderizado lleva `title=` con `file · row · column` (hover) y `data-ref` para el modo inspect (`?inspect=1`).
- `caution()` aplica la regla de ámbar de §2.1.
- Ningún componente importa `numpy`, `scipy`, `statsmodels` ni agrega columnas (§8.3).

---

## 4. Arquitectura de carpetas

### 4.1 Árbol definitivo

```text
sjd_research_explorer/
├── AGENTS.md                      # reglas duras para Codex (§8.7)
├── DASHBOARD_SPEC.md              # este archivo
├── OPEN_QUESTIONS.md              # Codex anota aquí lo que no puede resolver (§8.5)
├── README.md                      # un solo comando de arranque
├── pyproject.toml
├── app.py                         # st.navigation(...) construido desde navigation.yml
├── .streamlit/config.toml
│
├── config/
│   ├── navigation.yml
│   ├── studies.yml
│   ├── output_registry.yml        # §5.2 — única fuente de rutas
│   ├── variable_catalog.csv       # §6.3
│   └── labels.yml                 # etiquetas legibles, orden, decimales (§4.3)
│
├── shared/
│   ├── components/                # §3 (un archivo por componente)
│   ├── loaders/
│   │   ├── registry.py            # lee output_registry.yml; rechaza rutas no registradas
│   │   ├── csv_loader.py          # dtype=str, keep_default_na=False
│   │   ├── json_loader.py
│   │   ├── figure_loader.py       # PNG/SVG/PDF→PNG con cache
│   │   ├── manifest_loader.py     # run_manifest.json o run sintetizado
│   │   ├── parsers.py             # §5.4
│   │   └── refs.py                # §3.1
│   ├── utils/{formatting.py, paths.py, validation.py, provenance.py}
│   └── styles/{theme.css, typography.css}
│
├── objectives/
│   ├── objective_01/
│   │   ├── README.md
│   │   ├── page.py                # orquesta secciones
│   │   ├── narrative.md           # texto editorial con {{ref:...}}
│   │   ├── config.yml             # §4.2
│   │   ├── sections/
│   │   │   ├── 01_study_design.py
│   │   │   ├── 02_cohort.py
│   │   │   ├── 03_baseline.py
│   │   │   ├── 04_phenotype.py
│   │   │   ├── 05_glandular_serology.py
│   │   │   ├── 06_organ_involvement.py
│   │   │   ├── 07_followup.py
│   │   │   └── 08_variable_explorer.py
│   │   └── metadata/
│   │       ├── claims.yml
│   │       ├── figures.csv        # §6.4
│   │       └── tables.csv         # inventario de tablas consumidas
│   ├── objective_02/ … objective_04/   # plantillas vacías (§7)
│   ├── secondary_01/ … secondary_05_*/
│   └── exploratory/…
│
├── studies/                       # graph/, pharma/, economic/ — misma estructura que un objetivo
│
├── assets/{logos,icons,static}/
├── scripts/
│   ├── audit_outputs.py           # Fase 0: inventario de outputs reales (solo lectura)
│   ├── new_study.py               # scaffold (§7)
│   └── check_no_science.py        # lint de §8.3 (también corre en tests)
└── tests/
    ├── fixtures/objective_01/     # CSV reales agregados que aporte el investigador (§10.4)
    ├── test_registry.py
    ├── test_parsers.py
    ├── test_refs_objective_01.py  # cada ref de §1 resuelve al valor esperado
    ├── test_no_science.py
    ├── test_no_hardcoded_numbers.py
    ├── test_manifests.py
    ├── test_outputs_exist.py
    └── test_pages_smoke.py        # streamlit.testing.v1.AppTest
```

### 4.2 `objectives/objective_01/config.yml` (esquema)

```yaml
id: objective_01
order: 1
eyebrow: "DESCRIPTIVE / NATURAL HISTORY ANALYSIS"
title: "Primary Objective 1"
subtitle: "Baseline Characteristics"
study_family: natural_history
outputs_registry_ids: [t11_overall, t11_by_pop, t11_avail, t11_dict, t12_long, t12_ret, prov11, fig_swimmer]
sections:
  - id: study_design
    file: sections/01_study_design.py
    slides: [2, 3, 4]
    eyebrow: "UNIT OF ANALYSIS"
    title: "A date is not always a visit"
    illustrative: true
  - id: cohort
    file: sections/02_cohort.py
    slides: [5]
    eyebrow: "STARTING POINT"
    title: "What data we start with"
  # … una entrada por sección de §1.2
presentation:
  default_mode: read          # read | present
```

### 4.3 `config/labels.yml` (esquema)

Mapa de **clave de datos → etiqueta de UI**, orden y formato. Es editorial y no contiene números de resultados.

```yaml
labels:
  essdai_total:            {ui: "ESSDAI total", decimals: 1}
  esspri_total:            {ui: "ESSPRI total", decimals: 1}
  esspri_dryness:          {ui: "ESSPRI dryness", decimals: 1}
  mdafs_global:            {ui: "Fatigue, MDAFS global", decimals: 1}
  profad_total:            {ui: "Fatigue, PROFAD total", decimals: 1}
  sf36_pcs:                {ui: "SF-36 physical component (PCS)", decimals: 1}
  sf36_mcs:                {ui: "SF-36 mental component (MCS)", decimals: 1}
  igg__value:              {ui: "IgG", decimals: 0, rounding: half_up}
  esr__value:              {ui: "ESR", decimals: 1, long: "Erythrocyte sedimentation rate"}
  complement_c3__value:    {ui: "Complement C3", decimals: 1}
  complement_c4__value:    {ui: "Complement C4", decimals: 1}
  salivary_flow_unstimulated: {ui: "Unstimulated salivary flow", decimals: 1}
  ocular_schirmer_min:     {ui: "Minimum Schirmer", decimals: 1}
  biopsy_focus_score:      {ui: "Biopsy focus score", decimals: 1}
  n_extraglandular_domains_active: {ui: "Active extraglandular domains", decimals: 1}
groups:
  Pop1: {ui: "Pop1"}; Pop2: {ui: "Pop2"}; Pop3: {ui: "Pop3"}
  Unclassifiable: {ui: "Unclassifiable", short: "Unclass."}
categories:
  "race: BLACK OR AFRICAN AMERICAN": {ui: "Black or African American"}
  # …
```

**Regla de nombres:** la UI nunca muestra identificadores internos tipo `sf36_mcs`, `essdai_total`, `complement_c4__value`; siempre la etiqueta de `labels.yml` (abreviada si es larga). Un test falla si el DOM renderizado contiene un identificador de `labels.yml` como texto visible.

### 4.4 Separación de responsabilidades

- `objectives/<id>/` contiene **qué se cuenta** (secciones, narrativa, claims, figuras) y **dónde está** (registro). No contiene estilos ni loaders propios.
- `shared/` contiene **cómo se muestra** y **cómo se lee**. No contiene nada específico de un objetivo.
- `config/output_registry.yml` es la única fuente de rutas de archivos de `obj1_sjd`. Ningún `.py` contiene una ruta de resultados.

---

## 5. Reglas de ingestión (CSV / JSON / PNG / SVG / PDF)

### 5.1 Raíz de outputs

- Variable de entorno `SJD_OUTPUTS_ROOT` (por defecto `../obj1_sjd/outputs`) y `SJD_REPO_ROOT` (por defecto `../obj1_sjd`).
- Solo **lectura**. El dashboard nunca escribe, mueve ni borra nada dentro de `obj1_sjd`.
- No hay copia de datos al repo del dashboard, salvo `tests/fixtures/` (agregados, aportados por el investigador).

### 5.2 `config/output_registry.yml` — entradas de Objective 1 (rutas verificadas en el código)

Inventario completo con URLs de GitHub y estado de cada archivo: **Anexo C**. Rutas relativas a `SJD_OUTPUTS_ROOT` (`common.OUTPUTS_DIR`); las de `data/` relativas a `SJD_REPO_ROOT`. Convención de `common.py`: `outputs/tables/blockA/<script>/…`, `outputs/qc/blockA/<script>/…`, `outputs/figures/blockA/<script>/…`.

```yaml
version: 1
outputs:
  t11_overall:
    path: tables/blockA/11_integrated_baseline_characterization/11_table1_overall.csv
    kind: csv
    key: [Section, Variable]
    columns: [Section, Variable, "N available", "N missing", Summary]
    producer: src/block_A/11_integrated_baseline_characterization.py
  t11_by_pop:
    path: tables/blockA/11_integrated_baseline_characterization/11_table1_by_pop.csv
    kind: csv
    key: [Section, Variable]
    columns: [Section, Variable, Overall, Pop1, Pop2, Pop3, Unclassifiable]
    producer: src/block_A/11_integrated_baseline_characterization.py
  t11_avail:
    path: tables/blockA/11_integrated_baseline_characterization/11_baseline_variable_availability.csv
    kind: csv
    key: [variable]
    columns: [variable, clinical_block, n_baseline_total, n_nonmissing, pct_nonmissing, n_valid, pct_valid, n_available_for_analysis, pct_available_for_analysis]
    producer: src/block_A/11_integrated_baseline_characterization.py
  t11_dict:
    path: tables/blockA/11_integrated_baseline_characterization/11_variable_dictionary.csv
    kind: csv
    key: [variable]
    columns: [variable, clinical_block, role, is_canonical, include_in_table1, description, source_script, dtype]
    producer: src/block_A/11_integrated_baseline_characterization.py
  t12_long:
    path: tables/blockA/12_followup_characterization/12_followup_summary_long.csv
    kind: csv
    key: [Indicator, Cohort]
    columns: [Indicator, Cohort, Value]
    producer: src/block_A/12_followup_characterization.py
  t12_ret:
    path: tables/blockA/12_followup_characterization/12_retention.csv
    kind: csv
    key: [cohort, time]
    columns: [cohort, time, days, n_retained, denominator, pct_retained]
    producer: src/block_A/12_followup_characterization.py
  prov11:
    path_root: repo
    path: data/analytic/blockA/11_integrated_baseline_characterization/11_integrated_baseline_patient_level.provenance.json
    kind: json
    fields: [input_file, input_integration_version, run_date, git_commit, baseline_definition, unit_of_analysis, n_patients, script]
    note: "Solo se lee el .provenance.json. El .parquet/.csv vecino es de pacientes individuales y está prohibido (§5.7)."
  fig_swimmer:
    path: figures/blockA/01_pop_distribution/02_pop_distribution_plot.pdf
    kind: figure
    format: pdf
    panels_glob: figures/blockA/01_pop_distribution/02_pop_distribution_plot_*.pdf
    producer: src/block_A/01_pop_distribution.py
redundant:
  t12_wide:  tables/blockA/12_followup_characterization/12_followup_summary.csv
  t11_inventory: tables/blockA/11_integrated_baseline_characterization/11_baseline_full_variable_inventory.csv
```

Nota: las exportaciones recibidas traen sufijos de descarga (`-4`, `-2`); el nombre real es el de la tabla anterior.

Otras tablas del repo candidatas a objetivos futuros (rutas por código, sin verificar contenido): `tables/blockA/06_overlap_glandular/06_overlap_baseline.csv`, `06_overlap_by_clinical_visit_number.csv`, `06_extraglandular_domains_by_clinical_visit_number.csv`, `06_incident_extraglandular_domains.csv`, `06_pairwise_domain_associations_clinical_baseline.csv`; `tables/blockA/01_serological_profile/01_labs_episode_coverage.csv`. Incorporarlas solo cuando se construya su objetivo y con la auditoría de la Fase 0.

### 5.3 Reglas por tipo de archivo

**CSV**
1. Leer con `pandas.read_csv(..., dtype=str, keep_default_na=False, encoding="utf-8")`. Así `NA` y la celda vacía se conservan literalmente y se distinguen.
2. Validar contra `columns` del registro: faltante → `SchemaError` (se muestra `missing_output`, no se corrige).
3. Validar unicidad de `key`.
4. Prohibido: `fillna`, `astype(float)` masivo, `groupby`, `agg`, `merge` entre tablas de resultados. Solo conversión celda a celda vía `parsers.py`.
5. `Section`/`Variable`/`Indicator` se comparan **exactamente** (sin `lower()` ni `strip()` agresivo).
6. Rechazar cualquier CSV cuya cabecera contenga `patient_id`, `ids__*`, `*_dob`, o similares (§5.7).

**JSON**: leer con `json.load`; validar campos listados en el registro; fechas como texto.

**PNG / SVG**: mostrar tal cual. PNG con `st.image(path, use_container_width=True)` y botón de zoom (modal o `st.dialog`). SVG se inserta como `<img>` (no `innerHTML`; sin scripts). Sin recompresión, recorte ni cambios de color.

**PDF**: renderizar la página 1 a PNG (PyMuPDF, 200 dpi) y cachear en `.cache/figures/<run_key>/<hash>.png`; mostrar enlace de descarga del PDF original.

**Parquet**: **no se lee ningún parquet de resultados** en el MVP (los de `obj1_sjd` son de nivel paciente). Si en el futuro upstream publica parquets agregados, se registran explícitamente con `kind: parquet` y `aggregate_only: true`.

**Plotly JSON** (futuro): `kind: plotly_json`; se carga con `plotly.io.from_json` sin modificar trazas ni datos.

### 5.4 Parsers (celda → valor), estrictos y testeados

Todos operan sobre `str`; retornan `ParsedValue(kind, raw, fields)` o `ParseFailure(raw)`. **Si falla, se muestra `raw` y un warning; nunca se intenta "arreglar".**

| Patrón (regex de referencia) | Ejemplo | Campos |
|---|---|---|
| `^(?P<median>-?\d+(?:\.\d+)?) \((?P<q1>-?\d+(?:\.\d+)?)–(?P<q3>-?\d+(?:\.\d+)?)\); n=(?P<n>\d+)$` | `7.5 (6.0–13.2); n=32` | `median`, `q1`, `q3`, `n` |
| `^(?P<median>-?\d+(?:\.\d+)?) \((?P<q1>-?\d+(?:\.\d+)?)–(?P<q3>-?\d+(?:\.\d+)?)\)$` | `4.0 (1.9–6.1)` | `median`, `q1`, `q3` |
| `^(?P<k>\d+)/(?P<N>\d+) \((?P<pct>\d+(?:\.\d+)?)%\)$` | `49/62 (79.0%)` | `k`, `N`, `pct` |
| `^(?P<k>\d+) \((?P<pct>\d+(?:\.\d+)?)%\)$` | `155 (97.5%)` | `k`, `pct` |
| `^(?P<lo>\d+(?:\.\d+)?)–(?P<hi>\d+(?:\.\d+)?)$` | `394.2–765.0` | `range_lo`, `range_hi` |
| `^-?\d+(?:\.\d+)?$` | `1006.6` | `number` |
| `^NA$` | `NA` | `na` |
| `^$` | *(vacío)* | `empty` |

Notas: el separador de rango es **en-dash U+2013** (`–`); el `n=` va tras `; ` sin espacio alrededor del `=`. `Decimal` para todo; `ROUND_HALF_UP` solo al formatear. Casos de prueba: todas las celdas de §1 (los valores esperados se listan en las tablas de §1 para la exportación 2026-10-03) más tres cadenas malformadas para probar el fallo controlado.

### 5.5 Textos con números y claims

- En `narrative.md` y cualquier texto editorial **no se escriben cifras de resultados**; se usan plantillas: `{{ref:t11_overall:Cohort / demographics|N patients@Summary#raw}}`.
- Un test (`test_no_hardcoded_numbers.py`) falla si un `.md`/`.yml` editorial contiene un número decimal o un entero ≥ 10 fuera de `{{ref:…}}`, salvo una lista blanca de patrones (años, números de slide/sección, `Pop1`–`Pop3`, `SF-36`, `ESSDAI ≥ 5`/`ESSPRI ≥ 5`, fechas).
- **Claims** (`metadata/claims.yml`):

```yaml
- id: c.pop_pro.pop2_burden
  section: phenotype
  text: "Among Pop1–3, Pop2 shows the highest fatigue and the lowest SF-36."
  evidence_refs:
    - t11_by_pop:PROs|mdafs_global@Pop1#median
    - t11_by_pop:PROs|mdafs_global@Pop2#median
    - t11_by_pop:PROs|mdafs_global@Pop3#median
    - t11_by_pop:PROs|sf36_pcs@Pop2#median
    - t11_by_pop:PROs|sf36_mcs@Pop2#median
  reviewed_by: "<iniciales>"
  reviewed_hash: "<sha256 de los valores raw de evidence_refs al aprobar>"
  status: reviewed          # draft | reviewed
  manuscript_target: "Results / Baseline phenotype"
  script: "11_integrated_baseline_characterization.py"
  output: "tables/blockA/11_integrated_baseline_characterization/11_table1_by_pop.csv"
```

- Mecanismo de revisión: al cargar, se calcula el hash de los `raw` de `evidence_refs`. Si ≠ `reviewed_hash` → `pending_review_banner`. **No hay evaluación de si la conclusión sigue siendo cierta**; solo detección de cambio. Aprobar = editar `reviewed_hash` (acción humana).
- Este esquema mapea el `story_map.md` (Claim · Evidence · Script · Output · Manuscript target). Hoy `story_map.md` solo contiene el diagrama Step 10→11/12; los claims se redactan aquí.

### 5.6 Identidad de ejecución (`run`)

`obj1_sjd` hoy **no** escribe `run_manifest.json`. Dos modos:

- **Modo A (objetivo):** si existe `run_manifest.json` (esquema en §5.8) con `status: "complete"`, se usa su `run_id`. `latest.json` apunta a la última completa. Es lo que se debe pedir a upstream.
- **Modo B (interino, el que se implementa ya):** `run_key = sha256( para cada output registrado: (id, size, mtime_ns, sha256 del contenido de los CSV pequeños) ) [:12]`. `RunBadge` muestra: `run_key`, `prov11.run_date`, `prov11.git_commit`, y un aviso `Step 12 has no provenance file` mientras no exista.
- Reglas comunes:
  1. Todos los valores de una página se resuelven con el **mismo** `run_key` (se fija al inicio de la sesión en `st.session_state["run_key"]`; "Refresh" lo recalcula y limpia caché).
  2. Cachés (`st.cache_data`) incluyen `run_key` en la clave.
  3. Si los archivos cambian durante la sesión, se muestra `A newer run is available — Refresh`; no se recarga solo.
  4. Si un archivo registrado no existe: `missing_output`; **nunca** se reutiliza un valor de otra ejecución ni de `tests/fixtures/`.
  5. Si `prov11.n_patients` ≠ `N patients` de `t11_overall`, mostrar warning de inconsistencia.

### 5.7 Lista de rutas **prohibidas** (nivel paciente)

Contienen filas por paciente; el loader las rechaza aunque estén en el registro por error:
`11_integrated_baseline_patient_level.parquet|.csv`, `11_baseline_patient_audit.csv` (qc), `12_patient_followup_metrics.csv`, `12_intervisit_gaps.csv`, `12_followup_episode_audit.csv`, `12_zero_day_gap_audit.csv`, `10_integrated_longitudinal_clinical_episode.parquet`, `06_overlap_episode_level.*`, `06_overlap_baseline_patient_audit.csv`, `01_labs_episode_wide.parquet`, `*_patient_id_crosswalk_qc.csv`, y todo lo bajo `data/`. Test: `test_registry.py` verifica que ningún `path` coincide con estos patrones y que ningún CSV cargado tiene columnas de identificadores.

Las figuras (swimmer, perfiles de laboratorio) son agregadas visualmente pero **pueden mostrar puntos individuales** (los PNG de laboratorio dibujan valores observados por paciente). Mantenerlas fuera de despliegues externos hasta que el investigador confirme (Q6). Ejecución local/institucional por defecto.

### 5.8 Esquemas para pedir a upstream (no bloquean el MVP)

`run_manifest.json` (por script, junto a sus tablas):

```json
{
  "study": "objective_01",
  "script": "11_integrated_baseline_characterization.py",
  "run_id": "2026-10-03T214500Z_ab12cd3",
  "status": "complete",
  "created_at_utc": "2026-10-03T21:45:00Z",
  "git_commit": "ab12cd3",
  "schema_version": "1.0",
  "inputs": {"integration_version": "clinical_episode_curated_v2"},
  "assets": [
    {"id": "t11_overall", "path": "tables/blockA/11_integrated_baseline_characterization/11_table1_overall.csv", "sha256": "…"}
  ]
}
```

`summary.json` por script con los valores que hoy se obtienen parseando texto (`n_patients`, `n_clinical_episodes`, `n_variables_total`, `n_variables_table1`, medianas/IQR numéricos). Sidecar por figura: `<figure>.meta.json` con `figure_id`, `variable_id`, `view`, `kind`, `panel_groups`, `n_by_panel`, `source_script`, `run_id`.

---

## 6. Variable Explorer — diseño para los ~170 plots

### 6.1 Principio

Las figuras son **evidencia subordinada** a una pregunta (nunca estructura de navegación). ~170 PNG se exploran como una colección: `Domain → Variable → View`. Una sola página, un solo componente (`variable_explorer`).

### 6.2 Layout (una pantalla, sin scroll lateral)

```text
LONGITUDINAL VARIABLE EXPLORER                         [Run badge]
Category  [ Complement ▼ ]   Variable [ Complement C4 ▼ ]   View [ Longitudinal ▼ ]
─────────────────────────────────────────────────────────────────
  ◀ prev variable                                     next variable ▶
  [ FIGURE (ancho completo; zoom; download) ]
─────────────────────────────────────────────────────────────────
  Summary  |  Data availability  |  Interpretation  |  Source
```

- Controles: tres `selectbox` encadenados (Category, Variable, View). Las opciones salen de `variable_catalog.csv` ∩ `figures.csv` (solo lo que tiene figura). Si una combinación no tiene figura, la vista se ofrece deshabilitada con la razón.
- Búsqueda de texto sobre `display_name` (`st.text_input`) que filtra el selectbox de variable.
- Botones prev/next dentro del dominio; atajos de teclado ← → solo si se añade el pequeño componente JS de la fase 3.
- URL profunda: `?obj=objective_01&var=complement_c4&view=longitudinal` (`st.query_params`), para compartir una figura exacta en una reunión.
- Modo **Compare** (opcional, fase 3): dos columnas con dos variables/vistas; sin escalas compartidas (las figuras son estáticas).
- Modo **Coverage** (pestaña): matriz `variable × view` con ✔/— (texto "available"/"missing", no solo color) para ver de un vistazo qué falta.

### 6.3 `config/variable_catalog.csv`

Un solo catálogo; los dropdowns **no** están escritos a mano. Columnas (las 6 primeras son las del documento de contexto; el resto se añade):

```csv
variable_id,display_name,domain,data_type,default_view,active,table1_variable,abbreviation,decimals,unit_label,sort_order
anti_ro_ssa,Anti-Ro / SSA,Serology,categorical,categorical_longitudinal,true,anti_ro_ssa__ever_positive_through_episode,,,,10
complement_c3,Complement C3,Complement,numeric,longitudinal,true,complement_c3__value,C3,1,,20
complement_c4,Complement C4,Complement,numeric,longitudinal,true,complement_c4__value,C4,1,,21
igg,IgG,Immunoglobulins,numeric,longitudinal,true,igg__value,IgG,0,,30
wbc,WBC,Hematology,numeric,longitudinal,true,,WBC,1,,40
rheumatoid_factor,Rheumatoid Factor,Serology,numeric,longitudinal,true,,RF,1,,11
```

- `variable_id` = el `lab_id` que aparece en el título de la figura (`complement_c4`, `igg`, `wbc`, `rheumatoid_factor`, `anti_ro_ssa`).
- `table1_variable` = nombre en las tablas de Step 11 (`complement_c4__value`); permite cruzar con `t11_avail`/`t11_by_pop` para el panel "Data availability". Vacío → ese panel muestra `Not available in baseline tables`.
- `display_name` es lo único que se ve (nunca `variable_id`). Reglas de §4.3.
- Las **seis filas del ejemplo son las únicas verificadas** por las slides 15–19 (más `wbc`); el resto del catálogo (~170 plots → dominios, variables) se genera en la Fase 0 desde los outputs reales y se cura a mano. No inventar variables.

### 6.4 `objectives/objective_01/metadata/figures.csv`

```csv
figure_id,variable_id,view,kind,path,title,source_script,format,panel_groups,generated_run,status
```

- `view` ∈ `longitudinal` | `categorical_longitudinal` | `other:<slug>`.
- `kind` ∈ `longitudinal_numeric` | `categorical_stacked` | `swimmer` | `other`.
- `path` relativo a `SJD_OUTPUTS_ROOT`. `title` = el título de UI (editorial), **no** el texto dentro de la imagen.
- `panel_groups` (opcional, texto): `A: Class 1 history (no class 2) | B: Ever class 2 | C: Class 4 only` — copiado de la figura **solo si** lo confirma upstream (Q2).
- `status` ∈ `verified` | `candidate` | `missing`. Los `candidate` salen de `audit_outputs.py`; pasan a `verified` por revisión humana.

**Cómo se puebla (Fase 0, sin adivinar):**
1. `scripts/audit_outputs.py` recorre `SJD_OUTPUTS_ROOT/figures/**` (solo lectura) y escribe `audit/figures_inventory.csv` con: ruta relativa, extensión, tamaño, `mtime`, directorio productor.
2. Intento de mapeo automático **solo por nombre de archivo y por sidecar `.meta.json` si existe** (tokens del `variable_id` contra `variable_catalog.csv` y `view` por sufijo conocido). Lo no mapeado queda en `audit/figures_unmatched.csv`.
3. **Prohibido** inferir variable o vista por OCR, visión por computador o abriendo la imagen. Los no mapeados los cura el investigador.
4. Resultado: `figures.csv` con `status=candidate` + `OPEN_QUESTIONS.md` con las ambigüedades.

### 6.5 Contenido de cada pestaña

| Pestaña | Contenido | Fuente |
|---|---|---|
| Summary | Texto editorial corto de la variable (si existe) y, si hay `table1_variable`, la mediana (IQR) basal de `t11_overall` | `narrative.md` por `variable_id` + `t11_overall` |
| Data availability | `n_available_for_analysis` y `pct_available_for_analysis` de la variable; si procede, el `N` por Pop de `t11_by_pop` | `t11_avail` |
| Interpretation | Claim vinculado (`claims.yml`) con estado `reviewed`/`pending`; vacío si no hay claim (no se genera texto) | `claims.yml` |
| Source | archivo de figura, `source_script`, `run_key`, fecha, `commit` | `figures.csv` + `RunBadge` |

Las pestañas **no calculan** nada nuevo (§8). Una variable sin entrada en `t11_*` muestra `Not available in baseline tables`.

### 6.6 Rendimiento

- El explorer carga **solo `figures.csv` y `variable_catalog.csv`** al abrir; la imagen se carga al seleccionar. `st.cache_data` con `(run_key, path)`; `st.cache_resource` para el renderizado de PDF.
- Prefetch opcional de la figura siguiente/anterior (una sola, en segundo plano).
- Sin miniaturas en MVP. En fase 3: tira de miniaturas por dominio generadas con Pillow desde el PNG original, cacheadas y marcadas "thumbnail".

### 6.7 Estados y errores

| Caso | Comportamiento |
|---|---|
| Figura en `figures.csv` pero archivo ausente | `missing_output("Figure not found for this run", path)`; el resto de la página funciona |
| Variable sin figura para la vista elegida | opción deshabilitada con `No figure for this view` |
| Variable en catálogo con `active=false` | no aparece |
| Figura de un `run_key` distinto al de la tabla | no se muestra; aviso de inconsistencia |
| PDF no renderizable | descarga del original + mensaje, sin romper |

### 6.8 No-objetivos del explorer

No redibuja gráficos, no aplica filtros a los datos de la figura, no re-escala ejes, no cambia colores, no promedia entre figuras. Interactividad (hover/zoom en datos) solo si upstream entrega Plotly JSON (futuro, fase 4).

### 6.9 Casos dorados (tests)

Con los 5 ejemplos de §1.16 el test `test_explorer_golden.py` verifica: aparecen en el dominio correcto, su vista por defecto coincide con el catálogo (`anti_ro_ssa → categorical_longitudinal`, los otros → `longitudinal`), la figura se muestra sin modificar (hash del PNG servido = hash del archivo) y el enlace profundo funciona.

---

## 7. Convención para añadir nuevos estudios

### 7.1 Tipos

- **Objetivo** (`objectives/objective_NN/`): objetivos primarios/secundarios/exploratorios del protocolo.
- **Estudio** (`studies/<name>/`): `graph`, `pharma`, `economic`, futuros. Misma estructura interna que un objetivo.
- Nombre público de farmacología: **`pharma`** (no usar nombres de patrocinadores).

### 7.2 Procedimiento (checklist)

1. `python scripts/new_study.py --kind objective --id objective_02 --title "Disease Progression"` crea el esqueleto: `README.md`, `page.py`, `narrative.md`, `config.yml`, `sections/`, `metadata/{claims.yml, figures.csv, tables.csv}`. El script **no** genera contenido científico (solo plantillas vacías).
2. Añadir las salidas del nuevo estudio a `config/output_registry.yml` con IDs con prefijo de estudio (`o2_…`), rutas **verificadas** en el código del repo analítico (citar el script y la línea o la constante de `common.py`).
3. Ejecutar `scripts/audit_outputs.py --study objective_02` para el inventario de figuras/tablas reales.
4. Registrar la entrada en `config/navigation.yml` (§7.3) y en `config/studies.yml`.
5. Para cada slide/sección del estudio: rellenar `config.yml → sections[]` con `slides`, `eyebrow`, `title`, y la **tabla de procedencia por celda** (mismo formato que §1) en `objectives/<id>/README.md`.
6. Escribir `narrative.md` (solo texto; cifras con `{{ref:…}}`) y `claims.yml` (todo claim con `evidence_refs`).
7. Reutilizar `shared/components`. Si falta un componente, crearlo en `shared/` (no en el objetivo) y documentarlo en §3.
8. Añadir tests: `test_refs_<id>.py` (cada ref resuelve), `test_manifests.py` (outputs existen), humo de página.
9. Un objetivo nuevo **no** modifica secciones de otros objetivos ni `shared/` salvo para añadir componentes retro-compatibles.
10. Definition of done del estudio: §10.3.

### 7.3 `config/navigation.yml` (esquema)

```yaml
groups:
  - id: primary
    title: "Primary objectives"
    items:
      - {id: objective_01, title: "Objective 01 · Natural History / Baseline", path: objectives/objective_01, status: live}
      - {id: objective_02, title: "Objective 02 · Disease Progression",       path: objectives/objective_02, status: planned}
  - id: secondary
    title: "Secondary objectives"
    items: []
  - id: studies
    title: "Studies"
    items:
      - {id: graph,    title: "Graph / TDA", path: studies/graph,    status: planned}
      - {id: pharma,   title: "Pharma",      path: studies/pharma,   status: planned}
      - {id: economic, title: "Economic",    path: studies/economic, status: planned}
```

`status: planned` muestra una tarjeta "Coming soon" (sin contenido inventado). La navegación global usa `st.navigation`/`st.Page`; la clasificación final puede cambiarse sin tocar código.

### 7.4 Convención de nombres

- IDs: `objective_NN`, `secondary_NN_<slug>`, `exploratory_NN`; secciones `NN_<slug>.py`; claims `c.<section>.<slug>`; registro `t<step>_<slug>` o `<study>_<slug>`.
- Un test (`test_naming.py`) valida patrones y unicidad.

---

## 8. Regla inviolable: Codex no recalcula ni reinterpreta resultados científicos

### 8.1 Principio

```text
obj1_sjd (y repos de estudio)  = fuente de verdad científica
Dashboard                      = capa de presentación
```

El dashboard **solo consume** outputs producidos por `obj1_sjd`. Si un número, una tabla o una figura no existe como output, **no existe para el dashboard**.

### 8.2 Operaciones PERMITIDAS (lista cerrada)

| Código | Operación | Ejemplo |
|---|---|---|
| `SELECT` | seleccionar filas/columnas por clave | fila `essdai_total`, columna `Pop2` |
| `PARSE_CELL` | parsear el **texto de una celda** con los patrones de §5.4 | `49/62 (79.0%)` → `k=49`, `N=62`, `pct=79.0` |
| `FORMAT` | redondear/formatear **solo para mostrar** (`ROUND_HALF_UP`), separar miles, añadir unidades de `labels.yml` | `1996.5 → 1997` (IgG) |
| `ORDER` | ordenar/agrupar filas según `config` editorial | orden de razas |
| `LAYOUT` | traducir un valor publicado a una dimensión visual (ancho de barra = `pct` publicado) | barra 70.4 % |
| `COUNT_METADATA_ROWS` | contar filas de tablas de metadatos (`11_variable_dictionary.csv`) | `26 / 952` hasta que upstream publique el summary |
| `COMPARE_PUBLISHED_EQUALITY` | validaciones de consistencia **entre valores ya publicados** (igualdad, constancia) y señalizar | `n(sf36_pcs) == n(sf36_mcs)`; `denominator` constante |
| `HASH_EVIDENCE` | hash de valores `raw` para el control de revisión de claims (§5.5) | `reviewed_hash` |

### 8.3 Operaciones PROHIBIDAS

1. Calcular o **derivar** cualquier estadístico: medias, medianas, cuantiles, desviaciones, proporciones, razones, diferencias, tasas, IC, p-values, q-values, FDR, tamaños de efecto, regresiones, modelos, supervivencia, Kaplan–Meier, transiciones, clustering, correlaciones, bootstrap.
2. **Calcular un porcentaje** a partir de dos conteos si la fuente no lo publica (p. ej. `62/159`). Mostrar `k/N` y el `pct` publicado; si no hay `pct`, mostrar solo `k/N`.
3. Sumar, restar, normalizar o reescalar valores publicados (p. ej. forzar que una barra apilada sume 100).
4. Cruzar tablas de resultados (`merge`/`join`) para generar un valor nuevo.
5. **Redefinir** cohortes, baseline, episodios clínicos, poblaciones (Pop), umbrales (ESSDAI ≥ 5, ESSPRI ≥ 5), criterios de overlap, ventanas, denominadores o endpoints. Las definiciones se **citan** del código/slides, no se reimplementan.
6. Clasificar pacientes o aplicar umbrales clínicos propios. (Las bandas de color de la slide 7 son convención visual declarada, no clasificación.)
7. Interpretar: escribir conclusiones, "hallazgos", causalidad o recomendaciones clínicas nuevas; resumir con LLM; generar texto a partir de los números. Los párrafos interpretativos son **editoriales y humanos** (`claims.yml`).
8. Imputar, interpolar, suavizar, "completar" huecos o reutilizar valores de otra ejecución.
9. Inferir semántica de una figura mirándola (OCR, visión por computador) o de su nombre cuando haya ambigüedad.
10. Generar datos de ejemplo/mocks que se confundan con resultados. Los mocks viven solo en `tests/fixtures/` etiquetados `MOCK`; `--demo` muestra un banner permanente `DEMO DATA — NOT RESULTS`.
11. Modificar, mover o borrar archivos de `obj1_sjd`; modificar la lógica de sus scripts; reejecutar sus scripts.
12. Importar en `pages/`, `objectives/`, `studies/`, `shared/components/`: `scipy`, `statsmodels`, `sklearn`, `lifelines`, `pingouin`, `numpy.random`, y llamar a `.mean()`, `.median()`, `.std()`, `.quantile()`, `.corr()`, `.groupby()`, `.agg()`, `.rolling()`, `.merge()` sobre datos de resultados.
13. Leer datos de nivel paciente (§5.7).

### 8.4 Qué hacer cuando falta algo

| Situación | Acción |
|---|---|
| Archivo/figura no existe | `missing_output`; añadir a `OPEN_QUESTIONS.md`; **no** sustituir |
| Fila o columna ausente | `SchemaError` legible; añadir a `OPEN_QUESTIONS.md` |
| Celda no parsea | mostrar `raw` + warning; añadir a `OPEN_QUESTIONS.md` |
| Definición de un término no está en el código ni en este documento (p. ej. `Class 4 only`) | mostrar tal cual aparece en la figura; **no definir**; añadir a `OPEN_QUESTIONS.md` |
| Dos fuentes discrepan (p. ej. `n=162` vs 159) | mostrar ambas con su procedencia y aviso; **no elegir** |
| Se necesita un número que ningún output publica | pedirlo a upstream (§5.8); no calcularlo |

### 8.5 `OPEN_QUESTIONS.md`

Formato: `ID · fecha · sección/slide · qué falta · dónde se buscó · propuesta (upstream)`. Codex lo actualiza y **no continúa inventando** sobre ese punto. Semillas en §11.

### 8.6 Enforcement automático (obligatorio en CI/local)

- `tests/test_no_science.py`: AST sobre `pages/`, `objectives/`, `studies/`, `shared/components/` que falla ante imports/llamadas prohibidas de §8.3-12. Excepción única: `shared/utils/formatting.py` (`Decimal` para `FORMAT`).
- `tests/test_no_hardcoded_numbers.py` (§5.5).
- `tests/test_registry.py` (rutas registradas, sin rutas fuera del registro, sin rutas prohibidas).
- `tests/test_refs_objective_01.py`: cada ref de §1 resuelve; con las fixtures de la exportación 2026-10-03 los valores coinciden con los de las tablas de §1.
- `tests/test_labels_visible.py`: ningún identificador interno de `labels.yml` aparece como texto en la UI.
- Un cambio que rompa uno de estos tests **no se mergea**.

### 8.7 Contenido de `AGENTS.md` (copiar tal cual a la raíz del repo del dashboard)

```markdown
# AGENTS.md — SjD Research Explorer

You build a presentation layer. You are NOT the owner of any scientific result.

## Hard rules
1. Only consume outputs produced by obj1_sjd (and other registered study repos). Never compute, derive, estimate, impute, normalize, or re-interpret scientific results.
2. Every number on screen resolves through a registered ref (config/output_registry.yml + shared/loaders/refs.py). No hard-coded result numbers in .py, .md or .yml.
3. Allowed operations are only: SELECT, PARSE_CELL, FORMAT (display rounding, ROUND_HALF_UP), ORDER, LAYOUT, COUNT_METADATA_ROWS, COMPARE_PUBLISHED_EQUALITY, HASH_EVIDENCE (see DASHBOARD_SPEC.md §8.2).
4. Do not redefine cohorts, baseline, Pop thresholds, overlap rules or denominators. Quote the source; do not reimplement it.
5. Never read patient-level files (DASHBOARD_SPEC.md §5.7). Never write inside obj1_sjd. Never re-run its scripts.
6. If something is missing or ambiguous: render MissingOutput, append to OPEN_QUESTIONS.md, and STOP on that point. Do not guess, mock, or fill.
7. Interpretive text is editorial (claims.yml) and human-reviewed. Never generate conclusions from numbers.
8. UI language is English. Never show internal variable ids (e.g. sf36_mcs); use labels from config/labels.yml.
9. Scientific colors (Pop groups, figure panels) are fixed; do not recolor figures.
10. Tests in tests/ must pass before any change is considered done.
```

---

## 9. Procedencia visible en la UI (provenance)

Cada bloque muestra al pie (Courier New 16 px, `--color-text-muted`) una línea generada por `evidence_footer`:

```text
Source: 11_table1_by_pop.csv · Generated by: 11_integrated_baseline_characterization.py · Run: 2026-10-03 · a1b2c3d
```

- `Source`: archivos de las refs usadas por el bloque (nombres de archivo, sin ruta); si hay varios, separados por ` · `.
- `Generated by`: `producer` del registro (nombre del script). **No** reutilizar textos de pie del PPTX (H4).
- `Run`: `prov11.run_date` y `git_commit` (Modo B) o `run_id` (Modo A).
- Extras en el pie de la slide, cuando aplique, se generan desde `labels.yml` (`ESR = erythrocyte sedimentation rate`; `IgG rounded to integer`).
- Modo inspect (`?inspect=1`): al pasar el ratón sobre una cifra, tooltip `file · Section|Variable · column · line_hint`.
- Cadena de trazabilidad por claim: `Objective → Section → Claim → Output → Script → Run` (campos de `claims.yml`).

---

## 10. Plan de implementación, DoD y pruebas

### 10.1 Fases

**Fase 0 — Auditoría y andamiaje (no UI definitiva)**
1. Crear repo `sjd_research_explorer/` con el árbol de §4.1 vacío, `AGENTS.md`, `OPEN_QUESTIONS.md` (con las semillas de §11), `pyproject.toml` (Python ≥ 3.10; `streamlit>=1.36`, `pandas`, `pyyaml`, `pillow`, `pymupdf`, `plotly`; dev: `pytest`).
2. Implementar `registry.py`, `csv_loader.py`, `parsers.py`, `refs.py` y sus tests con las **fixtures** (§10.4).
3. Implementar `scripts/audit_outputs.py` y producir `audit/*.csv` **si** hay acceso a `SJD_OUTPUTS_ROOT`; si no, dejar el script probado con un árbol de prueba.
4. **Entregable:** `test_refs_objective_01.py` verde para todas las refs de §1; `variable_catalog.csv` y `figures.csv` iniciales (`candidate`).

**Fase 1 — MVP Objective 1 (lectura)**
Implementar `shared/components` y las 8 secciones de §1.2 con las 19 slides migradas (las slides 15–19 como ejemplos del explorer). Sin modo presentación todavía.

**Fase 2 — Datos vivos y variable explorer completo**
`RunBadge`, Refresh, claims con revisión, explorer con catálogo completo, coverage matrix, enlaces profundos.

**Fase 3 — Presentación y pulido**
Modo presentación (`?mode=present`, una sección por pantalla), atajos de teclado (componente JS mínimo), miniaturas, modo Compare, prueba en proyector.

**Fase 4 — Escalado**
Estabilizar `shared/` y añadir `objective_02` con mínimo diseño nuevo (§7). Plotly solo donde upstream entregue datos/JSON.

### 10.2 Presentación vs lectura

- `?mode=read` (por defecto): secciones apiladas con `MethodsDetails` colapsables y `Evidence`.
- `?mode=present`: una sección por vista, sin sidebar, tipografía ≥ 20 px, sin controles salvo prev/next; `Caution` y pie siempre visibles.

### 10.3 Definition of done (por sección y por objetivo)

Una sección está terminada cuando:
1. Cada elemento de su tabla de §1 resuelve a su celda (test verde) y el valor mostrado coincide con el esperado en las fixtures.
2. No hay cifras literales fuera de `{{ref}}`/`labels.yml`.
3. El pie de procedencia muestra archivo, script y run.
4. Falta de datos degrada con `missing_output` sin romper la página.
5. Comparación visual con la slide original: títulos, denominadores, unidades, colores de significado científico, leyendas y `Caution` equivalentes (se acepta diferencia de maquetación).
6. Pasa `test_no_science.py` y `test_labels_visible.py`.

El objetivo está terminado cuando: app local con un comando documentado; una nueva ejecución de `obj1_sjd` con outputs válidos actualiza cifras y figuras sin editar PowerPoint; no hay mezcla silenciosa de ejecuciones; se puede añadir una pregunta/figura sin tocar `shared/`; sirve para presentar al PI y para inspección metodológica.

### 10.4 Fixtures

El investigador aportará los 8 archivos de la exportación 2026-10-03 (`11_table1_overall.csv`, `11_table1_by_pop.csv`, `11_baseline_variable_availability.csv`, `11_variable_dictionary.csv`, `11_baseline_full_variable_inventory.csv`, `12_followup_summary.csv`, `12_followup_summary_long.csv`, `12_retention.csv`) para `tests/fixtures/objective_01/`. Son tablas **agregadas** sin identificadores. Si Codex no los tiene, **no los reconstruye**: pide los archivos. Los valores esperados para las pruebas son los de las tablas de §1.

### 10.5 Ejecución

```bash
export SJD_OUTPUTS_ROOT=/path/to/obj1_sjd/outputs
export SJD_REPO_ROOT=/path/to/obj1_sjd
pip install -e ".[dev]"
streamlit run app.py
pytest -q
```

---

## 11. Preguntas abiertas (semillas de `OPEN_QUESTIONS.md`)

| ID | Pregunta | Quién | Impacto |
|---|---|---|---|
| Q1 | La imagen del swimmer del PPTX (slide 4) tiene `n=162`, "Phase 1: Initial Full Evaluation" y eje en días. ¿Qué script/ejecución la produjo? El código actual dibuja un swimmer en años con N basal 159. ¿Se reemplaza por el output actual? | investigador | slide 4 |
| Q2 | Semántica de los paneles de laboratorio (`Class 1 history (no class 2)`, `Ever class 2`, `Class 4 only`) y de `Patient later transitioned to class 4`: ¿definición y origen (`SJOGRENS_CLASS`)? ¿Se publicarán N por panel en un sidecar JSON? | investigador / upstream | explorer, slides 15–19 |
| Q3 | ¿Escribir celdas vacías como `0/N (0.0%)` en `11_table1_by_pop.csv`? | upstream | H6, slide 12 |
| Q4 | ¿Publicar `summary.json` con `n_patients`, `n_clinical_episodes`, `n_variables_total`, `n_variables_table1`? | upstream | slide 5 |
| Q5 | ¿Publicar IgG y otras variables con más decimales o redondeo definido? | upstream | slide 11 |
| Q6 | ¿Las figuras con puntos individuales (swimmer, laboratorios) pueden mostrarse fuera de un entorno local/institucional? | investigador | despliegue |
| Q7 | La slide 2 se titula "diagnosis-anchored view of overlap" pero define baseline/seguimiento. ¿Mantener? | investigador | título |
| Q8 | ¿El script generador de figuras de laboratorio vive en otra rama o en el repo `eda_sjd`? ¿Nombres de archivo/convención? | investigador | `figures.csv` (H2) |
| Q9 | ¿Se reincorporan las slides retiradas (definiciones de Pop, limitaciones) como `MethodsDetails`? | investigador | §1.17 |
| Q10 | Step 12 sin `provenance.json` ni `run_manifest.json`: ¿se añadirá? | upstream | `RunBadge` |
| Q11 | Seguimiento por protocolo (11D/15D): el deck anterior lo mostraba y las exportaciones nuevas son solo `Overall`. ¿Se re-exportará con `Cohort` = `11D`/`15D`? | upstream | slides 13–14 |

---

## Anexo A — Valores de referencia de la exportación 2026-10-03 (para fixtures y tests)

Usar **solo** para pruebas contra las fixtures; nunca como datos de la UI.

- `t11_overall`: `N patients` = 159; `essdai_total` = `2.0 (0.0–4.0); n=148`; `esspri_total` = `5.3 (3.7–7.0); n=117`; `sex: FEMALE` = `151/159 (95.0%)`; `Anti-Ro/SSA positive, n/N (%)` = `49/62 (79.0%)`; `overlap_status: overlap` = `75/159 (47.2%)`.
- `t11_by_pop` columnas `Pop1/Pop2/Pop3/Unclassifiable`: `N patients` = 32/54/38/35; `essdai_total` = `7.5 (6.0–13.2); n=32` / `1.0 (0.0–2.0); n=54` / `1.0 (0.0–2.0); n=38` / `1.0 (0.0–2.2); n=24`; `esspri_total` en `Unclassifiable` = `NA`; `overlap_status: glandular_only` en `Pop1` = *(vacío)*; `igg__value` en `Pop1` = `1562.8 (1219.6–1996.5); n=32`.
- `t11_avail`: `essdai_total` → `n_available_for_analysis=148`, `pct_available_for_analysis≈93.08`; `anti_ro_ssa__ever_positive_through_episode` → `62`, `≈38.99`.
- `t12_long` (Cohort=Overall): `Clinical episodes`=`497`; `Unique patients`=`159`; `Follow-up, median (IQR), years`=`4.0 (1.9–6.1)`; `Median inter-visit gap, days`=`707.0`; `IQR inter-visit gap, days`=`394.2–765.0`; `Patients with >=3 clinical episodes`=`95 (59.7%)`; `Patients with at least one gap >730 days`=`86 (54.1%)`.
- `t12_ret`: `6 months,182,153,159,96.2` … `10 years,3652,6,159,3.8`.

## Anexo B — Mapeo editorial de eyebrows (del deck)

`UNIT OF ANALYSIS` → `01_study_design`; `STARTING POINT` → `02_cohort`; `BASELINE` → `03_baseline`; `BASELINE PHENOTYPE` / `BASELINE PHENOTYPE BY POPULATION` → `04_phenotype` y `05_glandular_serology`; `ORGAN INVOLVEMENT` → `06_organ_involvement`; `LONGITUDINAL FOLLOW-UP` → `07_followup`. El eyebrow de cada sección vive en `config.yml`, no en el código.

---

## Anexo C — Inventario de archivos y rutas (de dónde sale cada fuente)

Repositorio: <https://github.com/dasalazarb/obj1_sjd> · rama `main` · commit auditado `1dd1662` (2026-10-02). Los enlaces `blob/main/…` apuntan al código; **los outputs (CSV/PDF/JSON) no están en GitHub**: en el repo solo existen `outputs/{tables,figures,logs}/.gitkeep`. Los outputs viven en la máquina donde se ejecutó el pipeline (por defecto Biowulf, ver C.1) y el dashboard los lee desde `SJD_OUTPUTS_ROOT`.

Prefijo de URL de código: `https://github.com/dasalazarb/obj1_sjd/blob/main/` (en la columna "URL" solo se escribe el sufijo).

### C.1 Raíces de directorios (definidas en `common.py`)

| Raíz | Definición en el código | Valor por defecto | Variable en el dashboard |
|---|---|---|---|
| Repo | `PROJECT_ROOT = Path(__file__).resolve().parent` (`common.py`) | carpeta donde está clonado `obj1_sjd` | `SJD_REPO_ROOT` |
| Outputs | `OUTPUTS_DIR = PROJECT_ROOT / "outputs"` | `<repo>/outputs` | `SJD_OUTPUTS_ROOT` |
| Datos | `DATA_DIR = PROJECT_ROOT / "data"` (`raw/`, `intermediate/`, `analytic/`) | `<repo>/data` | derivada de `SJD_REPO_ROOT` |
| Upstream | `EDA_SJD_ROOT` (env var) | `/data/salazarda/data/eda_sjd` | **no la usa el dashboard** (solo documenta origen del spine) |
| Metadatos | `METADATA_DIR = PROJECT_ROOT / "metadata"` | `<repo>/metadata` (no versionada) | no la usa el dashboard |

Convención de salida por script (`common.script_output_dir`): `outputs/tables/blockA/<script>/…`, `outputs/qc/blockA/<script>/…`, `outputs/figures/blockA/<script>/…`; los scripts `00_*` escriben en la raíz. Estudios: `outputs/{tables,figures,qc,logs}/studies/<study>/` (`src/studies/_shared.py::create_study_dirs`).

### C.2 Outputs que el dashboard **consume** (ruta completa)

Todas las rutas son relativas a `SJD_OUTPUTS_ROOT` salvo `prov11` (relativa a `SJD_REPO_ROOT`).

| Registry id | Ruta relativa | Productor (URL de código) | Slides | Estado |
|---|---|---|---|---|
| `t11_overall` | `tables/blockA/11_integrated_baseline_characterization/11_table1_overall.csv` | `src/block_A/11_integrated_baseline_characterization.py` (L504) | 5, 6, 8, 10, 12 | recibido (`11_table1_overall-4.csv`) |
| `t11_by_pop` | `tables/blockA/11_integrated_baseline_characterization/11_table1_by_pop.csv` | idem (L505) | 6, 9, 10, 11, 12 | recibido (`11_table1_by_pop-4.csv`) |
| `t11_avail` | `tables/blockA/11_integrated_baseline_characterization/11_baseline_variable_availability.csv` | idem (L506) | 7 | recibido (`11_baseline_variable_availability-4.csv`) |
| `t11_dict` | `tables/blockA/11_integrated_baseline_characterization/11_variable_dictionary.csv` | idem (L527) | 5 (recuento `26 / 952`) | recibido (`11_variable_dictionary-4.csv`) |
| `t11_inventory` (redundante) | `tables/blockA/11_integrated_baseline_characterization/11_baseline_full_variable_inventory.csv` | idem (L507, copia de `t11_avail`) | — | recibido (`…inventory-3.csv`); **no usar** |
| `t12_long` | `tables/blockA/12_followup_characterization/12_followup_summary_long.csv` | `src/block_A/12_followup_characterization.py` (L334) | 5, 13 | recibido (`12_followup_summary_long-2.csv`) |
| `t12_ret` | `tables/blockA/12_followup_characterization/12_retention.csv` | idem (L335) | 14 | recibido (`12_retention-2.csv`) |
| `t12_wide` (redundante) | `tables/blockA/12_followup_characterization/12_followup_summary.csv` | idem (L333) | — | recibido (`12_followup_summary-2.csv`); **no usar** |
| `prov11` | `data/analytic/blockA/11_integrated_baseline_characterization/11_integrated_baseline_patient_level.provenance.json` | idem (L528, bloque `provenance`) | pie de procedencia | **no recibido**: pedir al investigador (C.7) |
| `fig_swimmer` | `figures/blockA/01_pop_distribution/02_pop_distribution_plot.pdf` (+ `…_pop1.pdf`, `…_pop2.pdf`, `…_pop3.pdf`, `…_unclassifiable.pdf`) | `src/block_A/01_pop_distribution.py` (L581, `fig.savefig(panel_path…)`) | 4 | **no recibido**; la imagen del PPT (n=162) está desactualizada → Q1 |
| `fig_lab_*` (~170) | **sin ruta verificada** (ver C.6) | **generador no encontrado en `main` ni en ramas `codex/*`** | 15–19 | pendiente: `figures.csv` curado por el investigador (Q8) |

Prohibidos (nivel paciente, §5.7) que viven junto a los anteriores y **nunca** se leen: `data/analytic/blockA/11_integrated_baseline_characterization/11_integrated_baseline_patient_level.{parquet,csv}`, `outputs/qc/blockA/11_integrated_baseline_characterization/11_baseline_patient_audit.csv`, `outputs/qc/blockA/12_followup_characterization/12_followup_episode_audit.csv`.

QC agregados que *podrían* mostrarse en el futuro (existen por código, sin contenido auditado): `outputs/qc/blockA/11_integrated_baseline_characterization/{11_baseline_structural_qc,11_baseline_variable_qc,11_baseline_missingness_qc,11_baseline_possible_duplicate_concepts,11_baseline_range_violations,baseline_refactor_regression_comparison}.csv` y `outputs/qc/blockA/12_followup_characterization/{12_followup_qc,12_zero_day_gap_audit}.csv`. También `tables/blockA/11_integrated_baseline_characterization/11_table1.xlsx` (equivalente en Excel; no consumir).

### C.3 Cadena de entrada del pipeline (para el pie de procedencia; el dashboard **no** lee estos archivos)

| Paso | Script (URL) | Entrada → Salida (rutas en `common.py`) |
|---|---|---|
| Spine clínico (upstream) | `src/00_input data.py` (URL con `%20`: `src/00_input%20data.py`) | `EDA_SJD_ROOT/data_analytic/clinical_episode_spine_sjd.parquet` → `data/raw/clinical_episode_spine_sjd.parquet` (+ `…_11D_15D.parquet`, `.provenance.json`, `list_ids_longitudinal.csv`) |
| 00 | `src/00_build_visit_spine.py` | `data/raw/…` → `data/intermediate/00_patient_visit_spine.parquet`, `00_episode_spine_all.parquet`, `00_clinical_visit_spine.parquet` |
| 01 Pop | `src/block_A/01_pop_distribution.py` | → `data/intermediate/01_pop_distribution/01_visit_level_classification.parquet`; figuras en `outputs/figures/blockA/01_pop_distribution/` |
| 01 Serología | `src/block_A/01_serological_profile.py` | → `data/intermediate/block_A/01_serological_profile/01_labs_episode_wide.parquet`; tabla `tables/blockA/01_serological_profile/01_labs_episode_coverage.csv` |
| 01 Fenotipo ext. | `src/block_A/01_extended_clinical_phenotype.py` | → `data/intermediate/block_A/01_extended_clinical_phenotype/` |
| 06 Overlap | `src/block_A/06_overlap_glandular.py` + `src/derivations/overlap_flags.py` | → `data/intermediate/block_A/06_overlap_glandular/06_overlap_episode_level.parquet` |
| 09 PROs | `src/block_A/09_pros_longitudinal.py` + `src/derivations/pro_scoring.py` | → `data/intermediate/09_pros_longitudinal/09_pros_episode_level.parquet` |
| **10 Integrado** | `src/block_A/10_build_integrated_longitudinal_dataset.py` + `src/integrated_schema.py` | → `data/analytic/10_build_integrated_longitudinal_dataset/10_integrated_longitudinal_clinical_episode.parquet` (`common.INTEGRATED_LONGITUDINAL_PARQUET`) |
| **11 Baseline** | `src/block_A/11_integrated_baseline_characterization.py` | entrada: el parquet del paso 10 → salidas de C.2 (`t11_*`, `prov11`) |
| **12 Seguimiento** | `src/block_A/12_followup_characterization.py` | entrada: el mismo parquet del paso 10 → salidas de C.2 (`t12_*`); escribe además en `data/analytic/blockA/12_followup_characterization/` (**sin** `provenance.json`, ver H12) |

Contrato de esquema del paso 10: `docs/step10_public_schema_v2.md`. Reglas de tiempo/missingness: `docs/clinical_missingness_and_hla_time_contract.md`. Mapa de la historia analítica: `story_map.md`.

### C.4 Código y documentación de referencia (versionados)

| Ruta en el repo | URL (sufijo) | Para qué sirve en el dashboard |
|---|---|---|
| `common.py` | `common.py` | fuente de **todas** las rutas (C.1) |
| `config.py` | `config.py` | parámetros del pipeline (solo lectura de contexto) |
| `src/block_A/11_integrated_baseline_characterization.py` | idem | formato de celdas de Table 1 (`median (q1–q3); n=N`, `k/N (pct%)`), esquema de `provenance` |
| `src/block_A/12_followup_characterization.py` | idem | definición de cada `Indicator` de `t12_long` y de `12_retention` |
| `src/block_A/01_pop_distribution.py` | idem | reglas Pop1/Pop2/Pop3 (umbrales ESSDAI 5 / ESSPRI 5), `POP_COLORS`, nombres de figuras |
| `src/derivations/overlap_flags.py` | idem | definición de overlap (glandular activo + ≥1 dominio extraglandular activo) |
| `src/derivations/pro_scoring.py` | idem | puntuación de PROs (SF-36, FACIT-F…) |
| `src/integrated_schema.py` | idem | columnas del dataset integrado |
| `docs/step10_public_schema_v2.md` | idem | contrato de columnas del paso 10 |
| `docs/clinical_missingness_and_hla_time_contract.md` | idem | semántica de ausentes |
| `story_map.md` | idem | narrativa de pasos 0–12 |
| `tests/test_integrated_baseline_characterization.py`, `tests/test_followup_characterization.py` | idem | casos de prueba de los parsers y formatos (útiles para las fixtures del dashboard) |
| `data/raw/graph_s4_lab_group_map.csv` | idem | mapa de grupos de laboratorio del estudio *graph* (no es el de las figuras de las slides 15–19; ver Q2) |
| `src/studies/{graph,pharma}/…` | idem | estudios futuros (§7); sus outputs irán en `outputs/{tables,figures,qc}/studies/<study>/` |

### C.5 Archivos que aportaste (nombre en el dashboard vs. nombre recibido)

| Nombre real (registry) | Nombre recibido | Uso |
|---|---|---|
| `11_table1_overall.csv` | `11_table1_overall-4.csv` | fixture `tests/fixtures/objective_01/` |
| `11_table1_by_pop.csv` | `11_table1_by_pop-4.csv` | fixture |
| `11_baseline_variable_availability.csv` | `11_baseline_variable_availability-4.csv` | fixture |
| `11_variable_dictionary.csv` | `11_variable_dictionary-4.csv` | fixture |
| `11_baseline_full_variable_inventory.csv` | `11_baseline_full_variable_inventory-3.csv` | no usar (redundante) |
| `12_followup_summary_long.csv` | `12_followup_summary_long-2.csv` | fixture |
| `12_followup_summary.csv` | `12_followup_summary-2.csv` | no usar (redundante) |
| `12_retention.csv` | `12_retention-2.csv` | fixture |
| `codebook_final_harmonized_once_quince.xlsx` | (archivo del Project "autoinmune") | referencia de nombres/etiquetas clínicas; **no** es un output consumido. El repo apunta a otro: `DEFAULT_CODEBOOK = metadata/Consolidated_Codebook_all_columns.xlsx` (no versionado) |
| `pres_po1.pptx` | `pres_po1.pptx` (19 slides) | especificación visual y de contenido (este documento) |

Los sufijos `-2/-3/-4` los añade la descarga; en el dashboard el nombre es el de la columna "Nombre real".

### C.6 Imágenes incrustadas en `pres_po1.pptx` (qué archivo es cada una)

| Slide | Media en el PPTX | Qué es | Fuente / ruta |
|---|---|---|---|
| 1 | `image1.png` (5.7 KB) | icono decorativo de portada | diseño; reproducir con CSS/SVG, no importar |
| 2 | `image2–5.png` (2–4 KB) | iconos del diagrama conceptual | diseño; SVG propios |
| 4 | `image6.png` (420 KB) | swimmer plot (n=162, **desactualizado**) | `outputs/figures/blockA/01_pop_distribution/02_pop_distribution_plot.pdf` (n=159 esperado) → Q1 |
| 5 | `image7.png` (0.7 KB) | icono | diseño |
| 15 | `image8.png` (141 KB) | `anti_ro_ssa: categorical laboratory profile` | **ruta no verificada** (generador ausente); menciona `01_labs_categorical_summary_by_visit.csv` |
| 16 | `image9.png` (279 KB) | `complement_c4: longitudinal laboratory profile` | idem |
| 17 | `image10.png` (261 KB) | `igg: longitudinal laboratory profile` | idem |
| 18 | `image11.png` (224 KB) | `rheumatoid_factor: longitudinal laboratory profile` | idem |
| 19 | `image12.png` (253 KB) | `wbc: longitudinal laboratory profile` | idem |

### C.7 Lo que falta para cerrar las rutas (pedir al investigador antes de la Fase 3)

1. **Ruta real de las ~170 figuras de laboratorio** y del script que las genera. No existe en `main` ni en las ramas `codex/*` (se buscó `laboratory profile`, `categorical laboratory`, `Observed group mean`, `labs_categorical_summary_by_visit`); el único hit de `SJOGRENS_CLASS` es `src/00_input data.py`. Hasta entonces el explorer lee solo `metadata/figures.csv` (§6.4).
2. **`01_labs_categorical_summary_by_visit.csv` y tablas hermanas** (citadas dentro de la figura de la slide 15): ruta y esquema.
3. **`11_integrated_baseline_patient_level.provenance.json`** de la corrida que produjo los CSV recibidos.
4. **PDF del swimmer plot** regenerado (n=159) y sus cuatro paneles por Pop.
5. **Un `provenance.json` para el paso 12** (propuesto en §5.8).
6. Confirmar la **raíz de outputs de producción** (¿Biowulf `/data/salazarda/...` o una copia local?) para fijar `SJD_OUTPUTS_ROOT` en el README del dashboard.

> Regla para Codex: si una ruta de este anexo no existe en `SJD_OUTPUTS_ROOT`, la sección muestra `MissingOutput` con la ruta esperada (§8.4); **no** se busca un archivo "parecido" ni se reconstruye el valor.
