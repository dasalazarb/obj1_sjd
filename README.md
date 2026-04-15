# Objetivo Primario 2 — Progresión de ESSDAI en el tiempo

---

## Contexto del análisis

**Objetivo:** Describir la progresión de la enfermedad basándose en el cambio
en ESSDAI a lo largo del tiempo en la cohorte general de SjD.

**Restricciones binding documentadas (informe_viabilidad_sjogren.docx):**
- Cohorte analítica: **C2** — pacientes con ≥2 ESSDAI evaluables (n=71)
- Subconjunto con ≥3 mediciones: n=36 (para sensibilidad con random slopes)
- Variable outcome: `essdai-_r__essdai_total_score` (versión recodificada, 83.5% cobertura)
- La familia `essdai__*` (sin el sufijo `-_r_`) tiene solo 40.5% cobertura → EXCLUIDA
- Análisis NO ejecutables a este n: LCMM, growth-mixture models, HMM, multi-state continuo
- Los intervalos son discretos (v1–v6), no fechas continuas en todos los casos

---

## Estructura del proyecto

```
obj2_essdai_progression/
├── README.md                ← este archivo
├── config.py                ← rutas, constantes, encodings ESSDAI
├── 01_build_cohort_c2.py    ← carga de datos y construcción de la cohorte C2
├── 02_eda_descriptive.py    ← EDA, cobertura, spaghetti plots, descriptivos
├── 03_lmm_primary.py        ← LMM random intercept (análisis primario)
├── 04_domain_analysis.py    ← análisis por dominio ESSDAI (GLMM / descriptivo)
├── 05_severity_strata.py    ← clasificación Pop 1-3, transiciones Sankey
├── 06_output_tables.py      ← tablas listas para manuscrito
└── outputs/                 ← figuras (.png/.svg) y tablas (.csv/.xlsx)
```

---

## Orden de ejecución

```bash
python 01_build_cohort_c2.py    # genera cohort_c2.parquet
python 02_eda_descriptive.py    # genera figuras EDA
python 03_lmm_primary.py        # genera tabla LMM + figura de trayectorias
python 04_domain_analysis.py    # genera tabla de dominios + heatmap
python 05_severity_strata.py    # genera tabla Pop 1-3 + Sankey
python 06_output_tables.py      # consolida tablas para manuscrito
```

---

## Variables centrales (nombres exactos en el dataset)

| Variable | Descripción | Tipo | Cobertura (n≥1 / n≥2) |
|---|---|---|---|
| `essdai-_r__essdai_total_score` | ESSDAI total recodificado | numeric | 132 / 71 |
| `essdai-_r__articular_domain` | Dominio articular | categorical | 132 / 71 |
| `essdai-_r__biological_domain` | Dominio biológico | categorical | 132 / 71 |
| `essdai-_r__hematologic` | Dominio hematológico | categorical | 132 / 71 |
| `essdai-_r__constitutional` | Dominio constitucional | categorical | 132 / 71 |
| `essdai-_r__gland_swell` | Inflamación glandular | boolean | 132 / 71 |
| `essdai-_r__cutaneous` | Dominio cutáneo | boolean | 132 / 71 |
| `essdai-_r__cns` | Sistema nervioso central | boolean | 132 / 71 |
| `essdai-_r__muscular_domain` | Dominio muscular | boolean | 132 / 71 |
| `essdai-_r__neuro_peripheral` | Neuropatía periférica | categorical | 132 / 71 |
| `essdai-_r__pulmonary` | Dominio pulmonar | categorical | 132 / 71 |
| `essdai-_r__renal` | Dominio renal | boolean | 132 / 71 |
| `essdai-_r__hema_lphdenopthy` | Linfadenopatía | categorical | 132 / 71 |

**Identificadores y tiempo:**
- `ids__patient_record_number` → ID paciente
- `ids__interval_name` → intervalo de visita (v1–v6)
- `ids__visit_date` → fecha real de visita (usar cuando disponible)
- `ids__age_at_visit` → edad en la visita
- `ids__sex`, `ids__race`, `ids__ethnicity` → demografía
- `visit_summary_form__sjogrens_class` → clase SjD (primary/secondary)
- `sjogren's_syndrome_history__sjogrens_dx_date` → fecha de diagnóstico

**ESSPRI (para clasificación Pop 1-3):**
- `esspri_questionnaire__dryness`, `__fatigue`, `__pain` → ítems ESSPRI centrales
