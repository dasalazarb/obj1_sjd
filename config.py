"""
config.py — Configuración global para Objetivo Primario 2
==========================================================
Centraliza rutas, constantes clínicas, encodings de dominios ESSDAI,
y umbrales de cohorte. Todos los scripts del proyecto importan desde aquí.

NOTA PARA CODEX/CLAUDE CODE:
- No modificar los encodings de dominio sin validación clínica del investigador.
- Los intervalos INTERVAL_ORDER son ordinales, no necesariamente equidistantes en tiempo.
  Usar ids__visit_date cuando esté disponible; caer en intervalo ordinal solo si falta.
- El threshold ESSDAI_SEVERE = 5 es el definido en el protocolo (uno.docx §7.1).
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────

# Ajustar DATA_DIR a la ubicación real del dataset en Biowulf
DATA_DIR = Path("data/raw")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivo fuente principal (formato wide: paciente × intervalo)
VISITS_FILE = DATA_DIR / "visits_long_collapsed_by_interval_codebook_type_recode-3.csv"

# Variable-summary (para triaje de variables, no para análisis principal)
VAR_SUMMARY_FILE = DATA_DIR / "longitudinal_variable_summary-12.csv"

# Cohort C2 serializada (generada por 01_build_cohort_c2.py)
COHORT_C2_FILE = OUT_DIR / "cohort_c2.parquet"

# ── Identificadores ────────────────────────────────────────────────────────────

PATIENT_ID   = "ids__patient_record_number"
INTERVAL_COL = "ids__interval_name"
VISIT_DATE   = "ids__visit_date"
AGE_COL      = "ids__age_at_visit"
SEX_COL      = "ids__sex"
RACE_COL     = "ids__race"
ETHNICITY_COL = "ids__ethnicity"
SJOGRENS_CLASS = "visit_summary_form__sjogrens_class"
DX_DATE_COL  = "sjogren's_syndrome_history__sjogrens_dx_date"

# ── Orden canónico de intervalos ───────────────────────────────────────────────
# v1-v6 son los 6 intervalos de follow-up de 11-D-0172.
# Se usan como tiempo ordinal cuando ids__visit_date no está disponible.
INTERVAL_ORDER = ["v1", "v2", "v3", "v4", "v5", "v6"]

# Tiempo nominal aproximado en meses por intervalo (del protocolo 11-D-0172).
# ADVERTENCIA: estos valores son aproximados; usar fechas reales cuando disponibles.
INTERVAL_MONTHS = {
    "v1":  0,   # basal / entrada
    "v2":  6,
    "v3":  18,
    "v4":  30,
    "v5":  42,
    "v6":  54,
}

# ── Variables ESSDAI ───────────────────────────────────────────────────────────

# OUTCOME PRINCIPAL: usar SIEMPRE la versión recodificada (-_r_)
ESSDAI_TOTAL = "essdai-_r__essdai_total_score"

# Familia cruda (40% cobertura) — solo para comparación en QC, no para análisis
ESSDAI_TOTAL_RAW = "essdai__essdai_total_score"   # NO USAR como outcome

# Dominios ESSDAI recodificados (12 dominios)
ESSDAI_DOMAINS = {
    "articular":    "essdai-_r__articular_domain",
    "biological":   "essdai-_r__biological_domain",
    "hematologic":  "essdai-_r__hematologic",
    "constitutional": "essdai-_r__constitutional",
    "gland_swell":  "essdai-_r__gland_swell",
    "cutaneous":    "essdai-_r__cutaneous",
    "cns":          "essdai-_r__cns",
    "muscular":     "essdai-_r__muscular_domain",
    "neuro_periph": "essdai-_r__neuro_peripheral",
    "pulmonary":    "essdai-_r__pulmonary",
    "renal":        "essdai-_r__renal",
    "lymphadenopathy": "essdai-_r__hema_lphdenopthy",
}

# Dominios con actividad no trivial en el dataset (change_rate > 5% en var_summary)
# Usar estos 4 como foco principal en modelos de dominio (04_domain_analysis.py)
ACTIVE_DOMAINS = ["articular", "biological", "hematologic", "lymphadenopathy"]

# Dominios esencialmente inactivos en esta cohorte (casi siempre = 0)
# Reportar solo como conteos; no modelar longitudinalmente
INACTIVE_DOMAINS = ["cns", "muscular", "renal", "cutaneous"]

# Pesos estándar ESSDAI por dominio (ACR/EULAR 2010)
# Usado para verificación y para reconstruir total si hay valores faltantes
ESSDAI_DOMAIN_WEIGHTS = {
    "constitutional":  3,
    "lymphadenopathy": 4,
    "gland_swell":     2,
    "articular":       4,
    "cutaneous":       3,
    "pulmonary":       5,
    "renal":           5,
    "muscular":        6,
    "neuro_periph":    5,
    "cns":             5,
    "hematologic":     2,
    "biological":      1,
}

# Encodings categoriales de dominios (valores esperados en dataset).
# ADVERTENCIA: validar contra los valores reales antes de correr análisis.
# Los valores 0/1/2/3 corresponden a los niveles de actividad del ESSDAI.
DOMAIN_LEVEL_LABELS = {
    0: "No activity",
    1: "Low activity",
    2: "Moderate activity",
    3: "High activity",
}

# ── Umbrales clínicos ESSDAI ───────────────────────────────────────────────────

ESSDAI_SEVERE      = 5   # ≥5 = moderate-to-severe (Pop 1)
ESSDAI_INACTIVE    = 5   # <5 = inactive/mild
ESSDAI_HIGH        = 14  # ≥14 = high activity (referencia de literatura)

# ── Variables ESSPRI (para clasificación Pop 1-3) ─────────────────────────────

ESSPRI_ITEMS = {
    "dryness":        "esspri_questionnaire__dryness",
    "fatigue":        "esspri_questionnaire__fatigue",
    "pain":           "esspri_questionnaire__pain",
    "mental_fatigue": "esspri_questionnaire__mental_fatigue",
}
ESSPRI_THRESHOLD = 5  # ≥5 = síntomas significativos (Pop 2)

# ESSPRI total = media de los 3 dominios principales: dryness, fatigue, pain
ESSPRI_MAIN_ITEMS = ["dryness", "fatigue", "pain"]

# ── Definición de poblaciones (OASIZ trial — uno.docx §7.2) ───────────────────

# Pop 1: ESSDAI ≥ 5 (cualquier ESSPRI) — actividad moderada a severa
# Pop 2: ESSPRI ≥ 5 AND ESSDAI < 5 — actividad inactiva/leve, síntomas significativos
# Pop 3: ESSDAI 0-4 AND ESSPRI < 5 — leve, sin síntomas significativos
# Pop 4: Sin enfermedad (comparador) — no incluir en análisis C2

POP_LABELS = {
    1: "Pop 1 (ESSDAI≥5, any ESSPRI)",
    2: "Pop 2 (ESSPRI≥5, ESSDAI<5)",
    3: "Pop 3 (ESSDAI<5, ESSPRI<5)",
}

# ── Criterios de elegibilidad para cohorte C2 ─────────────────────────────────

C2_MIN_ESSDAI_VISITS = 2   # ≥2 visitas con ESSDAI recodificado no-nulo
C2_MIN_ESSDAI_3_VISITS = 3  # umbral para sensibilidad con random slopes

# ── Covariables para modelos LMM ──────────────────────────────────────────────

# Covariables fijas en el modelo ajustado primario (§03)
# NOTA: disease_duration se deriva como (visit_date - dx_date) en 01_build_cohort_c2.py
LMM_COVARIATES = [
    "age_at_baseline",    # edad en primera visita ESSDAI evaluable
    "sex_female",         # 0/1 (female=1)
    "disease_duration_yrs",  # años desde dx_date a primera visita ESSDAI
    "essdai_baseline",    # ESSDAI en visita ancla (primera visita evaluable)
    "sjogrens_class_primary",  # 1 = SjD primaria, 0 = secundaria/otro
]

# ── Semilla aleatoria ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Configuración de figuras ───────────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"  # cambiar a "svg" para manuscrito
PALETTE_POP   = {1: "#d62728", 2: "#ff7f0e", 3: "#1f77b4"}  # rojo/naranja/azul
