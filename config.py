"""
config.py — Configuración global para Objetivo Primario 2
==========================================================
Fuente única de verdad para rutas, nombres de columnas, constantes clínicas,
encodings de dominios ESSDAI, y parámetros de cohorte.

Todos los scripts del proyecto importan desde aquí. No hardcodear
nombres de columnas ni umbrales en los scripts de análisis.

NOTAS:
- PHASE_LABELS mapea los nombres reales del campo ids__interval_name
  a etiquetas cortas para figuras y tablas.
- INTERVAL_MONTHS_NOMINAL son tiempos APROXIMADOS del protocolo.
  Usar SIEMPRE ids__visit_date cuando esté disponible.
  Los valores None indican que el tiempo nominal es desconocido para esa fase
  — fuerzan un error explícito si se intenta usar sin verificar.
- ESSDAI_SEVERE = 5 es el umbral del protocolo (uno.docx §7.1).
- No modificar ESSDAI_DOMAIN_WEIGHTS ni DOMAIN_LEVEL_LABELS sin
  validación clínica del investigador.
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────

# Ajustar DATA_DIR a la ubicación real del dataset en Biowulf
DATA_DIR = Path("data/raw")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Archivo fuente principal (CSV wide: paciente × fase, 2057 columnas)
VISITS_FILE = DATA_DIR / "visits_long_collapsed_by_interval_codebook_type_recode-3.csv"

# Variable-summary (métricas de cobertura — para triaje, no para análisis)
VAR_SUMMARY_FILE = DATA_DIR / "longitudinal_variable_summary-12.csv"

# Cohorte C2 serializada (generada por 01_build_cohort_c2.py)
COHORT_C2_FILE = OUT_DIR / "cohort_c2.parquet"

# ── Identificadores y columnas clave ──────────────────────────────────────────

PATIENT_ID    = "ids__patient_record_number"
INTERVAL_COL  = "ids__interval_name"          # nombre real de la fase en el dataset
VISIT_DATE    = "ids__visit_date"
AGE_COL       = "ids__age_at_visit"
SEX_COL       = "ids__sex"
RACE_COL      = "ids__race"
ETHNICITY_COL = "ids__ethnicity"
SJOGRENS_CLASS = "visit_summary_form__sjogrens_class"
DX_DATE_COL   = "sjogren's_syndrome_history__sjogrens_dx_date"

# ── Fases del protocolo ────────────────────────────────────────────────────────
#
# Mapeo de los valores reales de ids__interval_name a etiquetas cortas.
# El orden del dict define el orden canónico (phase_order 0-6).
# Dos entradas apuntan a "V1" porque algunos registros usan el nombre del
# protocolo anterior y otros tienen "(missing)" — verificar si son la misma
# visita antes de colapsar.

PHASE_LABELS: dict[str, str] = {
    "Natural History Protocol 478 Interval": "V1 (Nat. Hist.)",
    "(missing)":                             "V1 (missing)",
    "Phase 1: Initial Full Evaluation":      "V2 (Initial)",
    "Phase 1: Second Full Evaluation":       "V3 (Second)",
    "Phase 1: Final Full (Third Full) Evaluation": "V4 (Final/3rd)",
    "Phase 2: 4th Full Evaluation":          "V5 (4th)",
    "Phase 2: 5th Full Evaluation":          "V6 (5th)",
}

# Etiquetas cortas en orden canónico (para ejes de figuras)
INTERVAL_SHORT_LABELS: list[str] = list(PHASE_LABELS.values())

# ── Tiempos nominales por fase (FALLBACK — usar solo si ids__visit_date es nulo) ──
#
# ADVERTENCIA: estos valores son APROXIMACIONES del protocolo 11-D-0172,
# no tiempos reales. Usar SIEMPRE ids__visit_date cuando esté disponible.
#
# Valores None = tiempo nominal desconocido para esa fase.
# Si el código intenta usar un None como número, fallará explícitamente,
# lo cual es el comportamiento correcto (mejor un error que un número incorrecto
# en el LMM sin que nadie lo note).
#
# Verificar los tiempos reales con:
#   df.groupby(INTERVAL_COL)[VISIT_DATE].agg(['min','max','count'])
# y actualizar estos valores si corresponde.

INTERVAL_MONTHS_NOMINAL: dict[str, int | None] = {
    "Natural History Protocol 478 Interval": 0,
    "(missing)":                             0,
    "Phase 1: Initial Full Evaluation":      None,  # verificar contra fechas reales
    "Phase 1: Second Full Evaluation":       None,
    "Phase 1: Final Full (Third Full) Evaluation": None,
    "Phase 2: 4th Full Evaluation":          None,
    "Phase 2: 5th Full Evaluation":          None,
}

# ── Variables ESSDAI ───────────────────────────────────────────────────────────

# OUTCOME PRINCIPAL — usar SIEMPRE la versión recodificada (-_r_)
# Cobertura: 132 pacientes (83.5%) con ≥1 medición, 71 (44.9%) con ≥2
ESSDAI_TOTAL = "essdai-_r__essdai_total_score"

# Versión cruda — solo para QC; NO usar como outcome (40.5% cobertura)
ESSDAI_TOTAL_RAW = "essdai__essdai_total_score"

# 12 dominios ESSDAI recodificados
# Niveles: 0=No activity, 1=Low, 2=Moderate, 3=High (confirmado por investigador)
ESSDAI_DOMAINS: dict[str, str] = {
    "constitutional":   "essdai-_r__constitutional",
    "lymphadenopathy":  "essdai-_r__hema_lphdenopthy",
    "gland_swell":      "essdai-_r__gland_swell",
    "articular":        "essdai-_r__articular_domain",
    "cutaneous":        "essdai-_r__cutaneous",
    "pulmonary":        "essdai-_r__pulmonary",
    "renal":            "essdai-_r__renal",
    "muscular":         "essdai-_r__muscular_domain",
    "neuro_periph":     "essdai-_r__neuro_peripheral",
    "cns":              "essdai-_r__cns",
    "hematologic":      "essdai-_r__hematologic",
    "biological":       "essdai-_r__biological_domain",
}

# Dominios con actividad no trivial (change_rate > 5% en var_summary)
# → foco principal de modelos de dominio en 04_domain_analysis.py
ACTIVE_DOMAINS: list[str] = [
    "articular", "biological", "hematologic", "lymphadenopathy",
]

# Dominios esencialmente inactivos en esta cohorte (casi siempre = 0)
# → reportar solo como conteos; NO modelar longitudinalmente
INACTIVE_DOMAINS: list[str] = [
    "cns", "muscular", "renal", "cutaneous",
]

# Etiquetas legibles para los niveles de actividad (confirmadas por investigador)
DOMAIN_LEVEL_LABELS: dict[int, str] = {
    0: "No activity",
    1: "Low activity",
    2: "Moderate activity",
    3: "High activity",
}

# Pesos estándar ESSDAI por dominio (ACR/EULAR 2010)
# Usado en QC para reconstruir el total desde dominios y detectar discrepancias
ESSDAI_DOMAIN_WEIGHTS: dict[str, int] = {
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

# ── Umbrales clínicos ESSDAI ───────────────────────────────────────────────────

ESSDAI_SEVERE = 5    # ≥5 → moderate-to-severe / Pop 1 (protocolo uno.docx §7.1)
ESSDAI_HIGH   = 14   # ≥14 → high activity (referencia de literatura)
# ESSDAI_INACTIVE es simplemente < ESSDAI_SEVERE; no se define por separado

# ── Variables ESSPRI ──────────────────────────────────────────────────────────

ESSPRI_ITEMS: dict[str, str] = {
    "dryness":        "esspri_questionnaire__dryness",
    "fatigue":        "esspri_questionnaire__fatigue",
    "pain":           "esspri_questionnaire__pain",
    "mental_fatigue": "esspri_questionnaire__mental_fatigue",
}

# Ítems usados para el score total ESSPRI (media de estos 3)
ESSPRI_MAIN_ITEMS: list[str] = ["dryness", "fatigue", "pain"]

ESSPRI_THRESHOLD = 5  # ≥5 → síntomas significativos (define Pop 2)

# ── Poblaciones OASIZ (uno.docx §7.2) ────────────────────────────────────────
#
# Pop 1: ESSDAI ≥ 5 (cualquier ESSPRI)      → actividad moderada a severa
# Pop 2: ESSPRI ≥ 5 AND ESSDAI < 5          → inactiva/leve + síntomas significativos
# Pop 3: ESSDAI < 5 AND ESSPRI < 5          → leve sin síntomas significativos
# Pop 4: sin enfermedad (comparador)         → NO incluir en análisis C2

POP_LABELS: dict[int, str] = {
    1: "Pop 1 (ESSDAI≥5, any ESSPRI)",
    2: "Pop 2 (ESSPRI≥5, ESSDAI<5)",
    3: "Pop 3 (ESSDAI<5, ESSPRI<5)",
}

# ── Criterios de elegibilidad cohorte C2 ──────────────────────────────────────

C2_MIN_ESSDAI_VISITS   = 2   # ≥2 visitas con ESSDAI recodificado no-nulo (análisis primario)
C2_MIN_ESSDAI_3_VISITS = 3   # ≥3 visitas (subconjunto para random slopes en §03)

# ── Covariables LMM (modelo ajustado primario — §03) ─────────────────────────
#
# Todas estas variables son derivadas en 01_build_cohort_c2.py.
# Verificar varianza > 0 antes de incluir en el modelo.

LMM_COVARIATES: list[str] = [
    "age_at_baseline",           # edad en la visita ancla
    "sex_female",                # 0/1 (female=1)
    "disease_duration_yrs_model",# años desde dx_date hasta ancla (capeado ≥0)
    "essdai_baseline",           # ESSDAI en la visita ancla
    "sjogrens_class_primary",    # 1=SjD primaria, 0=secundaria/otro
]

# ── Reproducibilidad y figuras ────────────────────────────────────────────────

RANDOM_SEED   = 42
FIGURE_DPI    = 150
FIGURE_FORMAT = "png"   # cambiar a "svg" para versión de manuscrito

PALETTE_POP: dict[int, str] = {
    1: "#d62728",   # rojo   — Pop 1 (severo)
    2: "#ff7f0e",   # naranja — Pop 2 (síntomas)
    3: "#1f77b4",   # azul   — Pop 3 (leve)
}
