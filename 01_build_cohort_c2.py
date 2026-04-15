"""
01_build_cohort_c2.py — Construcción de la Cohorte C2 (ESSDAI Longitudinal)
=============================================================================

OBJETIVO:
    Cargar el dataset wide (paciente × intervalo), aplicar los criterios de
    elegibilidad de la cohorte C2, derivar variables de tiempo, y exportar
    un parquet en formato LONG (una fila por paciente × visita) listo para
    análisis.

CRITERIO C2 (del informe de cohortes y viabilidad):
    - Pacientes con SjD documentada (sjogrens_class no nulo)
    - ≥2 visitas con essdai-_r__essdai_total_score no nulo
    - Solo protocolo 11-D-0172 (datos actuales)

ADVERTENCIAS:
    1. El dataset de entrada está en formato WIDE (1 fila = 1 paciente × intervalo).
       Este script lo transforma a formato LONG (1 fila = 1 visita evaluable).
    2. La fecha de visita (ids__visit_date) puede ser nula en algunos intervalos.
       Si es nula, se usa el tiempo nominal del intervalo (INTERVAL_MONTHS en config.py)
       como fallback. Documentar qué pacientes usan fechas reales vs. nominales.
    3. La variable de clase SjD (visit_summary_form__sjogrens_class) puede
       tener valores distintos entre visitas del mismo paciente. Usar la primera
       visita no-nula para definir la clase basal; alertar si cambia.
    4. disease_duration_yrs puede ser negativo si el paciente entró al NIH antes
       del diagnóstico formal. Revisar y capear en 0 con advertencia.

OUTPUT:
    outputs/cohort_c2.parquet         — dataset long C2 para análisis
    outputs/cohort_c2_exclusions.csv  — log de exclusiones con motivo
    outputs/cohort_c2_demographics.csv — tabla demográfica basal (1 fila/paciente)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from config import (
    VISITS_FILE, COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, INTERVAL_COL, VISIT_DATE,
    AGE_COL, SEX_COL, RACE_COL, ETHNICITY_COL,
    SJOGRENS_CLASS, DX_DATE_COL,
    ESSDAI_TOTAL, ESSDAI_DOMAINS,
    ESSPRI_ITEMS, ESSPRI_MAIN_ITEMS,
    INTERVAL_ORDER, INTERVAL_MONTHS,
    C2_MIN_ESSDAI_VISITS, C2_MIN_ESSDAI_3_VISITS,
    LMM_COVARIATES,
)


# ── 1. Cargar dataset wide ────────────────────────────────────────────────────

def load_wide(filepath: Path) -> pd.DataFrame:
    """
    Carga el CSV de visitas y hace limpieza mínima de tipos.

    TAREA PARA CODEX:
    - Convertir VISIT_DATE a datetime con pd.to_datetime(..., errors='coerce').
    - Convertir DX_DATE_COL a datetime con errors='coerce'.
    - Convertir ESSDAI_TOTAL a numérico con pd.to_numeric(..., errors='coerce').
    - Convertir AGE_COL a numérico.
    - Asegurarse de que INTERVAL_COL sea string limpio ('v1', 'v2', etc.).
    - Reportar % de nulos en columnas clave.
    """
    df = pd.read_csv(filepath, low_memory=False)

    # TODO: conversión de tipos (ver docstring)
    # df[VISIT_DATE]   = pd.to_datetime(df[VISIT_DATE], errors='coerce')
    # df[DX_DATE_COL]  = pd.to_datetime(df[DX_DATE_COL], errors='coerce')
    # df[ESSDAI_TOTAL] = pd.to_numeric(df[ESSDAI_TOTAL], errors='coerce')
    # df[AGE_COL]      = pd.to_numeric(df[AGE_COL], errors='coerce')

    print(f"[load_wide] Shape: {df.shape}")
    print(f"[load_wide] Pacientes únicos: {df[PATIENT_ID].nunique()}")
    print(f"[load_wide] Intervalos: {sorted(df[INTERVAL_COL].dropna().unique())}")
    return df


# ── 2. Transformar de wide a long ─────────────────────────────────────────────

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    El dataset ya está en formato 'pseudo-long' (1 fila = paciente × intervalo),
    por lo que no se necesita un pivot clásico. Solo se seleccionan las columnas
    relevantes y se ordena correctamente.

    TAREA PARA CODEX:
    - Seleccionar columnas: PATIENT_ID, INTERVAL_COL, VISIT_DATE, AGE_COL,
      SEX_COL, SJOGRENS_CLASS, DX_DATE_COL, ESSDAI_TOTAL, todos los dominios
      en ESSDAI_DOMAINS.values(), y los ítems ESSPRI en ESSPRI_ITEMS.values().
    - Crear columna 'interval_order' como entero (v1→0, v2→1, etc.) usando
      INTERVAL_ORDER.
    - Crear columna 'time_months' usando VISIT_DATE cuando no sea nula; si es
      nula, usar INTERVAL_MONTHS[interval_name]. Documentar origen en columna
      'time_source' ('date_actual' | 'interval_nominal').
    - Crear columna 'time_yrs' = time_months / 12 (para LMM).
    - Ordenar por PATIENT_ID, interval_order.
    """
    cols_to_keep = (
        [PATIENT_ID, INTERVAL_COL, VISIT_DATE, AGE_COL, SEX_COL,
         RACE_COL, ETHNICITY_COL, SJOGRENS_CLASS, DX_DATE_COL,
         ESSDAI_TOTAL]
        + list(ESSDAI_DOMAINS.values())
        + list(ESSPRI_ITEMS.values())
    )
    # Mantener solo columnas que existen en el dataset
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    long = df[cols_to_keep].copy()

    # TODO: interval_order, time_months, time_yrs, time_source (ver docstring)

    return long


# ── 3. Derivar variables de tiempo y covariables ──────────────────────────────

def derive_time_variables(long: pd.DataFrame) -> pd.DataFrame:
    """
    TAREA PARA CODEX:
    - 'disease_duration_yrs': (VISIT_DATE - DX_DATE_COL).dt.days / 365.25
      Si negativo → poner 0 y registrar en log de advertencias.
      Si DX_DATE_COL es nulo → NaN; documentar proporción.
    - 'time_from_anchor_yrs': tiempo desde primera visita ESSDAI evaluable
      del paciente (no desde v1 global, sino desde primera visita no-nula en C2).
      Derivar como: visit_date - min(visit_date where ESSDAI_TOTAL not null)
      dentro de cada PATIENT_ID.
    - 'essdai_baseline': ESSDAI_TOTAL en la visita ancla (time_from_anchor_yrs==0).
    - 'age_at_baseline': AGE_COL en la visita ancla.
    - 'sex_female': 1 si SEX_COL == 'Female' (o equivalente); 0 si no.
      VERIFICAR los valores únicos reales de SEX_COL antes de hacer esta asignación.
    - 'sjogrens_class_primary': 1 si SJOGRENS_CLASS indica SjD primaria.
      VERIFICAR valores únicos reales de SJOGRENS_CLASS. Puede ser 'Primary',
      'primary', '1', etc.
    - 'esspri_total': media de los 3 ítems principales (dryness, fatigue, pain)
      si ≥2 de 3 no son nulos; si <2 → NaN.

    ALERTAS:
    - Si disease_duration_yrs < 0: print warning con patient_id y valor.
    - Si sjogrens_class varía entre visitas del mismo paciente: registrar en log.
    """
    # TODO: implementar todas las derivaciones descritas

    return long


# ── 4. Aplicar criterios de elegibilidad C2 ───────────────────────────────────

def build_c2(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica criterios de inclusión/exclusión para cohorte C2 y devuelve
    (c2_long, exclusion_log).

    Criterios de INCLUSIÓN:
    1. Paciente tiene SjD documentada: SJOGRENS_CLASS no es nulo y != 'Healthy Volunteer'
       y != 'Non-SjD' (verificar valores exactos en el dataset).
    2. Paciente tiene ≥ C2_MIN_ESSDAI_VISITS (=2) visitas con ESSDAI_TOTAL no nulo.

    Criterios de EXCLUSIÓN (registrar en exclusion_log con motivo):
    - Sin SjD documentada
    - Solo 1 visita con ESSDAI evaluable
    - Sin ninguna visita con ESSDAI evaluable

    TAREA PARA CODEX:
    - Construir exclusion_log como DataFrame con columnas:
      [PATIENT_ID, 'n_essdai_visits', 'sjogrens_class_value', 'exclusion_reason']
    - Reportar: total de pacientes en raw → excluidos por cada criterio → n en C2.
    - Agregar flag 'c2_subset_3plus' = True si el paciente tiene ≥3 visitas ESSDAI
      (para el análisis de sensibilidad con random slopes en §03).

    NOTA: El n esperado en C2 es ≈71. Si el resultado se desvía significativamente,
    revisar la lógica de filtrado y los valores de SJOGRENS_CLASS.
    """
    exclusion_log = []

    # TODO: lógica de exclusión

    # Reportar
    # print(f"[build_c2] Pacientes raw: {long[PATIENT_ID].nunique()}")
    # print(f"[build_c2] Pacientes en C2: {c2[PATIENT_ID].nunique()}")
    # print(f"[build_c2] C2 subset ≥3 visitas: {c2['c2_subset_3plus'].sum()}")

    c2 = long.copy()  # placeholder — reemplazar con filtrado real
    return c2, pd.DataFrame(exclusion_log)


# ── 5. Tabla demográfica basal (1 fila por paciente) ─────────────────────────

def build_demographics_table(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la Tabla 1 del manuscrito: descripción basal de la cohorte C2.

    TAREA PARA CODEX:
    Calcular para cada variable de la lista debajo, usando SOLO la visita ancla
    de cada paciente (time_from_anchor_yrs == 0):

    Variables continuas (reportar como mediana [IQR]):
    - age_at_baseline
    - disease_duration_yrs
    - essdai_baseline
    - esspri_total (solo para pacientes con ESSPRI disponible)

    Variables categóricas (reportar como n, %):
    - sex_female
    - race (agrupar categorías con n<5 como 'Other/Unknown')
    - ethnicity
    - sjogrens_class_primary
    - essdai_severe_baseline = (essdai_baseline >= ESSDAI_SEVERE) → Pop 1

    Estratificar por: OVERALL, c2_subset_3plus (≥3 vs 2 visitas).

    OUTPUT: DataFrame con columnas [variable, overall, stratum_2visits, stratum_3plus]
    """
    # TODO: implementar tabla demográfica

    demo = pd.DataFrame()  # placeholder
    return demo


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01_build_cohort_c2.py — Construcción de cohorte C2")
    print("=" * 60)

    # Cargar
    df_wide = load_wide(VISITS_FILE)

    # Transformar a long
    df_long = wide_to_long(df_wide)

    # Derivar tiempo y covariables
    df_long = derive_time_variables(df_long)

    # Aplicar criterios C2
    c2, exclusion_log = build_c2(df_long)

    # Tabla demográfica
    demographics = build_demographics_table(c2)

    # Guardar outputs
    c2.to_parquet(COHORT_C2_FILE, index=False)
    exclusion_log.to_csv(OUT_DIR / "cohort_c2_exclusions.csv", index=False)
    demographics.to_csv(OUT_DIR / "cohort_c2_demographics.csv", index=False)

    print(f"\n✓ Cohort C2 guardada en: {COHORT_C2_FILE}")
    print(f"✓ Log de exclusiones:    {OUT_DIR}/cohort_c2_exclusions.csv")
    print(f"✓ Tabla demográfica:     {OUT_DIR}/cohort_c2_demographics.csv")


if __name__ == "__main__":
    main()
