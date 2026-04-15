"""
06_output_tables.py — Consolidación de Tablas para Manuscrito
=============================================================

OBJETIVO:
    Consolidar todos los outputs de los scripts anteriores en tablas
    formateadas listas para insertar en el manuscrito. También genera
    un reporte de limitaciones pre-redactado para incluir en el SAP.

TABLAS DEL MANUSCRITO:
    Table 1: Características basales de la cohorte C2
    Table 2: ESSDAI total por intervalo (mediana, IQR, % severo)
    Table 3: Resultados del LMM (coeficientes, IC95%, p-values)
    Table 4: Actividad por dominio ESSDAI × intervalo
    Table 5: Distribución Pop 1-3 por intervalo

IMPORTANTE: La función check_sample_sizes() debe ejecutarse PRIMERO.
Si los n reales difieren significativamente de los esperados en el
informe de viabilidad (n=71 C2, n=36 ≥3 visitas), reportar al
investigador antes de proceder con el análisis.

OUTPUTS:
    outputs/manuscript_table1.xlsx
    outputs/manuscript_table2.xlsx
    outputs/manuscript_table3.xlsx
    outputs/manuscript_table4.xlsx
    outputs/manuscript_table5.xlsx
    outputs/sap_limitations_language.txt   — texto SAP-ready
    outputs/analytic_decisions_log.csv     — registro de decisiones analíticas
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from config import (
    COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, C2_MIN_ESSDAI_VISITS, C2_MIN_ESSDAI_3_VISITS,
    ESSDAI_TOTAL, ESSDAI_SEVERE,
    POP_LABELS,
)


# ── 1. Verificación de tamaños muestrales ─────────────────────────────────────

def check_sample_sizes(c2: pd.DataFrame) -> dict:
    """
    Verificar que los n reales coinciden con los esperados del informe de viabilidad.

    TAREA PARA CODEX:
    - Calcular n_patients_c2, n_essdai_ge1, n_essdai_ge2, n_essdai_ge3.
    - Calcular n_with_esspri (para Pop classification).
    - Calcular n_paired_essdai_esspri_ge2.
    - Comparar contra valores esperados:
        n_essdai_ge1 ≈ 132 (83.5% de 158)
        n_essdai_ge2 ≈ 71  (44.9%)
        n_essdai_ge3 ≈ 36  (22.8%)
    - Si la diferencia es >15%, lanzar warning prominente.

    RETORNA: dict con los n reales y comparación vs. esperados.
    """
    checks = {}
    # TODO: implementar
    return checks


# ── 2. Table 1: Características basales ───────────────────────────────────────

def build_table1(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 1 del manuscrito: características clínicas y demográficas en la
    visita ancla (primera visita ESSDAI evaluable por paciente).

    Estructura:
    Sección A — Demografía:
        - Edad en ancla (mediana [IQR])
        - Sexo femenino (n, %)
        - Raza / etnia (n, % por categoría)
        - Duración de la enfermedad en ancla en años (mediana [IQR])
        - Clase SjD: primaria vs. secundaria (n, %)

    Sección B — ESSDAI en ancla:
        - ESSDAI total (mediana [IQR])
        - Pop 1 (ESSDAI ≥5): n, %
        - Por dominio: % activo (n, %)

    Sección C — ESSPRI en ancla (solo para pacientes con ESSPRI disponible):
        - ESSPRI total (mediana [IQR])
        - Pop 2 (ESSPRI ≥5, ESSDAI <5): n, % del total C2

    Sección D — Follow-up:
        - Número de visitas ESSDAI: mediana [IQR], rango
        - Duración del follow-up en meses (mediana [IQR])

    TAREA PARA CODEX:
    - Para variables continuas: usar median_iqr() helper.
    - Para variables categóricas: usar n_pct() helper.
    - Formato de salida: cada fila = una variable, columnas = categorías de
      estratificación (Total, ≥3 visitas, 2 visitas).
    - Añadir p-value de comparación ≥3 vs. 2 visitas (Mann-Whitney U para
      continuas, chi-cuadrado para categóricas). Esto verifica si hay
      sesgo de selección por completeness.
    """
    # TODO: implementar
    return pd.DataFrame()


# ── 3. Table 3: Resultados LMM formateados ────────────────────────────────────

def build_table3_lmm() -> pd.DataFrame:
    """
    Cargar outputs de 03_lmm_primary.py y formatear para manuscrito.

    TAREA PARA CODEX:
    - Cargar outputs/tab_03_lmm_primary.csv y tab_03_lmm_sensitivity.csv.
    - Combinar en una sola tabla con columnas:
        Variable | Primary β (95% CI) | Primary p | Sensitivity β (95% CI) | Notes
    - Añadir filas de 'Model statistics' al pie:
        n patients | n visits | AIC | BIC | ICC | Convergence
    - Resaltar en notas: "Primary model: random intercept (n=71).
      Sensitivity: random slopes (n=36). See text for GEE comparison."
    """
    # TODO: implementar
    return pd.DataFrame()


# ── 4. Exportar a Excel ───────────────────────────────────────────────────────

def export_to_excel(tables: dict, filename: str):
    """
    Exportar múltiples DataFrames a un Excel multi-hoja.

    TAREA PARA CODEX:
    - Usar pd.ExcelWriter con engine='openpyxl'.
    - Una hoja por tabla en 'tables' dict (clave = nombre de hoja).
    - Aplicar formato básico:
        - Cabeceras en negrita.
        - Filas alternas en gris claro (para legibilidad).
        - Columnas de p-value con formato numérico especial.
    - Añadir hoja 'README' con descripción de cada tabla y fecha de generación.
    """
    if not HAS_OPENPYXL:
        # Fallback: exportar como CSV separados
        for name, df in tables.items():
            df.to_csv(OUT_DIR / f"{filename}_{name}.csv", index=False)
        return

    with pd.ExcelWriter(OUT_DIR / filename, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        # TODO: aplicar formato con openpyxl


# ── 5. Generar texto SAP-ready de limitaciones ────────────────────────────────

def generate_sap_limitations_text(checks: dict) -> str:
    """
    Genera texto SAP-ready para la sección de limitaciones del análisis,
    basado en los n reales y en el informe de viabilidad.

    TAREA PARA CODEX:
    - El texto debe incluir EXACTAMENTE los n reales (no los esperados).
    - Usar los checks de check_sample_sizes() como insumo.
    - Incluir los siguientes párrafos pre-redactados, llenando los blancos
      con los valores reales:

    ─── TEMPLATE ───────────────────────────────────────────────────────────────

    Analytic Cohort and Sample Size Limitations

    The longitudinal ESSDAI cohort (C2) comprised {n_c2} patients with primary
    or secondary Sjögren's disease, of whom {n_ge2} ({pct_ge2}%) had two or more
    evaluable ESSDAI assessments and {n_ge3} ({pct_ge3}%) had three or more.
    These counts represent the binding ceiling for all repeated-measures models
    of disease activity. The low proportion of patients with three or more
    evaluable visits ({pct_ge3}%) constrains the models to a random-intercept
    specification; random-slope models were evaluated as sensitivity analyses
    on the {n_ge3}-patient subset and are clearly labeled as exploratory.

    Models NOT Executed

    In accordance with the viability assessment (Sección 6.2), the following
    analyses described in earlier versions of the protocol are not executed
    in this report due to insufficient sample size: latent class mixed models,
    growth mixture models, hidden Markov models, and multi-state transition
    models. These approaches require a minimum of {min_n_advanced} patients with
    three or more complete assessments to yield stable parameter estimates;
    the current dataset falls substantially below this threshold.

    Population Classification Limitations

    Population classification into Pop 1–3 (OASIZ trial subgroups) required
    paired ESSDAI and ESSPRI at the same visit. Of the {n_c2} C2 patients,
    {n_paired} ({pct_paired}%) had at least one paired assessment. Patients
    with evaluable ESSDAI but missing ESSPRI at a given visit were classified
    into Pop 1 when ESSDAI ≥ 5 but could not be assigned to Pop 2 versus Pop 3;
    these visits are labeled "ESSDAI-only" in Table 5 and excluded from Pop 2/3
    prevalence denominators. Due to the high within-patient ESSPRI variability
    (estimated inter-visit change rate ≈ 70–80%), phenotypic classification at
    individual visits should be interpreted with caution. Cross-visit transitions
    are presented as descriptive Sankey diagrams only, not as model-derived
    transition rates.

    ─────────────────────────────────────────────────────────────────────────────

    TAREA: llenar los {placeholders} con los valores del dict 'checks'.
    """
    text = """
=============================================================
SAP LIMITATIONS LANGUAGE — Objetivo Primario 2
Generated from: 06_output_tables.py
=============================================================

[FILL IN actual n values from checks dict]

[See template in function docstring for full text]
"""
    # TODO: implementar sustitución de placeholders con valores reales
    return text


# ── 6. Registro de decisiones analíticas ─────────────────────────────────────

def build_analytic_decisions_log() -> pd.DataFrame:
    """
    Registro formal de todas las decisiones analíticas tomadas durante
    la implementación, para transparencia y reproducibilidad.

    TAREA PARA CODEX:
    - Crear un DataFrame con columnas:
      [decision_id, script, decision_point, choice_made, rationale, date]
    - Poblar con las decisiones documentadas en los docstrings de los scripts:
      * Uso de essdai-_r_ vs essdai_ (razón: cobertura 83% vs 40%)
      * Tiempo nominal vs. fechas reales (cuántos pacientes de cada tipo)
      * Método de LMM (statsmodels.mixedlm vs. pymer4)
      * Manejo de missingness (complete case vs. MCAR assumption)
      * Corrección de FDR en análisis de dominios (BH method)
      * Dominios excluidos de modelos (INACTIVE_DOMAINS) y razón
    - Actualizar este log cada vez que se tome una nueva decisión.

    OUTPUT: outputs/analytic_decisions_log.csv
    """
    log = [
        {
            "decision_id": "D01",
            "script": "01_build_cohort_c2.py",
            "decision_point": "ESSDAI variable selection",
            "choice_made": "Use essdai-_r__essdai_total_score (recodificada)",
            "rationale": "Coverage 83.5% vs 40.5% for raw version; documented in viability report §4.1",
            "date": "TBD",
        },
        {
            "decision_id": "D02",
            "script": "03_lmm_primary.py",
            "decision_point": "Random effects structure",
            "choice_made": "Random intercept only (primary); random slopes (sensitivity n=36)",
            "rationale": "Random slopes unstable at n=71; random slopes only on n=36 subset. Viability report §6.2.",
            "date": "TBD",
        },
        {
            "decision_id": "D03",
            "script": "03_lmm_primary.py",
            "decision_point": "Models NOT executed",
            "choice_made": "LCMM, growth mixture, HMM excluded",
            "rationale": "n=36 with 3+ visits; events in single/low double digits. Indefensible at this n.",
            "date": "TBD",
        },
        {
            "decision_id": "D04",
            "script": "05_severity_strata.py",
            "decision_point": "Pop 1-3 transition analysis type",
            "choice_made": "Descriptive Sankey only; no multi-state model",
            "rationale": "ESSPRI change rate ~70-80% inter-visit; transitions dominated by noise. Viability §6.7.",
            "date": "TBD",
        },
        # TODO: añadir decisiones adicionales tomadas durante implementación
    ]
    return pd.DataFrame(log)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("06_output_tables.py — Tablas para Manuscrito")
    print("=" * 60)

    c2 = pd.read_parquet(OUT_DIR / "cohort_c2_with_pop.parquet")

    # Verificar n
    checks = check_sample_sizes(c2)
    print("\n[check_sample_sizes]", checks)

    # Construir tablas
    table1 = build_table1(c2)
    table3 = build_table3_lmm()

    # Exportar a Excel
    export_to_excel(
        tables={
            "Table1_Baseline": table1,
            "Table3_LMM_Results": table3,
        },
        filename="manuscript_tables.xlsx"
    )

    # Texto de limitaciones SAP
    sap_text = generate_sap_limitations_text(checks)
    with open(OUT_DIR / "sap_limitations_language.txt", "w") as f:
        f.write(sap_text)
    print(f"\n✓ Texto SAP guardado en: {OUT_DIR}/sap_limitations_language.txt")

    # Log de decisiones
    log = build_analytic_decisions_log()
    log.to_csv(OUT_DIR / "analytic_decisions_log.csv", index=False)
    print(f"✓ Log de decisiones: {OUT_DIR}/analytic_decisions_log.csv")

    print("\n✓ Todas las tablas de manuscrito generadas. Ver outputs/")


if __name__ == "__main__":
    main()
