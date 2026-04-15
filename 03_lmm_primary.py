"""
03_lmm_primary.py — Modelo de Efectos Mixtos Lineales (Análisis Primario)
==========================================================================

OBJETIVO:
    Estimar el cambio promedio en ESSDAI total a lo largo del tiempo usando
    un Linear Mixed Model (LMM) con random intercept por paciente.

    Este es el ANÁLISIS PRIMARIO para el Objetivo Primario 2.

JUSTIFICACIÓN DEL MODELO:
    - LMM con random intercept maneja correctamente la estructura de datos
      anidados (visitas dentro de pacientes).
    - Permite missingness no informativa en las visitas (MCAR/MAR bajo
      supuestos razonables).
    - Random slopes se reserva para análisis de SENSIBILIDAD en el subset
      de n=36 (≥3 visitas). NO es el modelo primario por inestabilidad.

RESTRICCIONES CRÍTICAS (del informe de viabilidad):
    - n=71 con ≥2 ESSDAI → LMM con random intercept: EJECUTABLE.
    - n=36 con ≥3 ESSDAI → LMM con random slopes: solo como SENSIBILIDAD.
    - LCMM, growth mixture models, HMM: NO EJECUTAR a este n.
    - Modelos con >4 covariables: propensos a overfitting a n=71.

LIBRERÍAS:
    Opción A (recomendada): statsmodels.formula.api.mixedlm
    Opción B: pymer4 (wrapper de lme4 via rpy2) — solo si R disponible en Biowulf
    Opción C: sklearn + manual REML (no recomendado)

OUTPUTS:
    outputs/tab_03_lmm_primary.csv          — coeficientes, IC95%, p-values
    outputs/tab_03_lmm_sensitivity.csv      — random slopes (n=36)
    outputs/tab_03_lmm_model_fit.csv        — AIC, BIC, log-likelihood
    outputs/fig_06_lmm_fitted_trajectories.png — trayectorias ajustadas + residuos
    outputs/fig_07_lmm_residuals.png        — diagnóstico de residuos

ADVERTENCIA SOBRE TIEMPO:
    El tiempo debe ser 'time_from_anchor_yrs' (años desde primera visita ESSDAI
    evaluable), NO 'interval_order'. Usar fechas reales cuando disponibles.
    Si una parte de los pacientes usa tiempo nominal y otra usa fechas reales,
    reportar esto como sensibilidad adicional.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Librería principal para LMM
try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    warnings.warn("statsmodels no disponible. Instalar: pip install statsmodels")

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, ESSDAI_TOTAL,
    LMM_COVARIATES, C2_MIN_ESSDAI_3_VISITS,
    ESSDAI_SEVERE, FIGURE_DPI, FIGURE_FORMAT,
    RANDOM_SEED,
)


# ── 1. Cargar y preparar datos para LMM ──────────────────────────────────────

def prepare_lmm_data(c2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara dos datasets:
    - df_lmm: C2 completo (n=71) para análisis primario (random intercept)
    - df_lmm_3plus: subset ≥3 visitas (n=36) para sensibilidad (random slopes)

    TAREA PARA CODEX:
    - Filtrar a solo filas con ESSDAI_TOTAL no nulo.
    - Verificar que 'time_from_anchor_yrs' exista; si no, construirla desde
      INTERVAL_MONTHS (con advertencia).
    - Verificar que todas las covariables de LMM_COVARIATES existen y no tienen
      >30% de missingness en C2. Si alguna tiene missingness alta, reportar y
      considerar imputación o exclusión.
    - Centrar 'time_from_anchor_yrs' en 0 (ya debería estarlo desde §01).
    - Centrar covariables continuas (age_at_baseline, disease_duration_yrs,
      essdai_baseline) en su media para reducir colinealidad con el intercept.
      Guardar los centros de escala como dict para revertir en tablas de resultados.
    - Verificar que PATIENT_ID tenga ≥2 filas en df_lmm; si alguno tiene 1,
      es un error de filtrado en §01 — reportar y excluir.
    - Reportar: n pacientes, n visitas totales, n visitas por paciente (mediana/IQR).

    IMPORTANTE: si 'sex_female' o 'sjogrens_class_primary' tienen varianza ~0
    en la submuestra (p.ej., todos son mujeres), excluir esa covariable del modelo
    con nota. No puede entrar una variable sin varianza.
    """
    c2_filtered = c2[c2[ESSDAI_TOTAL].notna()].copy()

    df_lmm = c2_filtered.copy()
    df_lmm_3plus = c2_filtered[c2_filtered['c2_subset_3plus'] == True].copy()

    # TODO: centrar covariables, verificar missingness, verificar varianza

    return df_lmm, df_lmm_3plus


# ── 2. Modelo primario: random intercept ─────────────────────────────────────

def run_lmm_random_intercept(df_lmm: pd.DataFrame) -> dict:
    """
    Modelo primario: LMM con random intercept.

    Especificación:
        ESSDAI_total ~ time_from_anchor_yrs
                       + age_at_baseline
                       + sex_female
                       + disease_duration_yrs
                       + essdai_baseline
                       + sjogrens_class_primary
                       + (1 | patient_id)         ← random intercept

    TAREA PARA CODEX:
    Con statsmodels.formula.api.mixedlm:

        formula = (
            f"{ESSDAI_TOTAL} ~ "
            "time_from_anchor_yrs "
            "+ age_at_baseline "
            "+ sex_female "
            "+ disease_duration_yrs "
            "+ essdai_baseline "
            "+ sjogrens_class_primary"
        )
        model = smf.mixedlm(
            formula,
            data=df_lmm,
            groups=df_lmm[PATIENT_ID],   # random intercept por paciente
            # re_formula=None  → solo random intercept
        )
        result = model.fit(reml=True, method='lbfgs')

    Post-ajuste:
    - Extraer tabla de coeficientes con IC95% bootstrap (nboot=1000, resampleo
      a nivel de paciente para preservar la estructura de clustering).
      Si bootstrap es costoso, usar IC Wald como aproximación con nota.
    - Calcular ICC (Intraclass Correlation Coefficient):
        ICC = var_random_intercept / (var_random_intercept + var_residual)
      Reportar ICC como medida de cuánta varianza está explicada por el paciente.
    - Extraer: AIC, BIC, log-likelihood, n pacientes, n visitas.
    - Si el modelo no converge: probar method='nm' o 'powell'. Documentar.

    ADVERTENCIA: statsmodels.mixedlm puede tener problemas de convergencia con
    datos pequeños y covariables correlacionadas. Si hay problemas, reducir el
    modelo al univariado (solo time_from_anchor_yrs + essdai_baseline) como
    fallback, y reportar ambos.

    RETORNA: dict con 'result' (objeto statsmodels), 'coef_table' (DataFrame),
             'icc', 'model_fit_stats', 'convergence_notes'.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels requerido. pip install statsmodels")

    # TODO: implementar modelo
    results = {}
    return results


# ── 3. Modelo de sensibilidad: random slopes (n=36) ──────────────────────────

def run_lmm_random_slopes(df_lmm_3plus: pd.DataFrame) -> dict:
    """
    Análisis de SENSIBILIDAD: LMM con random intercept + random slope en tiempo.
    Solo en el subconjunto de ≥3 visitas (n=36).

    Especificación:
        ESSDAI_total ~ time_from_anchor_yrs
                       + age_at_baseline + sex_female + essdai_baseline
                       + (1 + time_from_anchor_yrs | patient_id)

    Con statsmodels:
        model = smf.mixedlm(
            formula,
            data=df_lmm_3plus,
            groups=df_lmm_3plus[PATIENT_ID],
            re_formula="~time_from_anchor_yrs"   # random slope + intercept
        )

    TAREA PARA CODEX:
    - Intentar fit con REML=True primero. Si no converge (muy probable a n=36),
      probar REML=False, luego reducir covariables.
    - Si el modelo converge: comparar AIC con modelo random-intercept solo
      (usando likelihood ratio test via REML=False para ambos modelos).
    - Si NO converge: documentar claramente y reportar que random slopes no
      son estimables a este n → confirma restricción del informe de viabilidad.
    - NUNCA presentar un modelo no convergido en el manuscrito.

    RETORNA: dict con resultado o nota de no-convergencia.
    """
    # TODO: implementar
    results = {"convergence_notes": "Pendiente de implementación"}
    return results


# ── 4. Tabla de resultados LMM ────────────────────────────────────────────────

def format_lmm_table(lmm_result: dict, model_label: str) -> pd.DataFrame:
    """
    Formatea la tabla de coeficientes para el manuscrito.

    Columnas output:
    - Variable (con etiqueta legible, no nombre de columna)
    - Coefficient (β)
    - SE
    - 95% CI Lower, Upper
    - p-value (con corrección para múltiples comparaciones si aplica)
    - Stars (* p<.05, ** p<.01, *** p<.001)

    TAREA PARA CODEX:
    - Renombrar variables de código a etiquetas clínicas:
      'time_from_anchor_yrs' → 'Time from anchor (years)'
      'age_at_baseline' → 'Age at baseline (years)'
      'sex_female' → 'Female sex'
      'disease_duration_yrs' → 'Disease duration at anchor (years)'
      'essdai_baseline' → 'ESSDAI at anchor (baseline)'
      'sjogrens_class_primary' → 'Primary SjD (vs. secondary)'
    - Incluir al pie de tabla: n pacientes, n visitas, AIC, BIC, ICC.
    - Incluir advertencia al pie si hubo problemas de convergencia.
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 5. Figura: trayectorias ajustadas por el modelo ──────────────────────────

def fig_lmm_fitted_trajectories(c2: pd.DataFrame, lmm_result: dict, save=True):
    """
    Figura de trayectorias ajustadas por el LMM.

    Panel izquierdo: "Population-average trajectory"
    - Línea de efectos fijos (fitted values a nivel poblacional).
    - Banda de IC95% del efecto fijo de tiempo.
    - Puntos observados individuales (jitter).

    Panel derecho: "Subject-specific trajectories"
    - Fitted values condicionados a los random effects (BLUPs/fitted per patient).
    - 10-15 pacientes seleccionados aleatoriamente (set.seed=RANDOM_SEED).
    - Línea poblacional superpuesta en negro grueso.

    TAREA PARA CODEX:
    - Extraer fitted values del objeto statsmodels con result.fittedvalues.
    - Extraer efectos fijos con result.fe_params y la covarianza result.cov_params().
    - Para IC de la línea poblacional: usar predict con covariates en el rango
      del tiempo observado, con delta method o bootstrap.
    - Añadir línea de referencia en ESSDAI=5.
    - Anotar R² marginal y R² condicional (ver fórmula de Nakagawa & Schielzeth
      2013 para LMMs):
        R2_marginal    = var(fixed effects) / (var(fixed) + var(random) + var(residual))
        R2_conditional = (var(fixed) + var(random)) / (var(fixed) + var(random) + var(residual))

    Guardar como: outputs/fig_06_lmm_fitted_trajectories.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_06_lmm_fitted_trajectories.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 6. Figura: diagnóstico de residuos ────────────────────────────────────────

def fig_lmm_residuals(lmm_result: dict, save=True):
    """
    Diagnóstico estándar del LMM:
    - Panel 1: Residuos marginales vs. valores ajustados (homoscedasticidad).
    - Panel 2: Q-Q plot de residuos (normalidad).
    - Panel 3: Residuos por paciente (identificar outliers).
    - Panel 4: Random effects (BLUPs) del intercept — ¿distribución normal?

    TAREA PARA CODEX:
    - Calcular residuos marginales: resid = observed - fitted_marginal.
    - Calcular residuos condicionales: resid_cond = observed - fitted_conditional.
    - Usar scipy.stats.probplot para Q-Q plot.
    - Identificar outliers (|residuo| > 3 SD) y listarlos por patient_id.
    - Un patrón sistemático en Panel 1 indicaría violación de homoscedasticidad →
      considerar transformación log o modelo GEE como sensibilidad.

    Guardar como: outputs/fig_07_lmm_residuals.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_07_lmm_residuals.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 7. Análisis de sensibilidad: GEE como alternativa ────────────────────────

def run_gee_sensitivity(df_lmm: pd.DataFrame) -> dict:
    """
    Análisis de sensibilidad: GEE (Generalized Estimating Equations) para
    comparar con el LMM. GEE estima efectos poblacionales (marginal model)
    sin asumir distribución de random effects.

    Especificación:
        ESSDAI_total ~ time_from_anchor_yrs + essdai_baseline + age_at_baseline
        Correlation structure: exchangeable (AR-1 como alternativa)
        Cluster: PATIENT_ID

    Con statsmodels:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Gaussian

        model = GEE(
            endog=df_lmm[ESSDAI_TOTAL],
            exog=sm.add_constant(df_lmm[['time_from_anchor_yrs', 'essdai_baseline', ...]]),
            groups=df_lmm[PATIENT_ID],
            family=Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        result = model.fit()

    TAREA PARA CODEX:
    - Intentar dos estructuras de correlación: exchangeable y AR-1.
    - Comparar coeficientes de tiempo con LMM; si son consistentes, reportar
      como evidencia de robustez. Si divergen, investigar la razón.
    - GEE con QIC puede usarse para seleccionar estructura de correlación.
    - Reportar: coef tiempo con SE robusto, QIC para cada estructura.

    RETORNA: dict con resultados de ambas estructuras.
    """
    # TODO: implementar
    results = {}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("03_lmm_primary.py — LMM Análisis Primario ESSDAI")
    print("=" * 60)

    c2 = pd.read_parquet(COHORT_C2_FILE)

    # Preparar datos
    df_lmm, df_lmm_3plus = prepare_lmm_data(c2)

    # Modelo primario: random intercept (n=71)
    lmm_ri = run_lmm_random_intercept(df_lmm)
    tab_ri  = format_lmm_table(lmm_ri, "Primary (random intercept, n=71)")
    tab_ri.to_csv(OUT_DIR / "tab_03_lmm_primary.csv", index=False)

    # Sensibilidad A: random slopes (n=36)
    lmm_rs = run_lmm_random_slopes(df_lmm_3plus)
    tab_rs  = format_lmm_table(lmm_rs, "Sensitivity (random slopes, n=36)")
    tab_rs.to_csv(OUT_DIR / "tab_03_lmm_sensitivity.csv", index=False)

    # Sensibilidad B: GEE
    gee_res = run_gee_sensitivity(df_lmm)
    # TODO: guardar tabla GEE

    # Figuras
    fig_lmm_fitted_trajectories(c2, lmm_ri)
    fig_lmm_residuals(lmm_ri)

    # Tabla de fit del modelo
    fit_stats = pd.DataFrame([{
        "model": "LMM random intercept",
        "n_patients": df_lmm[PATIENT_ID].nunique(),
        "n_visits": len(df_lmm),
        "AIC": lmm_ri.get("aic", "N/A"),
        "BIC": lmm_ri.get("bic", "N/A"),
        "ICC": lmm_ri.get("icc", "N/A"),
        "convergence": lmm_ri.get("convergence_notes", "OK"),
    }])
    fit_stats.to_csv(OUT_DIR / "tab_03_lmm_model_fit.csv", index=False)

    print("\n✓ LMM completado. Ver outputs/ para tablas y figuras.")


if __name__ == "__main__":
    main()
