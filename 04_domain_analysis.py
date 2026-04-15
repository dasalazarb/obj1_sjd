"""
04_domain_analysis.py — Análisis por Dominio ESSDAI
====================================================

OBJETIVO:
    Describir la actividad longitudinal de cada dominio ESSDAI individualmente,
    con modelos GLMM para los dominios con actividad no trivial.

    Este análisis contribuye al sub-objetivo de "factores de riesgo de involucro
    orgánico" (Objetivo Primario 3), pero en su versión degradada (sin laboratorios),
    que es lo único ejecutable con el dataset actual: asociaciones dominio-dominio.

ESTRATEGIA POR TIPO DE DOMINIO:
    - Dominios CATEGÓRICOS (articular, biological, hematologic, lymphadenopathy,
      constitutional, neuro_peripheral, pulmonary, cutaneous):
      Tratar como ordinal 0-3 (niveles de actividad). Modelo: GLMM ordinal
      o dicotomizar en active/inactive (≥1 vs 0) para GLMM logístico.

    - Dominios BOOLEANOS (gland_swell, cns, muscular, renal, cutaneous):
      Solo 0/1. Modelo: GLMM logístico.

    - Para los 4 DOMINIOS ACTIVOS (articular, biological, hematologic,
      lymphadenopathy): ejecutar GLMM longitudinal.

    - Para los DOMINIOS INACTIVOS (cns, muscular, renal): solo reportar
      prevalencia y n de eventos; NO modelar.

RESTRICCIÓN IMPORTANTE:
    Los dominios inactivos en esta cohorte (cns=0 en ~100% de visitas,
    muscular=0 en ~100%, renal=0 en ~100%) generan quasi-separación perfecta
    en cualquier modelo logístico. No intentar ajustar modelos a estos dominios.

OUTPUTS:
    outputs/tab_04_domain_prevalence.csv      — prevalencia × intervalo por dominio
    outputs/tab_04_glmm_active_domains.csv    — GLMM para 4 dominios activos
    outputs/fig_08_domain_trajectories.png    — series temporales por dominio
    outputs/fig_09_domain_cooccurrence.png    — matrix de co-ocurrencia de dominios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from config import (
    COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, INTERVAL_COL,
    ESSDAI_TOTAL, ESSDAI_DOMAINS, ACTIVE_DOMAINS, INACTIVE_DOMAINS,
    INTERVAL_ORDER, ESSDAI_SEVERE,
    FIGURE_DPI, FIGURE_FORMAT,
)


# ── 1. Tabla de prevalencia por dominio × intervalo ───────────────────────────

def table_domain_prevalence(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada dominio en ESSDAI_DOMAINS y cada intervalo:
    - n pacientes con dominio no nulo (denominador)
    - n con actividad ≥1 (numerador)
    - % de actividad (con IC95% por Wilson o Clopper-Pearson)
    - Para dominios categóricos: distribución de niveles (0/1/2/3)

    TAREA PARA CODEX:
    - Iterar sobre ESSDAI_DOMAINS.items().
    - Para cada dominio, hacer groupby([INTERVAL_COL]) y calcular estadísticas.
    - Binarizar: 'any_activity' = (domain_value > 0) y no nulo.
    - Usar statsmodels.stats.proportion.proportion_confint(count, nobs,
      method='wilson') para IC95%.
    - Crear MultiIndex en la tabla: (domain_name, interval).
    - Añadir fila de "Total / all intervals" para cada dominio.

    IMPORTANTE: Usar el mismo denominador para todos los dominios en un
    intervalo dado (n con ESSDAI_TOTAL no nulo en ese intervalo). Esto
    hace las prevalencias de dominio comparables entre sí.
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 2. GLMM logístico para dominios activos ────────────────────────────────────

def run_glmm_domain(c2: pd.DataFrame, domain_key: str) -> dict:
    """
    GLMM logístico para un dominio específico de ACTIVE_DOMAINS.

    Outcome: binary 'active' = (domain_value > 0)

    Modelo:
        active ~ time_from_anchor_yrs + essdai_baseline + age_at_baseline
                 + (1 | PATIENT_ID)

    TAREA PARA CODEX:
    Con statsmodels:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        # O alternativamente con pymer4/R si está disponible:
        # fm = Lmer("active ~ time + baseline + (1|patient)", data=df, family='binomial')

    Advertencia: statsmodels.formula.api.mixedlm NO soporta GLMMs (solo
    respuestas gaussianas). Para respuestas binomiales, opciones son:
    a) BinomialBayesMixedGLM de statsmodels (Bayesiano, aprox. a LMM logístico).
    b) GEE logístico con statsmodels.genmod.generalized_estimating_equations.
    c) glmer() de R via rpy2/pymer4.

    RECOMENDACIÓN: usar GEE logístico como alternativa marginal que es
    más estable a n pequeño que GLMM. Documenta la elección.

    TAREA PARA CODEX:
    - Verificar si el dominio tiene suficiente variabilidad para modelar:
      si % activo < 5% o > 95% en todos los intervalos → skip, solo reportar
      conteos (quasi-separación).
    - Si el modelo converge, extraer: OR (exp(coef)), IC95%, p-value para tiempo.
    - Si no converge: documentar y reportar solo descriptivo.

    RETORNA: dict con 'domain', 'model_type', 'or_time', 'ci_lower', 'ci_upper',
             'p_value', 'convergence_note', 'n_events'.
    """
    domain_col = ESSDAI_DOMAINS[domain_key]
    # TODO: implementar GLMM/GEE para cada dominio activo
    result = {"domain": domain_key, "convergence_note": "Pendiente"}
    return result


def run_all_active_domain_models(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta run_glmm_domain para cada dominio en ACTIVE_DOMAINS.

    TAREA PARA CODEX:
    - Iterar sobre ACTIVE_DOMAINS.
    - Recolectar resultados en lista de dicts.
    - Convertir a DataFrame.
    - Aplicar corrección de Benjamini-Hochberg (FDR) sobre los p-values de
      'tiempo' de los 4 modelos (4 comparaciones).
    - Añadir columna 'p_adj_BH' y 'significant_BH' (p_adj < 0.05).
    """
    results = [run_glmm_domain(c2, dk) for dk in ACTIVE_DOMAINS]
    tab = pd.DataFrame(results)
    # TODO: aplicar corrección FDR
    return tab


# ── 3. Figura: series temporales por dominio ─────────────────────────────────

def fig_domain_trajectories(c2: pd.DataFrame, save=True):
    """
    Grid de subplots: 1 subplot por dominio en ACTIVE_DOMAINS (4 paneles).

    Cada panel:
    - Barras o líneas: % con actividad ≥1 por intervalo.
    - Error bars: IC95% Wilson.
    - Anotar n en cada punto.
    - Escalar eje Y de 0 a 100%.
    - Título del panel = nombre legible del dominio + (n eventos total).

    TAREA PARA CODEX:
    - Usar plt.subplots(2, 2, figsize=(10, 8), sharey=True).
    - Colorear barras por intervalo usando paleta qualitativa.
    - Añadir subtítulo general: "Active domain activity over time — C2 cohort"
    - Nota al pie: "Proportions shown for intervals with n≥10 patients with
      evaluable ESSDAI. Intervals v5/v6 shown as points (n<10)."

    Guardar como: outputs/fig_08_domain_trajectories.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_08_domain_trajectories.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 4. Figura: matriz de co-ocurrencia de dominios ────────────────────────────

def fig_domain_cooccurrence(c2: pd.DataFrame, save=True):
    """
    Matriz de co-ocurrencia entre dominios ESSDAI.
    Solo usar visita ancla (v1 / primera visita evaluable) para evitar
    inflación por pacientes con múltiples visitas.

    Celda (i,j) = % de pacientes con AMBOS dominios i y j activos (≥1)
    entre los que tienen al menos uno de los dos activo (Jaccard-like).
    O alternativamente: phi coefficient de correlación tetracórica.

    TAREA PARA CODEX:
    - Filtrar a visita ancla por paciente.
    - Binarizar todos los dominios (active = domain > 0).
    - Calcular matriz de correlación phi o simplemente la matriz de co-ocurrencia
      de Jaccard entre pares de dominios.
    - Usar sns.heatmap con anotaciones de %.
    - Separar visualmente ACTIVE_DOMAINS de INACTIVE_DOMAINS con líneas.
    - Título: "Domain co-occurrence at anchor visit — C2 cohort (n=XX)"

    NOTA CLÍNICA: La co-ocurrencia glandular (gland_swell) con cualquier EGM
    es el análisis principal para el Objetivo Primario 5 (solapamiento glandular/EGM).
    Este plot es el punto de partida visual para ese objetivo.

    Guardar como: outputs/fig_09_domain_cooccurrence.{FIGURE_FORMAT}
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_09_domain_cooccurrence.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("04_domain_analysis.py — Análisis por Dominio ESSDAI")
    print("=" * 60)

    c2 = pd.read_parquet(COHORT_C2_FILE)

    # Tabla de prevalencia
    tab_prev = table_domain_prevalence(c2)
    tab_prev.to_csv(OUT_DIR / "tab_04_domain_prevalence.csv", index=False)

    # GLMM para dominios activos
    tab_glmm = run_all_active_domain_models(c2)
    tab_glmm.to_csv(OUT_DIR / "tab_04_glmm_active_domains.csv", index=False)

    # Figuras
    fig_domain_trajectories(c2)
    fig_domain_cooccurrence(c2)

    print("\n✓ Análisis de dominios completado. Ver outputs/")


if __name__ == "__main__":
    main()
