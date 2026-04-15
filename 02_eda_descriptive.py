"""
02_eda_descriptive.py — EDA y análisis descriptivo longitudinal de ESSDAI
==========================================================================

OBJETIVO:
    Generar todos los descriptivos, verificaciones de cobertura, y figuras
    exploratorias para el ESSDAI total y por dominio en la cohorte C2.
    Este script produce las tablas y figuras que van en la sección de
    "Estadísticas descriptivas y natural history" del manuscrito.

FIGURAS A PRODUCIR:
    1. fig_01_coverage_heatmap.png    — cobertura de ESSDAI por paciente × intervalo
    2. fig_02_spaghetti_essdai.png    — trayectorias individuales de ESSDAI total
    3. fig_03_median_trajectory.png   — mediana [IQR] de ESSDAI por intervalo
    4. fig_04_domain_heatmap.png      — actividad por dominio × intervalo (% activos)
    5. fig_05_essdai_distribution.png — distribución de ESSDAI en cada intervalo

TABLAS A PRODUCIR:
    tab_02_essdai_by_interval.csv     — mediana, IQR, n por intervalo
    tab_02_domain_activity.csv        — % activo por dominio × intervalo
    tab_02_within_patient_change.csv  — cambio dentro-paciente entre visitas consecutivas

ADVERTENCIAS DOCUMENTADAS:
    - Solo n=71 tienen ≥2 ESSDAI; n=36 tienen ≥3. Reportar n en todas las figuras.
    - Intervalos v5 (n=21) y v6 (n=5) tienen muy pocos datos → no modelar,
      solo mostrar como puntos descriptivos con nota.
    - change_rate ESSDAI total en var_summary ≈ 100% (cualquier cambio = cambio);
      esto es esperado para un score continuo de actividad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from config import (
    COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, INTERVAL_COL, VISIT_DATE,
    ESSDAI_TOTAL, ESSDAI_DOMAINS, ACTIVE_DOMAINS,
    INTERVAL_ORDER, INTERVAL_MONTHS,
    ESSDAI_SEVERE, ESSDAI_HIGH,
    PALETTE_POP, FIGURE_DPI, FIGURE_FORMAT,
)


# ── 1. Cargar cohorte C2 ──────────────────────────────────────────────────────

def load_c2() -> pd.DataFrame:
    """Carga el parquet generado por 01_build_cohort_c2.py."""
    c2 = pd.read_parquet(COHORT_C2_FILE)
    print(f"[load_c2] n pacientes: {c2[PATIENT_ID].nunique()}")
    print(f"[load_c2] n filas (visitas): {len(c2)}")
    return c2


# ── 2. Tabla descriptiva de ESSDAI por intervalo ──────────────────────────────

def table_essdai_by_interval(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada intervalo (v1–v6), calcular:
    - n pacientes con ESSDAI no nulo
    - mediana, IQR (p25, p75), min, max
    - % con ESSDAI ≥5 (Pop 1 / moderate-severe)
    - % con ESSDAI ≥14 (alta actividad)

    TAREA PARA CODEX:
    - Agrupar por INTERVAL_COL, filtrar filas con ESSDAI_TOTAL no nulo.
    - Usar .agg() con funciones lambda para percentiles.
    - Añadir columna 'pct_severe' = mean(ESSDAI_TOTAL >= ESSDAI_SEVERE) * 100
    - Añadir columna 'n_total_c2' = n pacientes únicos total en C2 (denominador
      para % cobertura del intervalo).
    - Ordenar por orden canónico de intervalos.

    OUTPUT: tabla lista para exportar a CSV y para construir Table 2 del manuscrito.
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 3. Figura: heatmap de cobertura ──────────────────────────────────────────

def fig_coverage_heatmap(c2: pd.DataFrame, save=True):
    """
    Heatmap binario: filas = pacientes (ordenados por n visitas ESSDAI),
    columnas = intervalos v1-v6. Color = tiene ESSDAI en ese intervalo (sí/no).

    TAREA PARA CODEX:
    - Crear pivot table: rows=PATIENT_ID, cols=INTERVAL_COL,
      values=ESSDAI_TOTAL, aggfunc='count' (luego binarizar a 0/1).
    - Reindexar columnas por INTERVAL_ORDER.
    - Ordenar filas por número total de visitas con ESSDAI (desc).
    - Usar sns.heatmap con cmap='YlOrRd', linewidths=0.3.
    - Anotar en el título: "n={n_c2} patients, C2 cohort".
    - Anotar en cada columna el n de pacientes con datos.
    - Tamaño: figsize=(8, max(4, n_patients * 0.08)).
    - Si hay más de 80 pacientes, suprimir ytick labels (demasiado denso).

    Guardar como: outputs/fig_01_coverage_heatmap.{FIGURE_FORMAT}
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # TODO: implementar pivot y heatmap

    if save:
        out = OUT_DIR / f"fig_01_coverage_heatmap.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 4. Figura: spaghetti plot (trayectorias individuales) ─────────────────────

def fig_spaghetti_essdai(c2: pd.DataFrame, save=True):
    """
    Líneas individuales de ESSDAI total vs. tiempo (time_from_anchor_yrs o
    interval_order si fecha no disponible), con líneas de referencia horizontales.

    TAREA PARA CODEX:
    - Separar pacientes en dos grupos: c2_subset_3plus (≥3 visitas, trazo sólido)
      vs. exactamente 2 visitas (trazo punteado).
    - Dibujar cada paciente como una línea gris semitransparente (alpha=0.35,
      linewidth=0.8), coloreada por ESSDAI basal ≥5 (rojo) vs <5 (azul).
    - Superponer mediana por intervalo como línea gruesa negra con marcadores.
    - Agregar bandas IQR como área sombreada (alpha=0.2).
    - Líneas de referencia horizontales: ESSDAI_SEVERE=5 (naranja, punteada)
      y ESSDAI_HIGH=14 (rojo, punteada).
    - Eje X: tiempo en años desde ancla (time_from_anchor_yrs). Si no disponible,
      usar tiempo nominal de INTERVAL_MONTHS.
    - Anotar n en riesgo por debajo del eje X (al estilo Kaplan-Meier).
    - Leyenda con n total, n en cada grupo de trayectoria.
    - Título: "Individual ESSDAI trajectories — C2 cohort (n={n})"

    NOTA: El eje X debe ser tiempo REAL (fechas), no intervalo ordinal, cuando
    las fechas están disponibles. Documentar en figura cuántos pacientes usan
    fechas reales vs. tiempo nominal.

    Guardar como: outputs/fig_02_spaghetti_essdai.{FIGURE_FORMAT}
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO: implementar spaghetti plot

    # Líneas de referencia
    # ax.axhline(ESSDAI_SEVERE, color='orange', linestyle='--', alpha=0.7,
    #            label=f'ESSDAI threshold (moderate-severe = {ESSDAI_SEVERE})')
    # ax.axhline(ESSDAI_HIGH, color='red', linestyle='--', alpha=0.5,
    #            label=f'High activity (ESSDAI = {ESSDAI_HIGH})')

    if save:
        out = OUT_DIR / f"fig_02_spaghetti_essdai.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 5. Figura: mediana + IQR por intervalo ────────────────────────────────────

def fig_median_trajectory(c2: pd.DataFrame, save=True):
    """
    Resumen de grupo de ESSDAI total por intervalo: mediana ± IQR (barras de error).

    TAREA PARA CODEX:
    - Calcular mediana, p25, p75 de ESSDAI_TOTAL por INTERVAL_COL (solo visitas
      con ESSDAI no nulo). Usar solo intervalos v1-v4 (v5/v6 tienen n muy bajo).
    - Plot de línea con barras de error (no barras de barra — usar errorbar).
    - Overlay de puntos individuales (jitter) para transparencia.
    - Anotar el n en cada punto ("n=XX").
    - Doble panel: panel izquierdo = C2 completo (n=71); panel derecho =
      subconjunto ≥3 visitas (n=36). Esto ilustra el sesgo potencial de
      selección por completeness.
    - Colorear intervalo v5/v6 en gris para indicar n muy bajo.

    Guardar como: outputs/fig_03_median_trajectory.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_03_median_trajectory.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 6. Figura: heatmap de actividad por dominio ───────────────────────────────

def fig_domain_heatmap(c2: pd.DataFrame, save=True):
    """
    Heatmap: filas = 12 dominios ESSDAI, columnas = intervalos v1-v6.
    Valor = % de pacientes con actividad en ese dominio (nivel ≥1 o ≥2 según dominio).

    TAREA PARA CODEX:
    - Para cada dominio en ESSDAI_DOMAINS, calcular proporción de pacientes
      con valor > 0 (o nivel ≥1 en categóricos) por intervalo.
    - Solo incluir pacientes con ESSDAI_TOTAL no nulo en ese intervalo
      (denominador correcto).
    - Ordenar dominios de mayor a menor prevalencia (columna v1/v2 como referencia).
    - Resaltar ACTIVE_DOMAINS con asterisco en la etiqueta de fila.
    - Usar paleta divergente: 0% = blanco, 100% = rojo oscuro.
    - Añadir anotaciones de texto en cada celda con el %.
    - Columnas v5/v6 con n muy bajo: añadir nota "(n<10)".

    Guardar como: outputs/fig_04_domain_heatmap.{FIGURE_FORMAT}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_04_domain_heatmap.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 7. Figura: distribución de ESSDAI por intervalo (violín/boxplot) ──────────

def fig_essdai_distribution(c2: pd.DataFrame, save=True):
    """
    Violin plot de ESSDAI total por intervalo, con boxplot interno.

    TAREA PARA CODEX:
    - Usar sns.violinplot con inner='box', palette='muted'.
    - Restringir a v1-v4 (v5/v6 tienen n insuficiente para violin).
    - Superponer puntos individuales (jitter) con alpha=0.5.
    - Línea de referencia horizontal en ESSDAI_SEVERE=5.
    - Anotar n por intervalo.
    - Dividir figura en dos filas: arriba = C2 completo, abajo = ≥3 visitas.

    Guardar como: outputs/fig_05_essdai_distribution.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_05_essdai_distribution.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 8. Tabla: cambio dentro-paciente entre visitas consecutivas ───────────────

def table_within_patient_change(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par de visitas consecutivas (v1→v2, v2→v3, v3→v4, v4→v5),
    calcular: n pares, mediana del cambio (ESSDAI_v2 - ESSDAI_v1),
    IQR del cambio, % con mejora (cambio < 0), % con estabilidad (cambio = 0),
    % con empeoramiento (cambio > 0), % que transitan de <5 a ≥5 (incidente severo).

    TAREA PARA CODEX:
    - Para cada paciente, ordenar visitas por interval_order.
    - Calcular diff() del ESSDAI_TOTAL entre visitas consecutivas NO nulas.
    - Agrupar por par de intervalos y agregar las métricas listadas.
    - Importante: usar solo pares donde AMBAS visitas tienen ESSDAI no nulo
      (no imputar). Documentar el n de pares válidos.
    - Esta tabla es la base para el análisis de "cobertura de pares consecutivos"
      que el var_summary reportó en 29.9%.

    NOTA: cobertura de pares consecutivos ≈30% implica que muchos pacientes tienen
    gaps entre visitas. Documentar el patrón de missingness: ¿es aleatorio
    (MCAR) o estructural (pacientes con más severidad tienen más visitas)?
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 9. QC: verificar consistencia ESSDAI total vs suma de dominios ────────────

def qc_essdai_total_vs_domains(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Verificar si essdai-_r__essdai_total_score es consistente con la suma
    ponderada de los dominios individuales (usando ESSDAI_DOMAIN_WEIGHTS).

    TAREA PARA CODEX:
    - Recalcular el ESSDAI total como suma de (domain_score × weight) para
      cada visita donde todos los dominios están disponibles.
    - Comparar con essdai-_r__essdai_total_score.
    - Reportar: % de visitas con discrepancia >0, >2, >5 puntos.
    - Si la discrepancia es sistemática, documentar y reportar al investigador
      antes de continuar con el análisis.
    - Las discrepancias pueden indicar: versión de encodings diferente,
      dominios que usan subescalas que no están en el dataset, o errores de
      recodificación.

    IMPORTANTE: Los dominios con tipo 'boolean' en el dataset pueden haber sido
    binarizados (0/1) sin los niveles de actividad intermedios del ESSDAI original.
    Esto afecta la reconstrucción del total. Documentar qué dominios son boolean
    vs. categorical en los datos.
    """
    # TODO: implementar QC
    qc_tab = pd.DataFrame()
    return qc_tab


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("02_eda_descriptive.py — EDA longitudinal ESSDAI")
    print("=" * 60)

    c2 = load_c2()

    # Tablas descriptivas
    tab_interval = table_essdai_by_interval(c2)
    tab_interval.to_csv(OUT_DIR / "tab_02_essdai_by_interval.csv", index=False)

    tab_domain = pd.DataFrame()  # placeholder para tabla de dominios
    tab_domain.to_csv(OUT_DIR / "tab_02_domain_activity.csv", index=False)

    tab_change = table_within_patient_change(c2)
    tab_change.to_csv(OUT_DIR / "tab_02_within_patient_change.csv", index=False)

    # QC
    qc_tab = qc_essdai_total_vs_domains(c2)
    qc_tab.to_csv(OUT_DIR / "qc_essdai_domain_consistency.csv", index=False)

    # Figuras
    fig_coverage_heatmap(c2)
    fig_spaghetti_essdai(c2)
    fig_median_trajectory(c2)
    fig_domain_heatmap(c2)
    fig_essdai_distribution(c2)

    print("\n✓ EDA completado. Ver outputs/ para figuras y tablas.")


if __name__ == "__main__":
    main()
