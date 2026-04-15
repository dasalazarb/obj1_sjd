"""
05_severity_strata.py — Clasificación Pop 1-3 y Transiciones de Severidad
==========================================================================

OBJETIVO:
    Clasificar a los pacientes de C2 en las subpoblaciones del OASIZ trial
    (Pop 1-3) y describir su distribución y dinámica a lo largo del tiempo.

DEFINICIONES (uno.docx §7.2 / protocolo):
    Pop 1: ESSDAI ≥ 5 (cualquier ESSPRI) → actividad moderada a severa
    Pop 2: ESSPRI ≥ 5 AND ESSDAI < 5    → inactiva/leve, síntomas significativos
    Pop 3: ESSDAI < 5 AND ESSPRI < 5    → leve, sin síntomas significativos
    Pop 4: sin enfermedad (comparador)   → NO incluir en análisis C2

RESTRICCIONES CRÍTICAS (del informe de viabilidad §6.7):
    - El n emparejado ESSDAI + ESSPRI tiene techo ≈ 71 pacientes.
    - ESSPRI tiene change_rate 70-80% entre visitas → mucho ruido.
    - Los modelos multi-state de tiempo continuo NO son ejecutables.
    - Las "transiciones" se reportan como DESCRIPTIVAS, no como estimaciones
      de tasas de transición confirmatorias.
    - Presentar como "distribución de fenotipo en cada visita", no como
      "modelo de transición de fenotipo".

OUTPUTS:
    outputs/tab_05_pop_distribution.csv    — distribución Pop 1-3 por intervalo
    outputs/tab_05_pop_transitions.csv     — tabla de transiciones descriptiva
    outputs/fig_10_pop_sankey.html         — diagrama Sankey (Plotly)
    outputs/fig_10_pop_alluvial.png        — alluvial plot (matplotlib)
    outputs/fig_11_essdai_esspri_scatter.png — scatter ESSDAI vs ESSPRI por intervalo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings

from config import (
    COHORT_C2_FILE, OUT_DIR,
    PATIENT_ID, INTERVAL_COL,
    ESSDAI_TOTAL, ESSPRI_ITEMS, ESSPRI_MAIN_ITEMS,
    ESSDAI_SEVERE, ESSPRI_THRESHOLD,
    POP_LABELS, PALETTE_POP,
    INTERVAL_ORDER, INTERVAL_MONTHS,
    FIGURE_DPI, FIGURE_FORMAT,
)


# ── 1. Derivar ESSPRI total y clasificación Pop ───────────────────────────────

def classify_populations(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva:
    1. 'esspri_total': media de dryness, fatigue, pain (si ≥2 de 3 no nulos).
    2. 'pop_class': 1, 2, 3 o NaN (si falta ESSDAI O ESSPRI para clasificar).

    Reglas de clasificación:
    - Si ESSDAI_TOTAL >= ESSDAI_SEVERE → Pop 1 (independiente de ESSPRI)
    - Si ESSDAI_TOTAL < ESSDAI_SEVERE AND esspri_total >= ESSPRI_THRESHOLD → Pop 2
    - Si ESSDAI_TOTAL < ESSDAI_SEVERE AND esspri_total < ESSPRI_THRESHOLD → Pop 3
    - Si ESSDAI_TOTAL es nulo → NaN (visita sin ESSDAI evaluable)
    - Si ESSDAI_TOTAL < ESSDAI_SEVERE pero ESSPRI nulo → 'ESSDAI_only' (subgrupo)
      Reportar este subgrupo por separado; no asumir que es Pop 2 o Pop 3.

    TAREA PARA CODEX:
    - Calcular 'esspri_total' usando las columnas en ESSPRI_ITEMS para los
      ítems en ESSPRI_MAIN_ITEMS (dryness, fatigue, pain). Media si ≥2 no nulos.
    - Aplicar la lógica de clasificación con np.select o pd.cut.
    - Reportar: distribución de pop_class en cada intervalo y proporción de NaN.
    - Verificar valores únicos reales de las columnas ESSPRI antes de calcular
      la media (pueden ser texto, 0-10 numérico, o codificados de otra manera).

    RETORNA: c2 con columnas 'esspri_total' y 'pop_class' añadidas.
    """
    c2 = c2.copy()
    # TODO: implementar clasificación
    return c2


# ── 2. Tabla de distribución Pop por intervalo ────────────────────────────────

def table_pop_distribution(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada intervalo en INTERVAL_ORDER, calcular:
    - n y % en Pop 1, Pop 2, Pop 3
    - n con ESSDAI pero sin ESSPRI ('ESSDAI_only', no clasificables en Pop 2/3)
    - n sin ESSDAI ('no data')
    - IC95% de cada proporción (Wilson)

    TAREA PARA CODEX:
    - Agrupar por INTERVAL_COL, filtrar a pacientes de C2.
    - La base para los porcentajes debe ser: n con ESSDAI no nulo en ese intervalo
      (no el n total de C2). Reportar ambos denominadores.
    - Añadir fila 'Overall (all visits pooled)' al final como referencia.
    - Advertencia: como los datos son longitudinales, la "distribución pooled"
      cuenta visitas, no pacientes. Reportar esto explícitamente.

    NOTA CLÍNICA IMPORTANTE:
    La prevalencia de Pop 1 (ESSDAI≥5) en v1 es la estimación clave del
    protocolo. El informe de viabilidad estima que a basal ~60-70% de los
    pacientes de 11-D pueden ser Pop 1 (cohorte sesgada hacia enfermedad activa).
    Verificar este número contra los datos reales y reportarlo como primer
    hallazgo descriptivo del manuscrito.
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 3. Tabla de transiciones entre visitas consecutivas ──────────────────────

def table_pop_transitions(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada par de visitas consecutivas, calcular la matriz de transición
    (origen Pop → destino Pop).

    Formato output: tabla con columnas
    [interval_pair, origin_pop, dest_pop, n, pct_of_origin]

    Ejemplo: v1→v2: Pop1→Pop1: n=20 (80%), Pop1→Pop2: n=3 (12%), ...

    TAREA PARA CODEX:
    - Para cada paciente, unir visita k con visita k+1 (solo cuando ambas
      tienen pop_class no nulo).
    - Calcular crosstab(origin_pop, dest_pop) por par de intervalos.
    - Calcular % de cada fila (origen) como denominador.
    - Calcular también: % que transitan de no-severo a severo (Pop2/3 → Pop1).
      Este es el "incidente severo" para el Objetivo Primario 4.
    - Reportar n de pares válidos para cada transición.

    ADVERTENCIA: Solo reportar transiciones con ≥5 eventos en la celda.
    Celdas con <5 → suprimir y reportar como "<5" para proteger privacidad
    y evitar estimaciones inestables.
    """
    # TODO: implementar
    tab = pd.DataFrame()
    return tab


# ── 4. Figura: alluvial / Sankey ──────────────────────────────────────────────

def fig_alluvial_pop(c2: pd.DataFrame, save=True):
    """
    Diagrama alluvial mostrando flujo de pacientes entre Pop 1-3 a lo largo
    del tiempo (v1 → v2 → v3 → v4).

    OPCIÓN A (matplotlib — sin dependencias extra):
    - Implementar alluvial básico con barras rectangulares por intervalo
      y curvas Bezier conectando los estratos entre intervalos.
    - Usar PALETTE_POP para colores.
    - Ancho de cada bloque proporcional al n de pacientes en esa Pop.
    - Solo incluir pacientes con ≥2 pop_class consecutivos no nulos.
    - Anotar n en cada bloque.

    OPCIÓN B (Plotly — HTML interactivo):
    - Usar plotly.graph_objects.Sankey.
    - Exportar como HTML en outputs/fig_10_pop_sankey.html.

    TAREA PARA CODEX:
    - Implementar Opción B (Plotly) como primaria (HTML más rico para exploración).
    - Implementar Opción A (matplotlib) como secundaria para el manuscrito.
    - Para Opción B: definir nodes y links desde la tabla de transiciones.
      Cada nodo = (pop_class, interval). Cada link = transición entre nodos.

    ADVERTENCIA: Si hay celdas con <5 eventos, no graficar esa transición
    (misma regla de supresión que en la tabla).
    - Solo mostrar intervalos v1-v4 (v5/v6 tienen n insuficiente para alluvial).

    Guardar como:
        outputs/fig_10_pop_sankey.html      (Plotly)
        outputs/fig_10_pop_alluvial.{FORMAT} (matplotlib)
    """
    # OPCIÓN A: matplotlib alluvial
    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO: implementar alluvial matplotlib

    if save:
        out = OUT_DIR / f"fig_10_pop_alluvial.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")

    # TODO: OPCIÓN B — Plotly Sankey (exportar HTML separado)

    plt.close(fig)


# ── 5. Figura: scatter ESSDAI vs ESSPRI ──────────────────────────────────────

def fig_essdai_esspri_scatter(c2: pd.DataFrame, save=True):
    """
    Scatter plot ESSDAI total (eje X) vs ESSPRI total (eje Y).
    Colorear puntos por Pop 1-3. Mostrar cuadrantes definidos por los
    umbrales ESSDAI_SEVERE=5 y ESSPRI_THRESHOLD=5.

    TAREA PARA CODEX:
    - Usar solo visitas con AMBOS ESSDAI y ESSPRI no nulos.
    - Dibujar líneas de cuadrante: x=5 (naranja punteado) y y=5 (azul punteado).
    - Anotar las cuatro esquinas con las etiquetas: Pop 1, Pop 2, Pop 3.
    - Superponer convex hull de cada subpoblación (opcional, si n suficiente).
    - Separar en facets por intervalo (v1, v2, v3, v4) en una grilla 2×2.
    - Anotar n de cada Pop en cada panel.
    - Incluir texto de advertencia sobre el alto ruido de ESSPRI entre visitas.

    Guardar como: outputs/fig_11_essdai_esspri_scatter.{FIGURE_FORMAT}
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    # TODO: implementar

    if save:
        out = OUT_DIR / f"fig_11_essdai_esspri_scatter.{FIGURE_FORMAT}"
        fig.savefig(out, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Guardado: {out}")
    plt.close(fig)


# ── 6. Estadística descriptiva de cambio en severidad ────────────────────────

def describe_severity_change(c2: pd.DataFrame) -> dict:
    """
    Calcular:
    - Proporción de pacientes con ESSDAI reducido ≥3 puntos entre v1 y v2
      (Minimal Clinically Important Difference, MCID ≈ 3 en literatura SjD).
    - Proporción que pasan de ESSDAI≥5 (Pop 1) a ESSDAI<5 (Pop 2/3) → mejora.
    - Proporción que pasan de ESSDAI<5 a ≥5 → empeoramiento (incidente severo).
    - Mediana del tiempo entre la primera visita ESSDAI<5 y el primer ESSDAI≥5
      (solo en pacientes con ese patrón; este es el análogo de Kaplan-Meier
      del Objetivo Primario 4, versión descriptiva simple).

    TAREA PARA CODEX:
    - Calcular para el par v1→v2 (mayor n de pares consecutivos).
    - Reportar todo con IC95% exacto (binomial de Wilson para proporciones,
      o bootstrap para la mediana de tiempo).
    - Recordar: el "tiempo entre visitas" puede ser de semanas a años dependiendo
      del paciente. Usar time_from_anchor_yrs, no interval_order.

    RETORNA: dict con todas las estadísticas de cambio.
    """
    # TODO: implementar
    stats = {}
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("05_severity_strata.py — Pop 1-3 y Transiciones")
    print("=" * 60)

    c2 = pd.read_parquet(COHORT_C2_FILE)

    # Clasificar
    c2 = classify_populations(c2)

    # Tablas
    tab_dist = table_pop_distribution(c2)
    tab_dist.to_csv(OUT_DIR / "tab_05_pop_distribution.csv", index=False)

    tab_trans = table_pop_transitions(c2)
    tab_trans.to_csv(OUT_DIR / "tab_05_pop_transitions.csv", index=False)

    # Estadísticas de cambio de severidad
    severity_stats = describe_severity_change(c2)
    pd.DataFrame([severity_stats]).to_csv(
        OUT_DIR / "tab_05_severity_change.csv", index=False
    )

    # Guardar c2 enriquecido con pop_class para uso en §06
    c2.to_parquet(OUT_DIR / "cohort_c2_with_pop.parquet", index=False)

    # Figuras
    fig_alluvial_pop(c2)
    fig_essdai_esspri_scatter(c2)

    print("\n✓ Análisis Pop 1-3 completado. Ver outputs/")


if __name__ == "__main__":
    main()
