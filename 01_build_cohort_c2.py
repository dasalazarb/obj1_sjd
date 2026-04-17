"""
01_build_cohort_c2.py — Construcción de la Cohorte C2 (ESSDAI Longitudinal)
=============================================================================

OBJETIVO:
    Cargar el dataset (1 fila = paciente × fase de protocolo), aplicar los
    criterios de elegibilidad de la cohorte C2, derivar variables de tiempo
    por paciente, y exportar un parquet en formato LONG listo para análisis.

CRITERIO C2:
    - Pacientes con SjD documentada (sjogrens_class no nulo / no control)
    - ≥2 visitas con essdai-_r__essdai_total_score no nulo

OUTPUT:
    outputs/cohort_c2.parquet
    outputs/cohort_c2_exclusions.csv
    outputs/cohort_c2_demographics.csv
    outputs/qc_time_source.csv           — visitas por tipo de fuente de tiempo
    outputs/qc_sjogrens_class_changes.csv — pacientes con clase SjD inconsistente
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
    PHASE_LABELS, INTERVAL_MONTHS_NOMINAL,
    C2_MIN_ESSDAI_VISITS, C2_MIN_ESSDAI_3_VISITS,
    ESSDAI_SEVERE,
)

# ---------------------------------------------------------------------------
# Valores de SJOGRENS_CLASS que califican como SjD (case-insensitive strip)
# AJUSTAR si los valores reales del dataset difieren — verificar con load_wide()
# ---------------------------------------------------------------------------
SJD_CLASS_INCLUDE = {
    "primary sjogrens", "secondary sjogrens",
    "primary", "secondary",
    "sjd", "sjd primary", "sjd secondary",
    "primary sjd", "secondary sjd",
    "1", "2",
}

SJD_CLASS_EXCLUDE = {
    "healthy volunteer", "hv",
    "non-sjd", "non sjd",
    "excluded", "control",
    "no disease",
}


# ── 1. Cargar dataset ─────────────────────────────────────────────────────────

def load_wide(filepath: Path) -> pd.DataFrame:
    """Carga el CSV y convierte tipos en columnas clave.

    Intenta UTF-8 primero y, si falla por codificación, prueba Latin-1 para
    archivos exportados con codificaciones legacy.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
        encoding_used = "utf-8"
    except UnicodeDecodeError as exc:
        warnings.warn(
            "[load_wide] No se pudo leer con UTF-8; se reintenta con latin-1. "
            f"Detalle: {exc}"
        )
        df = pd.read_csv(filepath, low_memory=False, encoding="latin-1")
        encoding_used = "latin-1"

    # Fechas
    df[VISIT_DATE]  = pd.to_datetime(df[VISIT_DATE],  errors="coerce")
    df[DX_DATE_COL] = pd.to_datetime(df[DX_DATE_COL], errors="coerce")

    # ESSDAI total → numérico
    df[ESSDAI_TOTAL] = pd.to_numeric(df[ESSDAI_TOTAL], errors="coerce")

    # Dominios ESSDAI → numérico
    for col in ESSDAI_DOMAINS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ESSPRI → numérico
    for col in ESSPRI_ITEMS.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Edad → numérico
    if AGE_COL in df.columns:
        df[AGE_COL] = pd.to_numeric(df[AGE_COL], errors="coerce")

    # INTERVAL_COL → string limpio
    df[INTERVAL_COL] = df[INTERVAL_COL].astype(str).str.strip()

    # Reporte de nulos en columnas clave
    key_cols = [c for c in [PATIENT_ID, INTERVAL_COL, VISIT_DATE, ESSDAI_TOTAL,
                             SJOGRENS_CLASS, DX_DATE_COL, AGE_COL, SEX_COL]
                if c in df.columns]
    null_pct = df[key_cols].isna().mean().mul(100).round(1)

    print(f"\n[load_wide] Encoding usado: {encoding_used}")
    print(f"[load_wide] Shape: {df.shape}")
    print(f"[load_wide] Pacientes únicos: {df[PATIENT_ID].nunique()}")
    print(f"[load_wide] Fases únicas:\n  " +
          "\n  ".join(sorted(df[INTERVAL_COL].unique())))
    print(f"\n[load_wide] % nulos en columnas clave:")
    for col, pct in null_pct.items():
        print(f"  {col:60s}: {pct:.1f}%")

    return df


# ── 2. Transformar wide → long ────────────────────────────────────────────────

def wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    El CSV de entrada tiene 2057 columnas (una por variable del CRF) y
    486 filas (una por paciente × fase de protocolo). Ya está en formato
    pseudo-long, pero necesita:

      1. Reducirse a las columnas analíticamente relevantes (de 2057 → ~40).
      2. Añadir 'phase_order' (entero ordinal 0-6) y 'phase_label' (etiqueta
         corta) mapeados desde PHASE_LABELS.
      3. Ordenarse por (PATIENT_ID, phase_order) para que todas las
         operaciones de ventana (diff, cummin, groupby.first) sean correctas.

    El resultado es el dataframe 'long' que usan todos los pasos siguientes.

    Columnas añadidas:
        phase_order  — entero 0-6 (ordinal de fase); NaN si fase no reconocida
        phase_label  — etiqueta corta, p.ej. 'V2 (Initial)'
    """

    # ── 2a. Seleccionar columnas relevantes ───────────────────────────────────
    desired = (
        [PATIENT_ID, INTERVAL_COL, VISIT_DATE, AGE_COL, SEX_COL,
         RACE_COL, ETHNICITY_COL, SJOGRENS_CLASS, DX_DATE_COL, ESSDAI_TOTAL]
        + list(ESSDAI_DOMAINS.values())
        + list(ESSPRI_ITEMS.values())
    )
    missing = [c for c in desired if c not in df.columns]
    if missing:
        warnings.warn(
            f"[wide_to_long] Las siguientes columnas deseadas no están en el "
            f"dataset y se omiten:\n  {missing}"
        )
    present = [c for c in desired if c in df.columns]
    long = df[present].copy()

    print(f"\n[wide_to_long] Columnas reducidas: {df.shape[1]} → {long.shape[1]}")
    print(f"[wide_to_long] Filas (paciente × fase): {long.shape[0]}")

    # ── 2b. Añadir phase_order y phase_label ─────────────────────────────────
    phase_order_map = {phase: i for i, phase in enumerate(PHASE_LABELS.keys())}

    long["phase_order"] = long[INTERVAL_COL].map(phase_order_map)
    long["phase_label"] = long[INTERVAL_COL].map(PHASE_LABELS)

    unrecognized = long[long["phase_order"].isna()][INTERVAL_COL].unique()
    if len(unrecognized) > 0:
        warnings.warn(
            f"[wide_to_long] Fases no reconocidas en PHASE_LABELS "
            f"(phase_order=NaN): {unrecognized}\n"
            f"  Si corresponden a visitas reales, agregarlas a PHASE_LABELS en config.py."
        )

    # ── 2c. Ordenar por paciente y fase ───────────────────────────────────────
    long = long.sort_values([PATIENT_ID, "phase_order"], na_position="last")
    long = long.reset_index(drop=True)

    print(f"[wide_to_long] Fases reconocidas: "
          f"{long['phase_order'].notna().sum()} / {len(long)} filas")
    print(f"[wide_to_long] Distribución de filas por fase:")
    phase_counts = long.groupby("phase_label", dropna=False).size()
    for label, n in phase_counts.items():
        print(f"  {str(label):35s}: {n} filas")

    return long


# ── 4. Derivar ancla por paciente y tiempo relativo ──────────────────────────

def derive_time_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ANCLA = primera visita con ESSDAI no nulo y fecha de visita no nula.
    Si ninguna visita ESSDAI tiene fecha, la ancla es la de menor phase_order
    entre las visitas con ESSDAI (ancla nominal).

    time_source por fila:
        'date_actual'        — visit_date real disponible + anchor_date real
        'phase_nominal'      — visita sin fecha; usa diferencia de phase_order
        'visit_no_anchor'    — visita tiene fecha pero el anchor del paciente es nominal
        'no_anchor'          — paciente sin ninguna visita ESSDAI evaluable

    Retorna: (df_enriquecido, qc_time_source_df)
    """
    df = df.copy()

    has_essdai = df[ESSDAI_TOTAL].notna()
    has_date   = df[VISIT_DATE].notna()

    # ── 4a. Ancla con fecha real ──────────────────────────────────────────────
    anchor_real = (
        df[has_essdai & has_date]
        .sort_values([PATIENT_ID, VISIT_DATE])
        .groupby(PATIENT_ID, sort=False)
        .first()[[VISIT_DATE, INTERVAL_COL, "phase_order"]]
        .rename(columns={
            VISIT_DATE:    "anchor_date",
            INTERVAL_COL:  "anchor_phase",
            "phase_order": "anchor_phase_order",
        })
    )

    # ── 4b. Ancla nominal (fallback: pacientes sin fecha en visitas ESSDAI) ───
    anchor_nominal = (
        df[has_essdai]
        .sort_values([PATIENT_ID, "phase_order"])
        .groupby(PATIENT_ID, sort=False)
        .first()[["phase_order"]]
        .rename(columns={"phase_order": "anchor_phase_order_nominal"})
    )

    df = df.join(anchor_real,    on=PATIENT_ID)
    df = df.join(anchor_nominal, on=PATIENT_ID)

    # ── 4c. time_from_anchor ─────────────────────────────────────────────────
    df["time_from_anchor_days"] = np.nan
    df["time_from_anchor_yrs"]  = np.nan
    df["time_source"] = "no_anchor"

    # Caso 1: visita con fecha + ancla con fecha → tiempo real
    real_mask = has_date & df["anchor_date"].notna()
    df.loc[real_mask, "time_from_anchor_days"] = (
        df.loc[real_mask, VISIT_DATE] - df.loc[real_mask, "anchor_date"]
    ).dt.days.astype(float)
    df.loc[real_mask, "time_from_anchor_yrs"] = (
        df.loc[real_mask, "time_from_anchor_days"] / 365.25
    )
    df.loc[real_mask, "time_source"] = "date_actual"

    # Caso 2: visita sin fecha → tiempo nominal (diferencia de phase_order)
    nominal_mask = ~has_date & df["anchor_phase_order_nominal"].notna()
    df.loc[nominal_mask, "time_from_anchor_yrs"] = (
        df.loc[nominal_mask, "phase_order"]
        - df.loc[nominal_mask, "anchor_phase_order_nominal"]
    ).astype(float)
    df.loc[nominal_mask, "time_source"] = "phase_nominal"

    # Caso 3: visita tiene fecha pero el ancla del paciente es nominal
    mixed_mask = has_date & df["anchor_date"].isna() & df["anchor_phase_order_nominal"].notna()
    df.loc[mixed_mask, "time_source"] = "visit_date_no_anchor"

    # QC: resumen de time_source para visitas con ESSDAI
    qc_time = (
        df[has_essdai]
        .groupby(["time_source"])
        .agg(
            n_visits  = (PATIENT_ID, "count"),
            n_patients= (PATIENT_ID, "nunique"),
        )
        .reset_index()
    )

    # ── 4d. disease_duration_yrs ─────────────────────────────────────────────
    dx_ok = has_date & df[DX_DATE_COL].notna()
    df["disease_duration_yrs"] = np.nan
    df.loc[dx_ok, "disease_duration_yrs"] = (
        df.loc[dx_ok, VISIT_DATE] - df.loc[dx_ok, DX_DATE_COL]
    ).dt.days / 365.25

    # Alertar y capear negativos
    neg = df["disease_duration_yrs"].notna() & (df["disease_duration_yrs"] < 0)
    if neg.any():
        n_neg_pats = df.loc[neg, PATIENT_ID].nunique()
        warnings.warn(
            f"[derive_time_variables] {n_neg_pats} paciente(s) con "
            f"disease_duration_yrs < 0 (visita anterior al diagnóstico formal). "
            f"Se conserva el valor original; se crea 'disease_duration_yrs_model' capeado en 0."
        )
        neg_detail = (
            df.loc[neg, [PATIENT_ID, VISIT_DATE, DX_DATE_COL, "disease_duration_yrs"]]
            .drop_duplicates(PATIENT_ID)
        )
        print(f"\n[WARN] Detalle disease_duration_yrs < 0:\n{neg_detail.to_string()}")

    df["disease_duration_yrs_model"] = df["disease_duration_yrs"].clip(lower=0)

    # ── 4e. Variables basales por paciente (desde la visita ancla) ────────────
    # Ancla = fila con time_from_anchor_yrs mínimo (>=0) por paciente
    anchor_idx = (
        df[df["time_from_anchor_yrs"].notna() & (df["time_from_anchor_yrs"] >= 0)]
        .groupby(PATIENT_ID)["time_from_anchor_yrs"]
        .idxmin()
    )
    anchor_rows = df.loc[anchor_idx, [PATIENT_ID, ESSDAI_TOTAL, AGE_COL]].copy()
    anchor_rows = anchor_rows.rename(columns={
        ESSDAI_TOTAL: "essdai_baseline",
        AGE_COL:      "age_at_baseline",
    })
    df = df.merge(anchor_rows, on=PATIENT_ID, how="left")

    # ── 4f. sex_female ────────────────────────────────────────────────────────
    if SEX_COL in df.columns:
        sex_vals = df[SEX_COL].dropna().unique()
        print(f"\n[derive_time_variables] Valores únicos de {SEX_COL}: {sex_vals}")

        sex_str = df[SEX_COL].astype(str).str.strip().str.lower()
        df["sex_female"] = np.nan
        df.loc[sex_str.isin({"female", "f", "woman", "2", "mujer"}), "sex_female"] = 1.0
        df.loc[sex_str.isin({"male",   "m", "man",   "1", "hombre"}), "sex_female"] = 0.0

        unmapped = df[df["sex_female"].isna() & df[SEX_COL].notna()][SEX_COL].unique()
        if len(unmapped) > 0:
            warnings.warn(
                f"[derive_time_variables] Valores de {SEX_COL} no mapeados a "
                f"sex_female (quedan NaN): {unmapped}"
            )

    # ── 4g. sjogrens_class_primary ────────────────────────────────────────────
    if SJOGRENS_CLASS in df.columns:
        class_vals = df[SJOGRENS_CLASS].dropna().unique()
        print(f"[derive_time_variables] Valores únicos de {SJOGRENS_CLASS}: {class_vals}")

        class_str = df[SJOGRENS_CLASS].astype(str).str.strip().str.lower()
        df["sjogrens_class_primary"] = np.nan
        df.loc[class_str.isin(
            {"primary sjogrens", "primary", "1", "sjd primary", "primary sjd"}
        ), "sjogrens_class_primary"] = 1.0
        df.loc[class_str.isin(
            {"secondary sjogrens", "secondary", "2", "sjd secondary", "secondary sjd"}
        ), "sjogrens_class_primary"] = 0.0

        # QC: pacientes con clase inconsistente entre visitas
        n_classes = (
            df[df[SJOGRENS_CLASS].notna()]
            .groupby(PATIENT_ID)[SJOGRENS_CLASS]
            .nunique()
        )
        inconsistent = n_classes[n_classes > 1].index.tolist()
        if inconsistent:
            warnings.warn(
                f"[derive_time_variables] {len(inconsistent)} paciente(s) con "
                f"{SJOGRENS_CLASS} inconsistente entre visitas."
            )
            qc_class = df[df[PATIENT_ID].isin(inconsistent)][
                [PATIENT_ID, INTERVAL_COL, SJOGRENS_CLASS, VISIT_DATE]
            ].sort_values([PATIENT_ID, "phase_order"])
            qc_class.to_csv(OUT_DIR / "qc_sjogrens_class_changes.csv", index=False)
            print(f"  → guardado: {OUT_DIR}/qc_sjogrens_class_changes.csv")

    # ── 4h. ESSPRI total ──────────────────────────────────────────────────────
    esspri_cols = [
        ESSPRI_ITEMS[k] for k in ESSPRI_MAIN_ITEMS
        if k in ESSPRI_ITEMS and ESSPRI_ITEMS[k] in df.columns
    ]
    if len(esspri_cols) >= 2:
        esspri_data = df[esspri_cols]
        n_valid = esspri_data.notna().sum(axis=1)
        df["esspri_total"] = np.where(
            n_valid >= 2,
            esspri_data.mean(axis=1, skipna=True),
            np.nan,
        )
    else:
        df["esspri_total"] = np.nan
        warnings.warn(
            f"[derive_time_variables] Menos de 2 columnas ESSPRI disponibles — "
            f"esspri_total = NaN en todas las filas."
        )

    # ── 4i. n_essdai_visits_total y c2_subset_3plus ───────────────────────────
    n_visits = (
        df[df[ESSDAI_TOTAL].notna()]
        .groupby(PATIENT_ID)
        .size()
        .rename("n_essdai_visits_total")
    )
    df = df.join(n_visits, on=PATIENT_ID)
    df["n_essdai_visits_total"] = df["n_essdai_visits_total"].fillna(0).astype(int)
    df["c2_subset_3plus"] = df["n_essdai_visits_total"] >= C2_MIN_ESSDAI_3_VISITS

    return df, qc_time


# ── 5. Aplicar criterios de elegibilidad C2 ───────────────────────────────────

def build_c2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica criterios de inclusión/exclusión.
    Una vez que el paciente califica, se incluyen TODAS sus filas
    (incluyendo visitas sin ESSDAI) para preservar el contexto temporal completo.
    """
    exclusion_log = []

    for pid in df[PATIENT_ID].unique():
        pat = df[df[PATIENT_ID] == pid]

        # Criterio 1: SjD documentada
        class_str = (
            pat[SJOGRENS_CLASS].dropna()
            .astype(str).str.strip().str.lower()
            .unique()
        )
        has_sjd      = any(v in SJD_CLASS_INCLUDE for v in class_str)
        is_excluded  = any(v in SJD_CLASS_EXCLUDE for v in class_str)
        n_essdai     = int(pat[ESSDAI_TOTAL].notna().sum())

        if is_excluded or len(class_str) == 0:
            reason = (
                "Excluded class (HV/non-SjD/control)"
                if is_excluded else "No SjD class documented"
            )
            exclusion_log.append({
                PATIENT_ID:           pid,
                "n_essdai_visits":    n_essdai,
                "sjogrens_class_obs": str(class_str.tolist()),
                "exclusion_reason":   reason,
            })
            continue

        if not has_sjd:
            exclusion_log.append({
                PATIENT_ID:           pid,
                "n_essdai_visits":    n_essdai,
                "sjogrens_class_obs": str(class_str.tolist()),
                "exclusion_reason":   (
                    "SjD class value not in SJD_CLASS_INCLUDE — "
                    "verify mapping in script header"
                ),
            })
            continue

        # Criterio 2: ≥2 visitas ESSDAI
        if n_essdai < C2_MIN_ESSDAI_VISITS:
            exclusion_log.append({
                PATIENT_ID:           pid,
                "n_essdai_visits":    n_essdai,
                "sjogrens_class_obs": str(class_str.tolist()),
                "exclusion_reason":   (
                    f"Only {n_essdai} evaluable ESSDAI visit(s) "
                    f"(need ≥{C2_MIN_ESSDAI_VISITS})"
                ),
            })

    excl_ids = {row[PATIENT_ID] for row in exclusion_log}
    c2 = df[~df[PATIENT_ID].isin(excl_ids)].copy()

    n_raw  = df[PATIENT_ID].nunique()
    n_c2   = c2[PATIENT_ID].nunique()
    n_3p   = c2[c2["c2_subset_3plus"]][PATIENT_ID].nunique()
    n_excl_class  = sum(1 for r in exclusion_log
                        if "class" in r["exclusion_reason"].lower())
    n_excl_essdai = sum(1 for r in exclusion_log
                        if "ESSDAI" in r["exclusion_reason"])

    print(f"\n[build_c2] ──────────────────────────────────")
    print(f"[build_c2] Pacientes raw:              {n_raw}")
    print(f"[build_c2] Excluidos total:            {len(excl_ids)}")
    print(f"[build_c2]   por clase SjD:            {n_excl_class}")
    print(f"[build_c2]   por <{C2_MIN_ESSDAI_VISITS} ESSDAI:             {n_excl_essdai}")
    print(f"[build_c2] Pacientes en C2:            {n_c2}  (esperado ≈71)")
    print(f"[build_c2] C2 subset ≥3 visitas:       {n_3p}  (esperado ≈36)")
    print(f"[build_c2] ──────────────────────────────────")

    if abs(n_c2 - 71) > 15:
        warnings.warn(
            f"\n[build_c2] ⚠ n C2 = {n_c2}, difiere >15 del esperado (≈71).\n"
            f"  Revisar: SJD_CLASS_INCLUDE / SJD_CLASS_EXCLUDE al inicio del script\n"
            f"  y verificar valores reales de {SJOGRENS_CLASS} con:\n"
            f"    df['{SJOGRENS_CLASS}'].value_counts(dropna=False)"
        )

    return c2, pd.DataFrame(exclusion_log)


# ── 6. Tabla demográfica basal ────────────────────────────────────────────────

def build_demographics_table(c2: pd.DataFrame) -> pd.DataFrame:
    """
    Tabla 1: una fila por paciente usando la visita ancla.
    Estratificada por: Overall | ≥3 visitas ESSDAI | exactamente 2 visitas ESSDAI.
    """
    # Visita ancla = fila con time_from_anchor_yrs mínimo (≥0) por paciente
    has_time = c2["time_from_anchor_yrs"].notna() & (c2["time_from_anchor_yrs"] >= 0)
    anchor_idx  = (
        c2[has_time]
        .groupby(PATIENT_ID)["time_from_anchor_yrs"]
        .idxmin()
    )
    anchors = c2.loc[anchor_idx].copy()
    anchors["essdai_severe_baseline"] = (
        anchors["essdai_baseline"] >= ESSDAI_SEVERE
    ).astype(float)

    # helpers
    def median_iqr(s):
        s = s.dropna()
        if len(s) == 0:
            return "N/A"
        return f"{s.median():.1f} [{s.quantile(0.25):.1f}–{s.quantile(0.75):.1f}]"

    def n_pct(s, value=1.0):
        s = s.dropna()
        if len(s) == 0:
            return "N/A"
        n = (s == value).sum()
        return f"{n} ({n/len(s)*100:.1f}%)"

    strata = {
        "Overall":                  anchors,
        "3+ ESSDAI visits":         anchors[anchors["c2_subset_3plus"]],
        "Exactly 2 ESSDAI visits":  anchors[~anchors["c2_subset_3plus"]],
    }

    rows = []
    vars_continuous = [
        ("N patients",                          None),
        ("Age at baseline, median [IQR]",       "age_at_baseline"),
        ("Disease duration at baseline (yrs)",  "disease_duration_yrs_model"),
        ("ESSDAI at baseline, median [IQR]",    "essdai_baseline"),
        ("ESSPRI total at baseline",            "esspri_total"),
        ("ESSDAI visits per patient",           "n_essdai_visits_total"),
    ]
    vars_binary = [
        (f"ESSDAI ≥{ESSDAI_SEVERE} at baseline (Pop 1), n (%)", "essdai_severe_baseline"),
        ("Female sex, n (%)",                                    "sex_female"),
        ("Primary SjD, n (%)",                                   "sjogrens_class_primary"),
    ]

    for stratum_name, sub in strata.items():
        n = len(sub)
        for label, col in vars_continuous:
            if col is None:
                val = str(n)
            else:
                val = median_iqr(sub[col]) if col in sub.columns else "N/A"
            rows.append({"variable": label, "stratum": stratum_name, "value": val})

        for label, col in vars_binary:
            val = n_pct(sub[col]) if col in sub.columns else "N/A"
            rows.append({"variable": label, "stratum": stratum_name, "value": val})

        # Race breakdown
        if RACE_COL in sub.columns:
            race_counts = sub[RACE_COL].dropna().value_counts()
            for cat, cnt in race_counts.items():
                pct = cnt / n * 100
                rows.append({
                    "variable": f"  Race: {cat}",
                    "stratum":  stratum_name,
                    "value":    f"{cnt} ({pct:.1f}%)",
                })

    demo = pd.DataFrame(rows)
    demo_wide = demo.pivot(index="variable", columns="stratum", values="value")
    # Orden de columnas
    col_order = [c for c in ["Overall", "3+ ESSDAI visits", "Exactly 2 ESSDAI visits"]
                 if c in demo_wide.columns]
    demo_wide = demo_wide[col_order].reset_index()
    return demo_wide


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01_build_cohort_c2.py — Construcción de cohorte C2")
    print("=" * 60)

    # 1. Cargar CSV wide (2057 columnas)
    df_wide = load_wide(VISITS_FILE)

    # 2. Transformar wide → long (seleccionar ~40 cols relevantes + phase_order)
    df_long = wide_to_long(df_wide)

    # 3. Derivar ancla por paciente y variables de tiempo
    df_long, qc_time = derive_time_variables(df_long)
    df = df_long  # alias para claridad
    c2, exclusion_log = build_c2(df)
    demographics = build_demographics_table(c2)

    # Guardar outputs
    c2.to_parquet(COHORT_C2_FILE, index=False)
    exclusion_log.to_csv(OUT_DIR / "cohort_c2_exclusions.csv",   index=False)
    demographics.to_csv( OUT_DIR / "cohort_c2_demographics.csv", index=False)
    qc_time.to_csv(      OUT_DIR / "qc_time_source.csv",         index=False)

    # Resumen de time_source
    if "time_source" in c2.columns:
        ts = (
            c2[c2[ESSDAI_TOTAL].notna()]
            .groupby("time_source", dropna=False)
            .agg(n_visits=(PATIENT_ID, "count"),
                 n_patients=(PATIENT_ID, "nunique"))
            .reset_index()
        )
        print(f"\n[time_source — visitas con ESSDAI]:")
        print(ts.to_string(index=False))

    print(f"\n✓ cohort_c2.parquet          → {COHORT_C2_FILE}")
    print(f"✓ cohort_c2_exclusions.csv   → {OUT_DIR}")
    print(f"✓ cohort_c2_demographics.csv → {OUT_DIR}")
    print(f"✓ qc_time_source.csv         → {OUT_DIR}")


if __name__ == "__main__":
    main()
