"""
00_input data.py — Carga inicial del parquet analítico corregido
=================================================================

Carga el archivo parquet fuente, une las visitas superpuestas por paciente,
filtra la cohorte para conservar solamente registros cuyo campo
``visit_summary_form__sjogrens_class`` incluya al menos una de las categorías
1, 2 o 4, y exporta el resultado en parquet y CSV bajo ``data/raw/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SOURCE_FILE = Path(
    "/data/salazarda/data/eda_sjd/data_analytic/"
    "visits_long_collapsed_by_interval_codebook_corrected.parquet"
)
OUTPUT_DIR = Path("data/raw")
OUTPUT_BASENAME = "visits_long_collapsed_by_interval_codebook_corrected"
OUTPUT_PARQUET = OUTPUT_DIR / f"{OUTPUT_BASENAME}.parquet"
OUTPUT_CSV = OUTPUT_DIR / f"{OUTPUT_BASENAME}.csv"
SJOGRENS_CLASS_COL = "visit_summary_form__sjogrens_class"
SJOGRENS_CLASS_INCLUDE = {"1", "2", "4"}
PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
INTERVAL_NAME_COL = "ids__interval_name"
NATURAL_HISTORY_INTERVAL = "Natural History Protocol 478 Interval"
_MISSING_VALUES = {"", "na", "n/a", "nan", "none", "unknown", "unk", "missing", "-99"}


def has_included_sjogrens_class(value: object) -> bool:
    """Return True when a value contains class code 1, 2, or 4.

    The source column may arrive as numeric values, strings, or multi-select
    strings with delimiters. Tokenizing on non-digits avoids matching values
    such as ``10`` while still accepting entries like ``"1, 4"``.
    """
    if pd.isna(value):
        return False

    tokens = [token for token in re.split(r"\D+", str(value).strip()) if token]
    return bool(SJOGRENS_CLASS_INCLUDE.intersection(tokens))


def _is_present(value: object) -> bool:
    """Return whether ``value`` is a usable source value for a merged row."""
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in _MISSING_VALUES


def _normalized_patient_id(value: object) -> str | None:
    if not _is_present(value):
        return None
    return re.sub(r"(?<=\d)\.0$", "", str(value).strip())


def _visit_date_keys(value: object) -> set[pd.Timestamp]:
    """Parse every valid pipe-delimited date in a source visit."""
    if not _is_present(value):
        return set()
    keys = set()
    for fragment in str(value).split("|"):
        parsed = pd.to_datetime(fragment.strip(), errors="coerce")
        if pd.notna(parsed):
            keys.add(pd.Timestamp(parsed).normalize())
    return keys


def _is_non_natural_history_visit(value: object) -> bool:
    return _is_present(value) and str(value).strip() != NATURAL_HISTORY_INTERVAL


def merge_matching_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Merge visits for a patient when their valid date fragments overlap.

    A visit dated ``2015-05-13`` therefore merges with a visit dated
    ``2015-05-13 | 2015-05-18``.  Within each matched set, values from a visit
    whose interval is not the Natural History Protocol 478 interval take
    precedence.  This applies to every column, including the source and audit
    fields, and falls back to the first usable value when the preferred visit
    has no value.
    """
    required = {PATIENT_ID_COL, VISIT_DATE_COL, INTERVAL_NAME_COL}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"No se encontraron las columnas requeridas: {sorted(missing)}")

    # Build connected components per patient: a pipe-delimited visit can join
    # more than two rows when it contains dates represented by separate rows.
    parent = list(range(len(df)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_for_patient_date: dict[tuple[str, pd.Timestamp], int] = {}
    for position, (_, row) in enumerate(df.iterrows()):
        patient = _normalized_patient_id(row[PATIENT_ID_COL])
        if patient is None:
            continue
        for date in _visit_date_keys(row[VISIT_DATE_COL]):
            key = (patient, date)
            if key in first_for_patient_date:
                union(position, first_for_patient_date[key])
            else:
                first_for_patient_date[key] = position

    components: dict[int, list[int]] = {}
    for position in range(len(df)):
        components.setdefault(find(position), []).append(position)

    merged_rows = []
    for positions in components.values():
        group = df.iloc[positions]
        preferred = group[group[INTERVAL_NAME_COL].map(_is_non_natural_history_visit)]
        ordered = pd.concat([preferred, group.drop(preferred.index)], axis=0)
        merged_rows.append({
            column: next((value for value in ordered[column] if _is_present(value)), pd.NA)
            for column in df.columns
        })
    return pd.DataFrame(merged_rows, columns=df.columns)


def load_filter_export() -> pd.DataFrame:
    """Load the source parquet, filter Sjogren's class, and write outputs."""
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo fuente: {SOURCE_FILE}")

    df = pd.read_parquet(SOURCE_FILE)
    if SJOGRENS_CLASS_COL not in df.columns:
        raise KeyError(
            f"No se encontró la columna requerida '{SJOGRENS_CLASS_COL}' "
            f"en {SOURCE_FILE}"
        )

    merged = merge_matching_visits(df)
    filtered = merged[merged[SJOGRENS_CLASS_COL].map(has_included_sjogrens_class)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUTPUT_PARQUET, index=False)
    filtered.to_csv(OUTPUT_CSV, index=False)

    print(f"Fuente: {SOURCE_FILE}")
    print(f"Filas originales: {len(df):,}")
    print(f"Filas filtradas: {len(filtered):,}")
    print(f"Filas después de unir visitas: {len(merged):,}")
    print(f"Parquet guardado: {OUTPUT_PARQUET}")
    print(f"CSV guardado: {OUTPUT_CSV}")

    return filtered


if __name__ == "__main__":
    load_filter_export()
