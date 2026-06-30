"""
00_input data.py — Carga inicial del parquet analítico corregido
=================================================================

Carga el archivo parquet fuente, filtra la cohorte para conservar solamente
registros cuyo campo ``visit_summary_form__sjogrens_class`` incluya al menos
una de las categorías 1, 2 o 4, y exporta el resultado en parquet y CSV bajo
``data/raw/``.
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

    filtered = df[df[SJOGRENS_CLASS_COL].map(has_included_sjogrens_class)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(OUTPUT_PARQUET, index=False)
    filtered.to_csv(OUTPUT_CSV, index=False)

    print(f"Fuente: {SOURCE_FILE}")
    print(f"Filas originales: {len(df):,}")
    print(f"Filas filtradas: {len(filtered):,}")
    print(f"Parquet guardado: {OUTPUT_PARQUET}")
    print(f"CSV guardado: {OUTPUT_CSV}")

    return filtered


if __name__ == "__main__":
    load_filter_export()
