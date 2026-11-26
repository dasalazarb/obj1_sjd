"""Write paired private exports with a reversible CSV representation of Arrow cells."""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json
import math
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as pq

from .settings import file_hash, write_json

csv.field_size_limit(sys.maxsize)
NULL = r"\N"
FORMULA = re.compile(r"^(?:\s*[=+\-@＝＋－＠]|[\t\r\n])")


def structured(value):
    """Produce a canonical JSON value for a non-text Arrow cell."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [structured(item) for item in value]
    if isinstance(value, dict):
        return {key: structured(item) for key, item in value.items()}
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Unsupported CSV logical value: {type(value).__name__}")


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cell(value):
    """Use JSON for non-text cells and an escape for spreadsheet-active text."""
    if value is None:
        return NULL
    if not isinstance(value, str):
        return json_cell(structured(value))
    if value.startswith(("'", "\\")) or FORMULA.match(value):
        return "'" + value
    return value


def unescape(value, dtype):
    """Decode the CSV wire representation using the canonical Arrow type."""
    if value == NULL:
        return None
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_string_view(dtype):
        return value[1:] if value.startswith("'") else value
    return json.loads(value)


def export_csv(path: Path) -> None:
    """Stream rows from Parquet, then verify every cell and record both hashes."""
    csv_path = path.with_suffix(".csv")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(schema.names)
        for batch in parquet.iter_batches(batch_size=2048):
            writer.writerows([cell(row[name]) for name in schema.names] for row in batch.to_pylist())
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader) != schema.names:
            raise ValueError(f"CSV header mismatch: {csv_path.name}")
        for batch in parquet.iter_batches(batch_size=2048):
            for expected in batch.to_pylist():
                row = next(reader, ())
                if len(row) != len(schema) or any(
                    json_cell(unescape(value, field.type)) != json_cell(structured(expected[field.name]))
                    for field, value in zip(schema, row)
                ):
                    raise ValueError(f"CSV roundtrip mismatch: {csv_path.name}")
        if next(reader, None) is not None:
            raise ValueError(f"CSV row count mismatch: {csv_path.name}")
    write_json(path.with_suffix(".csv.schema.json"), {
        "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
        "rows": parquet.metadata.num_rows,
        "encoding": "UTF-8; null = unescaped \\N; string escape = leading apostrophe; non-text = JSON; dates = ISO; binary = hex",
        "sha256": {artifact.name: file_hash(artifact) for artifact in (csv_path, path)},
    })
    csv_path.chmod(0o600)
    path.chmod(0o600)
    path.with_suffix(".csv.schema.json").chmod(0o600)


def pair_directory(root: Path) -> None:
    """Pair top-level tables, preserving the study subdirectory's legacy inputs."""
    for path in sorted(root.glob("*.csv")):
        parquet = path.with_suffix(".parquet")
        if not parquet.is_file():
            table = arrow_csv.read_csv(path, convert_options=arrow_csv.ConvertOptions(strings_can_be_null=True))
            pq.write_table(table, parquet)
    for path in sorted(root.glob("*.parquet")):
        sidecar = path.with_suffix(".csv.schema.json")
        if sidecar.is_file() and path.with_suffix(".csv").is_file():
            sha = json.loads(sidecar.read_text())["sha256"]
            if sha == {item.name: file_hash(item) for item in (path, path.with_suffix(".csv"))}:
                continue
        export_csv(path)
