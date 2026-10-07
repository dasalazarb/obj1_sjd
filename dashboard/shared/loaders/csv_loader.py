from io import BytesIO
import csv
import re
import pandas as pd
from shared.loaders.registry import SchemaError

IDENTIFIER = re.compile(r"patient[_ ]?id|^ids__|(^|_)dob$|date_of_birth|birth_date|record_number|medical_record|(^|_)mrn$|patient_name", re.I)


def validate_header(columns):
    if any(IDENTIFIER.search(col) for col in columns):
        raise SchemaError('Patient identifier columns rejected')
    if len(columns) != len(set(columns)):
        raise SchemaError('Duplicate source column names rejected')


def check_header(path):
    # Check only the header before reading, fingerprinting or ingesting any rows.
    with path.open(encoding='utf-8-sig', newline='') as handle:
        columns=next(csv.reader(handle),[])
    validate_header(columns)


def read_csv_bytes(data, entry):
    from io import StringIO
    validate_header(next(csv.reader(StringIO(data.decode('utf-8-sig'))), []))
    frame = pd.read_csv(BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")
    if any(IDENTIFIER.search(col) for col in frame.columns):
        raise SchemaError("Patient identifier columns rejected")
    missing = set(entry.spec.get("columns", [])) - set(frame.columns)
    if missing:
        raise SchemaError("Missing columns: " + ", ".join(sorted(missing)))
    keys = entry.spec.get("key", [])
    if keys and frame.duplicated(keys).any():
        raise SchemaError("Duplicate row keys in " + entry.path)
    return frame
