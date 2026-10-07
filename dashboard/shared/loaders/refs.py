from dataclasses import dataclass
from decimal import Decimal
import re
from shared.loaders.registry import MissingOutput, SchemaError
from shared.loaders.parsers import parse_cell, ParseFailure
from shared.utils.formatting import format_number

GRAMMAR = re.compile(r"^(?P<id>[a-zA-Z0-9_]+):(?P<key>[^@]+)@(?P<column>[^#]+)#(?P<field>raw|median|q1|q3|n|k|N|pct|range_lo|range_hi)$")


@dataclass(frozen=True)
class Provenance:
    file: str
    row_key: str
    column: str
    line_hint: int | None
    run_key: str
    producer: str


@dataclass(frozen=True)
class Value:
    raw: str
    number: Decimal | None
    display: str
    source: Provenance
    kind: str = ""
    warning: str = ""


def split_ref(ref):
    match = GRAMMAR.fullmatch(ref)
    if not match:
        raise SchemaError("Invalid reference: " + ref)
    return match.groupdict()


def resolve(ref, run, decimals=None):
    parts = split_ref(ref)
    entry = run.registry.get(parts["id"])
    frame = run.data(parts["id"])
    keys = entry.spec.get("key", [])
    values = parts["key"].split("|")
    if len(values) != len(keys):
        raise SchemaError("Row key does not match registry")
    selected = frame
    for key, value in zip(keys, values):
        selected = selected.loc[selected[key].eq(value)]
    if len(selected) != 1 or parts["column"] not in frame:
        raise MissingOutput("Requested cell is absent: " + ref)
    row = selected.iloc[0]
    raw = row[parts["column"]]
    source = Provenance(entry.path, parts["key"], parts["column"], int(selected.index[0]) + 2, run.key, entry.producer)
    parsed = parse_cell(raw)
    if isinstance(parsed, ParseFailure):
        return Value(raw, None, raw, source, "failure", "Unrecognized source-cell format; raw text shown")
    if parsed.kind in {"empty", "na"}:
        return Value(raw, None, "—" if parsed.kind == "empty" else "NA", source, parsed.kind, "empty in source" if parsed.kind == "empty" else "")
    if parsed.kind == "median_n" and entry.id == "t11_overall":
        available = parse_cell(row["N available"])
        if isinstance(available, ParseFailure) or available.fields.get("number") != parsed.fields["n"]:
            raise SchemaError("N available differs from the source-cell n")
    field = parts["field"]
    number = parsed.fields.get("number") if field == "raw" else parsed.fields.get(field)
    if field != "raw" and field not in parsed.fields:
        raise SchemaError("Requested field is absent in source cell: " + ref)
    display = raw if decimals is None or number is None else format_number(number, decimals)
    return Value(raw, number, display, source, parsed.kind)


def metadata_counts(run):
    frame = run.data("t11_dict")
    flags = frame["include_in_table1"].tolist()
    if any(flag not in {"True", "False", "true", "false"} for flag in flags):
        raise SchemaError("Unrecognized metadata boolean")
    return len([flag for flag in flags if flag in {"True", "true"}]), len(flags)


def render_template(text, run):
    def display(match):
        ref=match.group(1)
        field=ref.rsplit('#',1)[1]
        decimals=1 if field in {'pct','median','q1','q3','range_lo','range_hi'} or '@pct_' in ref else 0 if field in {'n','k','N'} else None
        return resolve(ref,run,decimals).display
    return re.sub(r"\{\{ref:(.*?)\}\}",display,text)
