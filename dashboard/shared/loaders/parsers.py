from dataclasses import dataclass
from decimal import Decimal
import re


@dataclass(frozen=True)
class ParsedValue:
    kind: str
    raw: str
    fields: dict


@dataclass(frozen=True)
class ParseFailure:
    raw: str


NUM = r"-?\d+(?:\.\d+)?"
POS = r"\d+(?:\.\d+)?"
PATTERNS = [
    ("median_n", rf"(?P<median>{NUM}) \((?P<q1>{NUM})–(?P<q3>{NUM})\); n=(?P<n>\d+)"),
    ("median", rf"(?P<median>{NUM}) \((?P<q1>{NUM})–(?P<q3>{NUM})\)"),
    ("fraction", rf"(?P<k>\d+)/(?P<N>\d+) \((?P<pct>{POS})%\)"),
    ("count_pct", rf"(?P<k>\d+) \((?P<pct>{POS})%\)"),
    ("range", rf"(?P<range_lo>{POS})–(?P<range_hi>{POS})"),
    ("number", rf"(?P<number>{NUM})"),
]


def parse_cell(raw):
    if raw == "":
        return ParsedValue("empty", raw, {})
    if raw == "NA":
        return ParsedValue("na", raw, {})
    for kind, pattern in PATTERNS:
        match = re.fullmatch(pattern, raw)
        if match:
            return ParsedValue(kind, raw, {k: Decimal(v) for k, v in match.groupdict().items()})
    return ParseFailure(raw)
