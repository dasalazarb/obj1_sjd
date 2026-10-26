"""Strict source dates; partial dates retain bounds without an invented day."""
import calendar
import re
from datetime import date, datetime
import pandas as pd

def raw(value):
    return "" if value is None or pd.isna(value) else str(value)

def present(value):
    return raw(value).strip().casefold() not in {"", "nan", "nat", "none", "null", "<na>"}

def parse_date(value, *, slash_order="MDY"):
    """Retain uncertainty bounds separately from observed dates; strict formats."""
    result = dict(raw=raw(value), parsed="", precision="", lower_bound="", upper_bound="",
                  parse_rule="", status="missing")
    if not present(value):
        return result
    text = raw(value).strip()
    try:
        if isinstance(value, datetime):
            result.update(parsed=value.isoformat(), precision="datetime", parse_rule="native_datetime")
            lower = upper = value.date()
        elif isinstance(value, date):
            result.update(parsed=value.isoformat(), precision="day", parse_rule="native_date")
            lower = upper = value
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}.*", text):
            stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
            result.update(parsed=stamp.isoformat(), precision="datetime", parse_rule="ISO_datetime")
            lower = upper = stamp.date()
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            lower = upper = date.fromisoformat(text)
            result.update(parsed=lower.isoformat(), precision="day", parse_rule="YYYY-MM-DD")
        elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2}(?:[.]\d+)?)?(?:\s*[APap][Mm])?(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?", text):
            if slash_order != "MDY":
                result.update(status="ambiguous_format", parse_rule="slash_datetime_not_authorized")
                return result
            from dateutil.parser import parse
            stamp = parse(text, dayfirst=False, yearfirst=False, fuzzy=False)
            result.update(parsed=stamp.isoformat(), precision="datetime", parse_rule="MM/DD/YYYY_explicit_timestamp")
            lower = upper = stamp.date()
        elif re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
            if slash_order != "MDY":
                result.update(status="ambiguous_format", parse_rule="slash_format_not_authorized")
                return result
            lower = upper = datetime.strptime(text, "%m/%d/%Y").date()
            result.update(parsed=lower.isoformat(), precision="day", parse_rule="MM/DD/YYYY")
        elif re.fullmatch(r"\d{1,2}/\d{4}", text) or re.fullmatch(r"\d{4}-\d{2}", text):
            if "/" in text:
                month, year = map(int, text.split("/"))
                rule = "MM/YYYY"
            else:
                year, month = map(int, text.split("-"))
                rule = "YYYY-MM"
            lower = date(year, month, 1)
            upper = date(year, month, calendar.monthrange(year, month)[1])
            result.update(precision="month", parse_rule=rule)
        elif re.fullmatch(r"\d{4}", text):
            year = int(text)
            lower, upper = date(year, 1, 1), date(year, 12, 31)
            result.update(precision="year", parse_rule="YYYY")
        else:
            result.update(status="unsupported_format", parse_rule="no_explicit_rule")
            return result
        result.update(lower_bound=lower.isoformat(), upper_bound=upper.isoformat(), status="valid")
    except (ValueError, OverflowError):
        result.update(parsed="", precision="", status="invalid_date", parse_rule="invalid_calendar_date")
    return result

def exact_day(value):
    parsed = parse_date(value)
    return pd.Timestamp(parsed["lower_bound"]) if parsed["status"] == "valid" and parsed["precision"] in {"day", "datetime"} else pd.NaT
