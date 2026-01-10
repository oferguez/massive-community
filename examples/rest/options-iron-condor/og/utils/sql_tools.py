from __future__ import annotations

from datetime import date, datetime
from typing import Sequence


def format_sql_value(value: object) -> str:
    match value:
        case None:
            return "NULL"
        case bool() as flag:
            return "TRUE" if flag else "FALSE"
        case datetime() as dt:
            return f"'{dt.isoformat(sep=' ')}'"
        case date() as d:
            return f"'{d.isoformat()}'"
        case str() as text:
            return f"'{text.replace("'", "''")}'"
        case _:
            return str(value)


def format_query(query: str, params: Sequence[object]) -> str:
    parts = query.split("?")
    if len(parts) - 1 != len(params):
        return query.strip()

    formatted = parts[0]
    for part, param in zip(parts[1:], params, strict=False):
        formatted += format_sql_value(param) + part
    return formatted.strip()
