from __future__ import annotations

from datetime import date, datetime

from models.quotes import OptionQuoteRow
from utils.conversions import Utils


def format_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4g}"


def format_int(value: int | None) -> str:
    return "None" if value is None else str(value)


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (date, datetime)):
        return Utils.to_yyyy_mm_dd(value) or value.isoformat()
    return str(value)


def format_quote(quote: OptionQuoteRow, price: float) -> str:
    moneyness = None
    if quote.strike is not None:
        moneyness = (quote.strike - price) / price
    return (
        f"{quote.right or '?'} strike={format_float(quote.strike)} "
        f"dte={quote.dte} "
        f"bid={format_float(quote.bid)} ask={format_float(quote.ask)} "
        f"mid={format_float(quote.mid)} size={format_int(quote.bid_size)}/{format_int(quote.ask_size)} "
        f"mny={format_float(moneyness)}"
    )
