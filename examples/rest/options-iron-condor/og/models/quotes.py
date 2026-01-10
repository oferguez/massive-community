from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional


@dataclass(frozen=True)
class OptionQuoteRow:
    symbol: Optional[str] = None
    quote_date: Optional[date] = None
    as_of: Optional[datetime] = None
    expiration: Optional[date] = None
    dte: Optional[int] = None
    right: Optional[str] = None
    strike: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    iv: Optional[float] = None
    volume: Optional[int] = None
    size_raw: Optional[str] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    mid: Optional[float] = None
    spread: Optional[float] = None
    spread_pct: Optional[float] = None
    is_missing_market: Optional[bool] = None
    is_crossed: Optional[bool] = None
    raw: Optional[dict[Any, Any]] = None
