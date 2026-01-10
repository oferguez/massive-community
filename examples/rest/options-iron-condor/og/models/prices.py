from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional


@dataclass(frozen=True)
class InstrumentPriceRow:
    ticker: Optional[str] = None
    date: Optional[date] = None
    price: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    as_of: Optional[date] = None
    raw: Optional[dict[Any, Any]] = None
