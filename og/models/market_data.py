from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Sequence

from models.prices import InstrumentPriceRow
from models.quotes import OptionQuoteRow


@dataclass(frozen=True)
class OptionChain:
    symbol: str
    quote_date: date
    as_of: Optional[datetime] = None
    quotes: Sequence[OptionQuoteRow] = field(default_factory=list)


@dataclass(frozen=True)
class InstrumentSnapshot:
    symbol: str
    quote_date: date
    price: Optional[InstrumentPriceRow] = None
    option_chain: Optional[OptionChain] = None


@dataclass(frozen=True)
class MarketSnapshot:
    quote_date: date
    instruments: Dict[str, InstrumentSnapshot] = field(default_factory=dict)
