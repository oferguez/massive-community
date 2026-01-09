from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Optional


class ContractType(Enum):
    CALL = "call"
    PUT = "put"
    OTHER = "other"

    def __lt__(self, other: Any):
        if isinstance(other, ContractType):
            return self.value < other.value
        return NotImplemented


class ExerciseStyle(Enum):
    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"
    UNKNOWN = "unknown"

    def __lt__(self, other: Any):
        if isinstance(other, ExerciseStyle):
            return self.value < other.value
        return NotImplemented


@dataclass(frozen=True)
class OptionContractRow:
    ticker: Optional[str] = None
    underlying_ticker: Optional[str] = None
    contract_type: Optional[ContractType] = None
    expiration_date: Optional[date] = None
    strike_price: Optional[float] = None
    exercise_style: Optional[ExerciseStyle] = None
    shares_per_contract: Optional[float] = None
    primary_exchange: Optional[str] = None
    cfi: Optional[str] = None
    correction: Optional[int] = None
    raw: Optional[dict[Any, Any]] = None
