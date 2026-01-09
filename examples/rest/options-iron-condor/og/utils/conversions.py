from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class Utils:
    @staticmethod
    def to_yyyy_mm_dd(d: str | date | datetime, alert_on_none: bool = True) -> str | None:
        if isinstance(d, str):
            return d
        if isinstance(d, datetime):
            return d.date().isoformat()
        try:
            return d.isoformat()
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Invalid date type: {type(d)} {e}"
                logger.warning(msg)
                raise TypeError(f"Invalid date type: {type(d)} {e}") from e
            return None

    @staticmethod
    def to_float(d: Any, alert_on_none: bool = True) -> float | None:
        try:
            return float(d)
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Cannot convert to float: {type(d)} exception: {e}"
                logger.warning(msg)
                raise TypeError(msg) from e
            return None

    @staticmethod
    def to_int(d: Any, alert_on_none: bool = True) -> int | None:
        try:
            return int(d)
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Cannot convert to int: {type(d)} exception: {e}"
                logger.warning(msg)
                raise TypeError(msg) from e
            return None

    @staticmethod
    def to_str(d: Any, alert_on_none: bool = True) -> str | None:
        try:
            return str(d)
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Cannot convert to str: {type(d)} exception: {e}"
                logger.warning(msg)
                raise TypeError(msg) from e
            return None

    @staticmethod
    def to_date(d: Any) -> date:
        if isinstance(d, date) and not isinstance(d, datetime):
            return d
        if isinstance(d, datetime):
            return d.date()
        if isinstance(d, str):
            return datetime.strptime(d, "%Y-%m-%d").date()
        raise TypeError(f"Unsupported type: {type(d)}")

    @staticmethod
    def safe_enum[E: Enum](enum_cls: type[E], val: Any, fallback: E | None = None) -> E | None:
        try:
            return enum_cls(val) if val else fallback
        except ValueError:
            return fallback
