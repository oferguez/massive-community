from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any, List

from massive import RESTClient  # type: ignore

from fetchers.prices import IPriceFetcher
from models.prices import InstrumentPriceRow
from utils.conversions import Utils

logger = logging.getLogger(__name__)


class MassivePriceFetcher(IPriceFetcher):
    def __init__(self) -> None:
        self.massive_client: RESTClient = RESTClient(api_key=os.environ["MASSIVE_API_KEY"])
        super().__init__()

    @staticmethod
    def parse_row(ticker: str, price_date: date, agg: Any) -> InstrumentPriceRow:
        close_price = getattr(agg, "close", None)
        if close_price is None:
            close_price = getattr(agg, "after_hours", None)

        return InstrumentPriceRow(
            ticker=ticker,
            date=price_date,
            price=Utils.to_float(close_price, alert_on_none=False),
            open=Utils.to_float(getattr(agg, "open", None), alert_on_none=False),
            high=Utils.to_float(getattr(agg, "high", None), alert_on_none=False),
            low=Utils.to_float(getattr(agg, "low", None), alert_on_none=False),
            close=Utils.to_float(getattr(agg, "close", None), alert_on_none=False),
            volume=Utils.to_float(getattr(agg, "volume", None), alert_on_none=False),
            raw=getattr(agg, "__dict__", None),
        )

    def fetch_prices(self, ticker: str, date_from: str, date_to: str) -> List[InstrumentPriceRow]:
        start = Utils.to_date(date_from)
        end = Utils.to_date(date_to)
        if start > end:
            raise ValueError(f"date_from must be <= date_to; got {date_from} > {date_to}")

        rows: List[InstrumentPriceRow] = []
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                agg = self.massive_client.get_daily_open_close_agg(ticker=ticker, date=date_str)
                rows.append(self.parse_row(ticker, current, agg))
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Failed to fetch price for %s on %s: %s", ticker, date_str, e)
            current += timedelta(days=1)

        return rows
