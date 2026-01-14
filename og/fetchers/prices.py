from __future__ import annotations

from abc import abstractmethod
from typing import List, Protocol

from models.prices import InstrumentPriceRow


class IPriceFetcher(Protocol):
    @abstractmethod
    def fetch_prices(self, ticker: str, date_from: str, date_to: str) -> List[InstrumentPriceRow]:
        ...


class PriceFetcher:
    def __init__(self, md_client: IPriceFetcher, verbose: bool = False):
        self.client: IPriceFetcher = md_client
        self.verbose: bool = verbose

    def fetch_prices(self, ticker: str, date_from: str, date_to: str) -> List[InstrumentPriceRow]:
        rows = self.client.fetch_prices(ticker=ticker, date_from=date_from, date_to=date_to)
        return rows
