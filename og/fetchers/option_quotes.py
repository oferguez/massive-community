from __future__ import annotations

from abc import abstractmethod
from typing import List, Protocol

from models.quotes import OptionQuoteRow


class IOptionQuoteFetcher(Protocol):
    @abstractmethod
    def fetch_option_quotes(self, symbol: str, date_from: str, date_to: str, min_dte: int, max_dte: int) -> List[OptionQuoteRow]:
        ...


class OptionQuoteFetcher:
    def __init__(self, md_client: IOptionQuoteFetcher, verbose: bool = False):
        self.client: IOptionQuoteFetcher = md_client
        self.verbose: bool = verbose

    def fetch_option_quotes(self, symbol: str, date_from: str, date_to: str, min_dte: int, max_dte: int) -> List[OptionQuoteRow]:
        return self.client.fetch_option_quotes(
            symbol=symbol,
            date_from=date_from,
            date_to=date_to,
            min_dte=min_dte,
            max_dte=max_dte,
        )
