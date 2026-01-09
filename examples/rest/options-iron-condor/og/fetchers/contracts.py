from __future__ import annotations

from abc import abstractmethod
from typing import List, Protocol

from models.contracts import OptionContractRow


class IContractFetcher(Protocol):
    @abstractmethod
    def fetch_contracts_by_expiry(
        self,
        underlying_ticker: str,
        exp_from: str,
        exp_to: str,
    ) -> List[OptionContractRow]:
        ...

    @abstractmethod
    def fetch_contracts_by_date(
        self,
        underlying_ticker: str,
        as_of: str,
    ) -> List[OptionContractRow]:
        ...


class ContractFetcher:
    def __init__(self, md_client: IContractFetcher, verbose: bool = False):
        self.client: IContractFetcher = md_client
        self.verbose: bool = verbose

    def fetch_contracts_by_expiry(self, underlying_ticker: str, exp_from: str, exp_to: str) -> List[OptionContractRow]:
        df = self.client.fetch_contracts_by_expiry(
            underlying_ticker=underlying_ticker,
            exp_from=exp_from,
            exp_to=exp_to,
        )
        return df

    def fetch_contracts_by_date(self, underlying_ticker: str, as_of: str) -> List[OptionContractRow]:
        df = self.client.fetch_contracts_by_date(underlying_ticker=underlying_ticker, as_of=as_of)
        return df
