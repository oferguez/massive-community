from __future__ import annotations

from abc import abstractmethod
import logging
import os
from time import sleep
from typing import Any, Dict, Iterator, List, Protocol

from massive import RESTClient  # type: ignore
from massive.rest.models import OptionsContract  # type: ignore
from urllib3 import HTTPResponse

from models.contracts import ContractType, ExerciseStyle, OptionContractRow
from utils.conversions import Utils

logger = logging.getLogger(__name__)


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


class MassiveContractFetcher(IContractFetcher):
    def __init__(self) -> None:
        self.massive_client: RESTClient = RESTClient(api_key=os.environ["MASSIVE_API_KEY"])
        super().__init__()

    @staticmethod
    def parse_row(d: OptionsContract) -> OptionContractRow:
        return OptionContractRow(
            ticker=d.ticker,
            underlying_ticker=d.underlying_ticker,
            contract_type=Utils.safe_enum(ContractType, d.contract_type, ContractType.OTHER),
            expiration_date=Utils.to_date(d.expiration_date),
            strike_price=Utils.to_float(d.strike_price),
            exercise_style=Utils.safe_enum(ExerciseStyle, d.exercise_style, ExerciseStyle.UNKNOWN),
            shares_per_contract=Utils.to_float(d.shares_per_contract),
            primary_exchange=Utils.to_str(d.primary_exchange),
            cfi=Utils.to_str(d.primary_exchange),
            correction=Utils.to_int(d.correction, alert_on_none=False),
            raw=d.__dict__,
        )

    def fetch_contracts_by_expiry(
        self,
        underlying_ticker: str,
        exp_from: str,
        exp_to: str,
    ) -> List[OptionContractRow]:
        base_params: Dict[str, Any] = {
            "underlying_ticker": underlying_ticker,
            "limit": 1000,
            "expired": True,
            "order": "asc",
            "sort": "expiration_date",
        }

        filter_params = {
            "expiration_date.gte": exp_from,
            "expiration_date.lte": exp_to,
        }

        contracts = self.massive_client.list_options_contracts(**base_params, params=filter_params)
        if isinstance(contracts, HTTPResponse):
            msg = f"API request failed; Response content: {contracts}"
            logger.warning(msg)
            raise ValueError(msg)

        assert hasattr(OptionContractRow, "__dataclass_fields__"), "OptionContractRow must be a dataclass"
        df = [MassiveContractFetcher.parse_row(c) for c in contracts]  # type: ignore

        return df

    def fetch_contracts_by_date(self, underlying_ticker: str, as_of: str) -> List[OptionContractRow]:
        contracts: Iterator[OptionsContract] | HTTPResponse = self.massive_client.list_options_contracts(
            underlying_ticker=underlying_ticker,
            as_of=as_of,
            expired=False,
            order="asc",
            limit=1000,
            sort="ticker",
        )

        if isinstance(contracts, HTTPResponse):
            msg = f"API request failed; Response content: {contracts}"
            logger.warning(msg)
            raise ValueError(msg)

        assert hasattr(OptionContractRow, "__dataclass_fields__"), "OptionContractRow must be a dataclass"

        df: List[OptionContractRow] = []
        sleepover = 1
        try:
            for idx, contract in enumerate(contracts):
                df.append(MassiveContractFetcher.parse_row(contract))
                logger.info("appended %s", idx)
                if idx and idx % 4000 == 0:
                    sleepover += 1
                    if sleepover > 3:
                        logger.warning("breaking out after %s contracts", idx)
                        break
                    logger.warning("throttling requesting due to free tier rate limits %s", idx)
                    sleep(60)
        except Exception as e:
            logger.error("Error fetching options: %s", e)
            raise

        return df
