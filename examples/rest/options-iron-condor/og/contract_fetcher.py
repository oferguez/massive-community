#pylint: disable=missing-docstring
#pylint: disable=unused-import
#pylint: disable=line-too-long

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import date, datetime
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Union
import logging
from dotenv import load_dotenv
from massive import RESTClient # type: ignore
from massive.rest.models import OptionsContract # type: ignore
import pandas as pd
from urllib3 import HTTPResponse

# ---------------------------
# setup
# ---------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(
    LOG_DIR,
    f"contract_fetcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    filename=log_file,
    filemode="w"
)

logger = logging.getLogger(__name__)
logger.info('starting')
load_dotenv()

# ---------------------------
# ENUMS
# ---------------------------

class ContractType(Enum):
    CALL = "call"
    PUT = "put"
    OTHER = "other"

    def __lt__(self, other: Any):
        if  isinstance(other, ContractType):
            return self.value < other.value
        return NotImplemented

class ExerciseStyle(Enum):
    AMERICAN = "american"
    EUROPEAN = "european"
    BERMUDAN = "bermudan"
    UNKNOWN = "unknown"

    def __lt__(self, other: Any):
        if  isinstance(other, ExerciseStyle):
            return self.value < other.value
        return NotImplemented


# ---------------------------
# MODEL
# ---------------------------

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
    raw: Optional[dict[Any,Any]] = None


# ---------------------------
# INTERFACE
# ---------------------------

class IContractFetcher(Protocol):
    @abstractmethod
    def fetch_contracts(
        self,
        underlying_ticker: str,
        exp_from: str,
        exp_to: str
    ) -> List[OptionContractRow]:
        ...




# ---------------------------
# MASSIVE FETCHER
# ---------------------------

class MassiveContractFetcher(IContractFetcher):
    def __init__(self) -> None:
        self.massive_client:RESTClient = RESTClient(api_key=os.environ["MASSIVE_API_KEY"])
        super().__init__()

    @staticmethod
    # def parse_row(d: Dict[str, Any]) -> OptionContractRow:
    def parse_row(d: OptionsContract) -> OptionContractRow:
        return OptionContractRow(
            ticker = d.ticker,
            underlying_ticker = d.underlying_ticker,
            contract_type = Utils.safe_enum(ContractType, d.contract_type, ContractType.OTHER),
            expiration_date = Utils.to_date(d.expiration_date),
            strike_price = Utils.to_float(d.strike_price),
            exercise_style = Utils.safe_enum(ExerciseStyle, d.exercise_style, ExerciseStyle.UNKNOWN),
            shares_per_contract = Utils.to_float(d.shares_per_contract),
            primary_exchange =  Utils.to_str(d.primary_exchange),
            cfi = Utils.to_str(d.primary_exchange),
            correction = Utils.to_int(d.correction, alert_on_none = False),
            raw=d.__dict__
        )

    def fetch_contracts(self, underlying_ticker: str, exp_from: str, exp_to: str) -> List[OptionContractRow]:

        base_params: Dict[str, Any] = {
            "underlying_ticker": underlying_ticker,
            "limit": 1000,
            "expired": True,
            "order": "asc",
            "sort": "expiration_date",
            # `as_of` defaults to "today"; you can set it explicitly if you want
            # "as_of": "2025-01-15",
        }

        filter_params = {
            "expiration_date.gte": exp_from,
            "expiration_date.lte": exp_to,
        }

        contracts = self.massive_client.list_options_contracts(**base_params,params=filter_params)
        if isinstance(contracts, HTTPResponse):
            raise ValueError(f"API request failed; check your API key and parameters. Response content: {contracts}")

        assert hasattr(OptionContractRow, "__dataclass_fields__"), "OptionContractRow must be a dataclass"
        df = [MassiveContractFetcher.parse_row(c) for c in contracts] # type: ignore

        return df


class ContractFetcher:
    def __init__(self, md_client: IContractFetcher, verbose: bool = False):
        self.client:IContractFetcher = md_client
        self.verbose:bool = verbose

    def fetch_contracts(self, underlying_ticker: str, exp_from: str, exp_to: str) -> List[OptionContractRow]:
        df = self.client.fetch_contracts(underlying_ticker=underlying_ticker, exp_from=exp_from, exp_to=exp_to)
        return df


# ---------------------------
# UTILS
# ---------------------------

class Utils:

    @staticmethod
    def to_yyyy_mm_dd(d: str|date|datetime, alert_on_none:bool = True) -> str|None:
        if isinstance(d, str):
            return d
        if isinstance(d, datetime):
            return d.date().isoformat()
        try:
            # Assume it's date, otherwise raise
            return d.isoformat()
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Invalid date type: {type(d)} {e}"
                logger.warning(msg)
                raise TypeError(f"Invalid date type: {type(d)} {e}") from e
            else:
                return None

    @staticmethod
    def to_float(d: Any, alert_on_none:bool = True) -> float|None:
        try:
            return float(d)
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Cannot convert to float: {type(d)} exception: {e}"
                logger.warning(msg)
                raise TypeError(msg) from e
            return None

    @staticmethod
    def to_int(d: Any, alert_on_none:bool = True) -> int|None:
        try:
            return int(d)
        except (TypeError, ValueError) as e:
            if alert_on_none:
                msg = f"Cannot convert to int: {type(d)} exception: {e}"
                logger.warning(msg)
                raise TypeError(msg) from e
            return None


    @staticmethod
    def to_str(d: Any, alert_on_none: bool = True) -> str|None:
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
        if isinstance(d, str): # type: ignore
            return datetime.strptime(d, "%Y-%m-%d").date()
        raise TypeError(f"Unsupported type: {type(d)}")

    @staticmethod
    def safe_enum[E: Enum](enum_cls:type[E], val:Any, fallback:E|None=None) -> E | None :
        try:
            return enum_cls(val) if val else fallback
        except ValueError:
            return fallback

    @staticmethod
    def postprocess_contracts_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        sort_cols = [c for c in ["expiration_date", "contract_type", "strike_price", "ticker"] if c in df.columns]
        df = df.sort_values(sort_cols, ascending=True, kind="stable").reset_index(drop=True)

        return df


# ---------------------------
# Example usage
# ---------------------------

def test_aapl_massive():
    contracts_fetcher = ContractFetcher(md_client=MassiveContractFetcher(), verbose=True)
    contracts = contracts_fetcher.fetch_contracts("AAPL", "2024-01-01", "2024-01-15")

    contracts_df = pd.DataFrame([
        {k: v for k, v in asdict(c).items() if k != "raw"}
        for c in contracts
    ])
    print(contracts_df.head())
    print(contracts_df['strike_price'].describe())
    print(contracts_df.describe(include='all'))

    script_dir = Path(__file__).parent
    output_path = script_dir / '../notebooks/data/aapl.csv'
    contracts_df.to_csv(output_path.resolve())

if __name__ == "__main__":
    test_aapl_massive()
