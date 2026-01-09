from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import logging

from dotenv import load_dotenv
import pandas as pd

from fetchers.contracts import ContractFetcher
from fetchers.massive_contract_fetcher import MassiveContractFetcher
from fetchers.prices import PriceFetcher
from fetchers.massive_price_fetcher import MassivePriceFetcher
from utils.logging_config import configure_logging
from data.db_tools import describe_db

logger = logging.getLogger(__name__)


def test_scan(ticker: str, as_of: str) -> None:
    contracts_fetcher = ContractFetcher(md_client=MassiveContractFetcher(), verbose=True)
    prices_fetcher = PriceFetcher(md_client=MassivePriceFetcher(), verbose=True)
    contracts = contracts_fetcher.fetch_contracts_by_date(ticker, as_of)
    prices = prices_fetcher.fetch_prices(ticker, as_of, as_of)

    contracts_df = pd.DataFrame([{k: v for k, v in asdict(c).items() if k != "raw"} for c in contracts])
    logger.info(contracts_df.head())
    logger.info(contracts_df.describe(include="all"))
    script_dir = Path(__file__).parent
    output_path = script_dir / "../notebooks/data/aapl_contracts.csv"
    contracts_df.to_csv(output_path.resolve())

    logger.info("Fetched %d contracts and %d prices for %s as of %s", len(contracts), len(prices), ticker, as_of)
    logger.info("Prices:%s", prices)
    logger.info("Contracts: %s", contracts)


def test_aapl_massive() -> None:
    contracts_fetcher = ContractFetcher(md_client=MassiveContractFetcher(), verbose=True)
    contracts = contracts_fetcher.fetch_contracts_by_expiry("AAPL", "2025-01-01", "2025-01-15")

    contracts_df = pd.DataFrame([{k: v for k, v in asdict(c).items() if k != "raw"} for c in contracts])
    print(contracts_df.head())
    print(contracts_df.describe(include="all"))

    script_dir = Path(__file__).parent
    output_path = script_dir / "../notebooks/data/aapl_contracts.csv"
    contracts_df.to_csv(output_path.resolve())

    prices_fetcher = PriceFetcher(md_client=MassivePriceFetcher(), verbose=True)
    prices = prices_fetcher.fetch_prices("AAPL", "2025-01-01", "2025-01-15")
    prices_df = pd.DataFrame([{k: v for k, v in asdict(p).items() if k != "raw"} for p in prices])
    print(prices_df.head())
    print(prices_df.describe(include="all"))
    output_path = script_dir / "../notebooks/data/aapl_prices.csv"
    prices_df.to_csv(output_path.resolve())


def main() -> None:
    configure_logging()
    load_dotenv()
    describe_db()


if __name__ == "__main__":
    main()
