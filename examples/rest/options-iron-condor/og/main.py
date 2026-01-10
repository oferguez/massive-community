from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import logging
import argparse

from dotenv import load_dotenv
import pandas as pd

from fetchers.contracts import ContractFetcher
from fetchers.duckdb_option_quote_fetcher import DuckDbOptionQuoteFetcher
from fetchers.duckdb_price_fetcher import DuckDbPriceFetcher
from fetchers.massive_contract_fetcher import MassiveContractFetcher
from fetchers.massive_price_fetcher import MassivePriceFetcher
from fetchers.prices import PriceFetcher
from utils.market_snapshot_builder import MarketSnapshotBuilder
from utils.logging_config import configure_logging
from utils.db_tools import describe_db

logger = logging.getLogger(__name__)

OTM_SAMPLE_SIZE = 5
ATM_SAMPLE_SIZE = 8


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
    parser = argparse.ArgumentParser(description="Build a market snapshot from DuckDB data.")
    parser.add_argument("--quote-date", default="2020-01-02", help="Target quote date (YYYY-MM-DD).")
    parser.add_argument("--symbols", default="AAPL", help="Comma-separated list of symbols.")
    parser.add_argument("--min-dte", type=int, default=1, help="Minimum days to expiration filter.")
    parser.add_argument("--max-dte", type=int, default=7, help="Maximum days to expiration filter (exclusive).")
    args = parser.parse_args()
    db_path = Path(__file__).parent / "data" / "aapl_options_norm.duckdb"

    configure_logging()
    load_dotenv()
    # describe_db(db_path=db_path)

    price_fetcher = DuckDbPriceFetcher(db_path=db_path)
    quote_fetcher = DuckDbOptionQuoteFetcher(db_path=db_path)
    builder = MarketSnapshotBuilder(price_fetcher=price_fetcher, option_quote_fetcher=quote_fetcher)
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    snapshot = builder.build_for_date(args.quote_date, symbols=symbols, min_dte=args.min_dte, max_dte=args.max_dte)
    logger.info("Snapshot instruments: %s", list(snapshot.instruments.keys()))
    for symbol, instrument in snapshot.instruments.items():
        if instrument.option_chain:
            quote_count = len(instrument.option_chain.quotes)
            expirations = {quote.expiration for quote in instrument.option_chain.quotes if quote.expiration}
        else:
            quote_count = 0
            expirations = set()
        logger.info(
            "%s: price=%s quotes=%d expirations=%d",
            symbol,
            instrument.price.close if instrument.price else None,
            quote_count,
            len(expirations),
        )
        if not instrument.option_chain or not instrument.price or instrument.price.close is None:
            continue
        price = instrument.price.close
        quotes_with_strike = [quote for quote in instrument.option_chain.quotes if quote.strike is not None]
        if not quotes_with_strike:
            continue
        otm_calls = [quote for quote in quotes_with_strike if quote.right == "C" and quote.strike > price]
        otm_puts = [quote for quote in quotes_with_strike if quote.right == "P" and quote.strike < price]
        otm_calls_sorted = sorted(otm_calls, key=lambda quote: quote.strike, reverse=True)
        otm_puts_sorted = sorted(otm_puts, key=lambda quote: quote.strike)
        atm_sorted = sorted(quotes_with_strike, key=lambda quote: abs(quote.strike - price))

        def format_float(value: float | None) -> str:
            if value is None:
                return "None"
            return f"{value:.4g}"

        def format_int(value: int | None) -> str:
            return "None" if value is None else str(value)

        def format_quote(quote: object) -> str:
            moneyness = None
            if quote.strike is not None:
                moneyness = (quote.strike - price) / price
            return (
                f"{quote.right or '?'} strike={format_float(quote.strike)} "
                f"dte={quote.dte} "
                f"bid={format_float(quote.bid)} ask={format_float(quote.ask)} "
                f"mid={format_float(quote.mid)} size={format_int(quote.bid_size)}/{format_int(quote.ask_size)} "
                f"mny={format_float(moneyness)}"
            )

        logger.info("%s: most OTM calls (%d):", symbol, OTM_SAMPLE_SIZE)
        for quote in otm_calls_sorted[:OTM_SAMPLE_SIZE]:
            logger.info("  %s", format_quote(quote))
        logger.info("%s: most OTM puts (%d):", symbol, OTM_SAMPLE_SIZE)
        for quote in otm_puts_sorted[:OTM_SAMPLE_SIZE]:
            logger.info("  %s", format_quote(quote))
        logger.info("%s: near ATM (%d):", symbol, ATM_SAMPLE_SIZE)
        for quote in atm_sorted[:ATM_SAMPLE_SIZE]:
            logger.info("  %s", format_quote(quote))


if __name__ == "__main__":
    main()
