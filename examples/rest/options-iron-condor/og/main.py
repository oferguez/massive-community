from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from datetime import timedelta
import logging
import argparse
import heapq

from dotenv import load_dotenv
import pandas as pd

from fetchers.contracts import ContractFetcher
from fetchers.duckdb_option_quote_fetcher import DuckDbOptionQuoteFetcher
from fetchers.duckdb_price_fetcher import DuckDbPriceFetcher
from fetchers.massive_contract_fetcher import MassiveContractFetcher
from fetchers.massive_price_fetcher import MassivePriceFetcher
from fetchers.prices import PriceFetcher
from models.iron_condor import build_iron_condor_candidates, estimate_probability_of_profit
from utils.conversions import Utils
from utils.market_snapshot_builder import MarketSnapshotBuilder
from utils.logging_config import configure_logging
from utils.db_tools import describe_db

logger = logging.getLogger(__name__)

OTM_SAMPLE_SIZE = 5
ATM_SAMPLE_SIZE = 8
BEARISH_SAMPLE_SIZE = 4
BULLISH_SAMPLE_SIZE = 4
POP_SAMPLE_SIZE = 4


def format_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4g}"


def format_int(value: int | None) -> str:
    return "None" if value is None else str(value)


def quote_mid(quote: object) -> float | None:
    if quote.mid is not None:
        return quote.mid
    if quote.bid is not None and quote.ask is not None:
        return (quote.bid + quote.ask) / 2
    return None


def condor_metadata(condor: object) -> tuple[int | None, object | None, object | None]:
    dte_values = [
        condor.short_put.dte,
        condor.long_put.dte,
        condor.short_call.dte,
        condor.long_call.dte,
    ]
    dte_values = [value for value in dte_values if value is not None]
    dte = min(dte_values) if dte_values else None
    expirations = [
        condor.short_put.expiration,
        condor.long_put.expiration,
        condor.short_call.expiration,
        condor.long_call.expiration,
    ]
    expirations = [exp for exp in expirations if exp]
    expiration = min(expirations) if expirations else None
    as_of_values = [
        condor.short_put.as_of,
        condor.long_put.as_of,
        condor.short_call.as_of,
        condor.long_call.as_of,
    ]
    as_of_values = [value for value in as_of_values if value]
    as_of_value = min(as_of_values) if as_of_values else None
    return dte, expiration, as_of_value


def format_condor(condor: object) -> str:
    dte, expiration, as_of_value = condor_metadata(condor)
    return (
        f"short_put={format_float(condor.short_put.strike)}@{format_float(quote_mid(condor.short_put))} "
        f"long_put={format_float(condor.long_put.strike)}@{format_float(quote_mid(condor.long_put))} "
        f"short_call={format_float(condor.short_call.strike)}@{format_float(quote_mid(condor.short_call))} "
        f"long_call={format_float(condor.long_call.strike)}@{format_float(quote_mid(condor.long_call))} "
        # f"dte={dte} "
        f"exp={expiration} "
        # f"as_of={as_of_value}"
    )


def condor_signature(condor: object) -> tuple[float | None, float | None, float | None, float | None, int | None, object | None, object | None]:
    dte, expiration, as_of_value = condor_metadata(condor)
    return (
        condor.short_put.strike,
        condor.long_put.strike,
        condor.short_call.strike,
        condor.long_call.strike,
        dte,
        expiration,
        as_of_value,
    )


def select_samples(
    candidates: list[object],
    size: int,
    key_fn,
) -> list[object]:
    samples: list[object] = []
    seen: set[tuple[float | None, float | None, float | None, float | None, int | None, object | None, object | None]] = set()
    for condor in sorted(candidates, key=key_fn):
        signature = condor_signature(condor)
        if signature in seen:
            continue
        seen.add(signature)
        samples.append(condor)
        if len(samples) >= size:
            break
    return samples


def condor_net_credit(
    condor: object,
    lookup: dict[tuple[object, object, object, object], object] | None = None,
    quote_date: object | None = None,
) -> float | None:
    if lookup is None:
        legs = [condor.short_put, condor.short_call, condor.long_put, condor.long_call]
    else:
        if quote_date is None:
            return None
        legs = []
        for leg in [condor.short_put, condor.short_call, condor.long_put, condor.long_call]:
            key = (leg.expiration, leg.right, leg.strike, quote_date)
            legs.append(lookup.get(key))
        if any(leg is None for leg in legs):
            return None
    short_put_mid = quote_mid(legs[0])
    short_call_mid = quote_mid(legs[1])
    long_put_mid = quote_mid(legs[2])
    long_call_mid = quote_mid(legs[3])
    if None in (short_put_mid, short_call_mid, long_put_mid, long_call_mid):
        return None
    return (short_put_mid + short_call_mid) - (long_put_mid + long_call_mid)


def run_closeout_simulation(
    symbol: str,
    condors: list[object],
    start_date: object,
    quote_fetcher: DuckDbOptionQuoteFetcher,
    price_fetcher: DuckDbPriceFetcher,
    days: int = 14,
) -> None:
    end_date = start_date + timedelta(days=days)
    start_str = Utils.to_yyyy_mm_dd(start_date) or start_date.isoformat()
    end_str = Utils.to_yyyy_mm_dd(end_date) or end_date.isoformat()
    backtest_quotes = quote_fetcher.fetch_option_quotes(
        symbol=symbol,
        date_from=start_str,
        date_to=end_str,
        min_dte=0,
        max_dte=3650,
    )
    quote_index: dict[tuple[object, object, object, object], object] = {}
    for quote in backtest_quotes:
        if not quote.quote_date or quote.strike is None or not quote.right or not quote.expiration:
            continue
        quote_index[(quote.expiration, quote.right, quote.strike, quote.quote_date)] = quote
    price_rows = price_fetcher.fetch_prices(symbol, start_str, end_str)
    price_index = {row.date: row for row in price_rows if row.date}

    logger.info("  %s: %d-day close-out cashflow:", symbol, days)
    for condor in condors:
        open_credit = condor_net_credit(condor)
        logger.info(
            "  condor=%s, open_credit=%s",
            format_condor(condor),
            format_float(open_credit),
        )
        for offset in range(1, days + 1):
            current_date = start_date + timedelta(days=offset)
            expirations = [
                condor.short_put.expiration,
                condor.short_call.expiration,
                condor.long_put.expiration,
                condor.long_call.expiration,
            ]
            expirations = [expiration for expiration in expirations if expiration]
            if expirations and current_date > min(expirations):
                logger.info(
                    "%s: reached expiration %s; stopping closeout",
                    symbol,
                    min(expirations),
                )
                break
            close_price = None
            if current_date in price_index:
                close_price = price_index[current_date].close
            leg_keys = [
                (condor.short_put.expiration, condor.short_put.right, condor.short_put.strike, current_date),
                (condor.short_call.expiration, condor.short_call.right, condor.short_call.strike, current_date),
                (condor.long_put.expiration, condor.long_put.right, condor.long_put.strike, current_date),
                (condor.long_call.expiration, condor.long_call.right, condor.long_call.strike, current_date),
            ]
            if any(key not in quote_index for key in leg_keys):
                logger.error(
                    "%s: missing quotes for %s; skipping closeout",
                    symbol,
                    current_date,
                )
                continue
            close_credit = condor_net_credit(condor, lookup=quote_index, quote_date=current_date)
            if close_credit is None:
                logger.error(
                    "%s: missing mid prices for %s; skipping closeout",
                    symbol,
                    current_date,
                )
                continue
            logger.info(
                "      %s close_date=%s spot_close=%s close_cashflow=%s",
                symbol,
                current_date,
                format_float(close_price),
                format_float(close_credit),
            )


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
    parser.add_argument(
        "--strike-distance-pct",
        type=float,
        default=0.2,
        help="Strike window as +/- percent from spot (e.g. 0.2 for 20%%).",
    )
    parser.add_argument("--max-spread-pct", type=float, default=0.08, help="Maximum spread width as pct of spot.")
    parser.add_argument(
        "--pop-bias-pct",
        type=float,
        default=0.0,
        help="Bias PoP spot by this pct (positive = bullish, negative = bearish).",
    )
    args = parser.parse_args()
    db_path = Path(__file__).parent / "data" / "aapl_options_norm.duckdb"

    configure_logging()
    load_dotenv()
    # describe_db(db_path=db_path)

    price_fetcher = DuckDbPriceFetcher(db_path=db_path)
    quote_fetcher = DuckDbOptionQuoteFetcher(db_path=db_path)
    builder = MarketSnapshotBuilder(price_fetcher=price_fetcher, option_quote_fetcher=quote_fetcher)
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    snapshot = builder.build_for_date(
        args.quote_date,
        symbols=symbols,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        strike_distance_pct=args.strike_distance_pct,
        max_spread_pct=args.max_spread_pct,
    )
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
        condor_candidates = build_iron_condor_candidates(instrument.option_chain)
        logger.info("%s: iron condor candidates=%d", symbol, len(condor_candidates))
        spread_width_limit = price * args.max_spread_pct
        logger.info("%s: spread width limit=%s", symbol, format_float(spread_width_limit))
        quotes_with_strike = [quote for quote in instrument.option_chain.quotes if quote.strike is not None]
        if not quotes_with_strike:
            continue
        otm_calls = [quote for quote in quotes_with_strike if quote.right == "C" and quote.strike > price]
        otm_puts = [quote for quote in quotes_with_strike if quote.right == "P" and quote.strike < price]
        otm_calls_sorted = sorted(otm_calls, key=lambda quote: quote.strike, reverse=True)
        otm_puts_sorted = sorted(otm_puts, key=lambda quote: quote.strike)
        atm_sorted = sorted(quotes_with_strike, key=lambda quote: abs(quote.strike - price))

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

        def bearish_distance(condor: object) -> float:
            return (condor.short_call.strike or 0.0) - price

        def bullish_distance(condor: object) -> float:
            return price - (condor.short_put.strike or 0.0)

        def pop_value(condor: object) -> float:
            return estimate_probability_of_profit(condor, price, bias_pct=args.pop_bias_pct)

        bearish_candidates = [condor for condor in condor_candidates if bearish_distance(condor) >= 0]
        bullish_candidates = [condor for condor in condor_candidates if bullish_distance(condor) >= 0]
        bearish_samples = select_samples(
            bearish_candidates,
            BEARISH_SAMPLE_SIZE,
            lambda condor: (bearish_distance(condor), -pop_value(condor)),
        )
        bullish_samples = select_samples(
            bullish_candidates,
            BULLISH_SAMPLE_SIZE,
            lambda condor: (bullish_distance(condor), -pop_value(condor)),
        )
        pop_samples = select_samples(
            condor_candidates,
            POP_SAMPLE_SIZE,
            lambda condor: (-pop_value(condor),),
        )
        logger.info("%s: bearish samples (%d):", symbol, BEARISH_SAMPLE_SIZE)
        for condor in bearish_samples:
            logger.info(
                "  %s dist=%s pop=%s",
                format_condor(condor),
                format_float(bearish_distance(condor)),
                format_float(pop_value(condor)),
            )
        logger.info("%s: bullish samples (%d):", symbol, BULLISH_SAMPLE_SIZE)
        for condor in bullish_samples:
            logger.info(
                "  %s dist=%s pop=%s",
                format_condor(condor),
                format_float(bullish_distance(condor)),
                format_float(pop_value(condor)),
            )
        logger.info("%s: top PoP samples (%d):", symbol, POP_SAMPLE_SIZE)
        for condor in pop_samples:
            net_credit = condor_net_credit(condor)
            logger.info(
                "  %s pop=%s credit=%s",
                format_condor(condor),
                format_float(pop_value(condor)),
                format_float(net_credit),
            )

        if pop_samples or bullish_samples or bearish_samples:
            start_date = Utils.to_date(args.quote_date)
            logger.info("%s: bearish sample closeout", symbol)
            run_closeout_simulation(symbol, bearish_samples, start_date, quote_fetcher, price_fetcher)
            logger.info("")

            logger.info("%s: bullish sample closeout", symbol)
            run_closeout_simulation(symbol, bullish_samples, start_date, quote_fetcher, price_fetcher)
            logger.info("")

            logger.info("%s: PoP sample closeout", symbol)
            run_closeout_simulation(symbol, pop_samples, start_date, quote_fetcher, price_fetcher)
            logger.info("")


if __name__ == "__main__":
    main()
