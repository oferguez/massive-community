from __future__ import annotations

from pathlib import Path
import argparse
import logging

from dotenv import load_dotenv

from fetchers.duckdb_option_quote_fetcher import DuckDbOptionQuoteFetcher
from fetchers.duckdb_price_fetcher import DuckDbPriceFetcher
from models.condor_sampling import build_condor_samples
from models.iron_condor import build_iron_condor_candidates
from simulations.closeout import run_closeout_simulation
from snapshots.reporting import log_condor_samples, log_option_chain_samples
from utils.formatting import format_float
from utils.conversions import Utils
from utils.logging_config import configure_logging
from utils.market_snapshot_builder import MarketSnapshotBuilder

logger = logging.getLogger(__name__)

OTM_SAMPLE_SIZE = 5
ATM_SAMPLE_SIZE = 8
BEARISH_SAMPLE_SIZE = 4
BULLISH_SAMPLE_SIZE = 4
POP_SAMPLE_SIZE = 4


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
    parser.add_argument(
        "--closeout-csv-dir",
        default=None,
        help="Directory to write closeout CSVs (set to empty to disable).",
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
        log_option_chain_samples(
            symbol,
            instrument.option_chain.quotes,
            price,
            OTM_SAMPLE_SIZE,
            ATM_SAMPLE_SIZE,
        )
        metrics = build_condor_samples(
            condor_candidates,
            price,
            args.pop_bias_pct,
            BEARISH_SAMPLE_SIZE,
            BULLISH_SAMPLE_SIZE,
            POP_SAMPLE_SIZE,
        )
        log_condor_samples(symbol, metrics, POP_SAMPLE_SIZE)

        if metrics.pop_samples or metrics.bullish_samples or metrics.bearish_samples:
            start_date = Utils.to_date(args.quote_date)
            csv_dir = None if args.closeout_csv_dir is None else args.closeout_csv_dir.strip()
            output_dir = Path(csv_dir) if csv_dir else None
            logger.info("%s: bearish sample closeout", symbol)
            run_closeout_simulation(
                symbol,
                metrics.bearish_samples,
                start_date,
                quote_fetcher,
                price_fetcher,
                "bearish",
                output_dir,
            )
            logger.info("")

            logger.info("%s: bullish sample closeout", symbol)
            run_closeout_simulation(
                symbol,
                metrics.bullish_samples,
                start_date,
                quote_fetcher,
                price_fetcher,
                "bullish",
                output_dir,
            )
            logger.info("")

            logger.info("%s: PoP sample closeout", symbol)
            run_closeout_simulation(
                symbol,
                metrics.pop_samples,
                start_date,
                quote_fetcher,
                price_fetcher,
                "pop",
                output_dir,
            )
            logger.info("")


if __name__ == "__main__":
    main()
