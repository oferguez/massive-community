from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import csv
import logging

from fetchers.duckdb_option_quote_fetcher import DuckDbOptionQuoteFetcher
from fetchers.duckdb_price_fetcher import DuckDbPriceFetcher
from utils.condor_helpers import condor_net_credit, format_condor
from utils.conversions import Utils
from utils.formatting import format_float, normalize_date

logger = logging.getLogger(__name__)


def write_closeout_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "symbol",
        "sample_label",
        "condor",
        "start_date",
        "close_date",
        "days_from_open",
        "short_put_strike",
        "long_put_strike",
        "short_call_strike",
        "long_call_strike",
        "expiration",
        "open_credit",
        "close_credit",
        "spot_close",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_closeout_simulation(
    symbol: str,
    condors: list[object],
    start_date: object,
    quote_fetcher: DuckDbOptionQuoteFetcher,
    price_fetcher: DuckDbPriceFetcher,
    sample_label: str,
    output_dir: Path | None,
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

    rows: list[dict[str, object]] = []
    logger.info("  %s: %d-day close-out cashflow:", symbol, days)
    for condor in condors:
        open_credit = condor_net_credit(condor)
        expirations = [
            condor.short_put.expiration,
            condor.short_call.expiration,
            condor.long_put.expiration,
            condor.long_call.expiration,
        ]
        expirations = [expiration for expiration in expirations if expiration]
        expiration = min(expirations) if expirations else None
        base_row = {
            "symbol": symbol,
            "sample_label": sample_label,
            "condor": format_condor(condor),
            "start_date": start_str,
            "short_put_strike": condor.short_put.strike,
            "long_put_strike": condor.long_put.strike,
            "short_call_strike": condor.short_call.strike,
            "long_call_strike": condor.long_call.strike,
            "expiration": normalize_date(expiration),
            "open_credit": open_credit,
        }
        logger.info(
            "  condor=%s, open_credit=%s",
            format_condor(condor),
            format_float(open_credit),
        )
        for offset in range(1, days + 1):
            current_date = start_date + timedelta(days=offset)
            if expirations and current_date > min(expirations):
                logger.info(
                    "      %s: reached expiration %s; stopping closeout",
                    symbol,
                    min(expirations),
                )
                rows.append(
                    {
                        **base_row,
                        "close_date": normalize_date(current_date),
                        "days_from_open": offset,
                        "spot_close": None,
                        "close_credit": None,
                        "status": "expired_stop",
                    }
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
                rows.append(
                    {
                        **base_row,
                        "close_date": normalize_date(current_date),
                        "days_from_open": offset,
                        "spot_close": close_price,
                        "close_credit": None,
                        "status": "missing_quotes",
                    }
                )
                continue
            close_credit = condor_net_credit(condor, lookup=quote_index, quote_date=current_date)
            if close_credit is None:
                logger.error(
                    "%s: missing mid prices for %s; skipping closeout",
                    symbol,
                    current_date,
                )
                rows.append(
                    {
                        **base_row,
                        "close_date": normalize_date(current_date),
                        "days_from_open": offset,
                        "spot_close": close_price,
                        "close_credit": None,
                        "status": "missing_mid",
                    }
                )
                continue
            logger.info(
                "      %s close_date=%s spot_close=%s close_cashflow=%s",
                symbol,
                current_date,
                format_float(close_price),
                format_float(close_credit),
            )
            rows.append(
                {
                    **base_row,
                    "close_date": normalize_date(current_date),
                    "days_from_open": offset,
                    "spot_close": close_price,
                    "close_credit": close_credit,
                    "status": "ok",
                }
            )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"closeout_{symbol}_{sample_label}_{start_str}.csv"
        write_closeout_csv(rows, output_path)
        logger.info("Wrote closeout CSV to %s", output_path)
