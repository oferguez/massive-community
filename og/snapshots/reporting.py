from __future__ import annotations

from collections.abc import Sequence
import logging

from models.quotes import OptionQuoteRow
from models.condor_sampling import CondorSampleMetrics
from utils.condor_helpers import condor_net_credit, format_condor
from utils.formatting import format_float, format_quote

logger = logging.getLogger(__name__)


def log_option_chain_samples(
    symbol: str,
    quotes: Sequence[OptionQuoteRow],
    price: float,
    otm_sample_size: int,
    atm_sample_size: int,
) -> None:
    quotes_with_strike: list[OptionQuoteRow] = []
    otm_calls: list[OptionQuoteRow] = []
    otm_puts: list[OptionQuoteRow] = []
    for quote in quotes:
        strike = quote.strike
        if strike is None:
            continue
        quotes_with_strike.append(quote)
        if quote.right == "C" and strike > price:
            otm_calls.append(quote)
        elif quote.right == "P" and strike < price:
            otm_puts.append(quote)
    if not quotes_with_strike:
        return

    def strike_value(quote: OptionQuoteRow) -> float:
        return quote.strike if quote.strike is not None else 0.0

    otm_calls_sorted = sorted(otm_calls, key=strike_value, reverse=True)
    otm_puts_sorted = sorted(otm_puts, key=strike_value)
    atm_sorted = sorted(quotes_with_strike, key=lambda quote: abs(strike_value(quote) - price))

    logger.info("%s: most OTM calls (%d):", symbol, otm_sample_size)
    for quote in otm_calls_sorted[:otm_sample_size]:
        logger.info("  %s", format_quote(quote, price))
    logger.info("%s: most OTM puts (%d):", symbol, otm_sample_size)
    for quote in otm_puts_sorted[:otm_sample_size]:
        logger.info("  %s", format_quote(quote, price))
    logger.info("%s: near ATM (%d):", symbol, atm_sample_size)
    for quote in atm_sorted[:atm_sample_size]:
        logger.info("  %s", format_quote(quote, price))


def log_condor_samples(symbol: str, metrics: CondorSampleMetrics, pop_sample_size: int) -> None:
    logger.info("%s: bearish samples (%d):", symbol, len(metrics.bearish_samples))
    for condor in metrics.bearish_samples:
        logger.info(
            "  %s dist=%s pop=%s",
            format_condor(condor),
            format_float(metrics.bearish_distance(condor)),
            format_float(metrics.pop_value(condor)),
        )
    logger.info("%s: bullish samples (%d):", symbol, len(metrics.bullish_samples))
    for condor in metrics.bullish_samples:
        logger.info(
            "  %s dist=%s pop=%s",
            format_condor(condor),
            format_float(metrics.bullish_distance(condor)),
            format_float(metrics.pop_value(condor)),
        )
    logger.info("%s: top PoP samples (%d):", symbol, pop_sample_size)
    for condor in metrics.pop_samples:
        net_credit = condor_net_credit(condor)
        logger.info(
            "  %s pop=%s credit=%s",
            format_condor(condor),
            format_float(metrics.pop_value(condor)),
            format_float(net_credit),
        )
