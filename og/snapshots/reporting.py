from __future__ import annotations

import logging

from utils.condor_helpers import condor_net_credit, format_condor
from utils.formatting import format_float, format_quote

logger = logging.getLogger(__name__)


def log_option_chain_samples(
    symbol: str,
    quotes: list[object],
    price: float,
    otm_sample_size: int,
    atm_sample_size: int,
) -> None:
    quotes_with_strike = [quote for quote in quotes if quote.strike is not None]
    if not quotes_with_strike:
        return
    otm_calls = [quote for quote in quotes_with_strike if quote.right == "C" and quote.strike > price]
    otm_puts = [quote for quote in quotes_with_strike if quote.right == "P" and quote.strike < price]
    otm_calls_sorted = sorted(otm_calls, key=lambda quote: quote.strike, reverse=True)
    otm_puts_sorted = sorted(otm_puts, key=lambda quote: quote.strike)
    atm_sorted = sorted(quotes_with_strike, key=lambda quote: abs(quote.strike - price))

    logger.info("%s: most OTM calls (%d):", symbol, otm_sample_size)
    for quote in otm_calls_sorted[:otm_sample_size]:
        logger.info("  %s", format_quote(quote, price))
    logger.info("%s: most OTM puts (%d):", symbol, otm_sample_size)
    for quote in otm_puts_sorted[:otm_sample_size]:
        logger.info("  %s", format_quote(quote, price))
    logger.info("%s: near ATM (%d):", symbol, atm_sample_size)
    for quote in atm_sorted[:atm_sample_size]:
        logger.info("  %s", format_quote(quote, price))


def log_condor_samples(
    symbol: str,
    bearish_samples: list[object],
    bullish_samples: list[object],
    pop_samples: list[object],
    bearish_distance,
    bullish_distance,
    pop_value,
    pop_sample_size: int,
) -> None:
    logger.info("%s: bearish samples (%d):", symbol, len(bearish_samples))
    for condor in bearish_samples:
        logger.info(
            "  %s dist=%s pop=%s",
            format_condor(condor),
            format_float(bearish_distance(condor)),
            format_float(pop_value(condor)),
        )
    logger.info("%s: bullish samples (%d):", symbol, len(bullish_samples))
    for condor in bullish_samples:
        logger.info(
            "  %s dist=%s pop=%s",
            format_condor(condor),
            format_float(bullish_distance(condor)),
            format_float(pop_value(condor)),
        )
    logger.info("%s: top PoP samples (%d):", symbol, pop_sample_size)
    for condor in pop_samples:
        net_credit = condor_net_credit(condor)
        logger.info(
            "  %s pop=%s credit=%s",
            format_condor(condor),
            format_float(pop_value(condor)),
            format_float(net_credit),
        )
