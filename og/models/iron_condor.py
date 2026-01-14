from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from models.market_data import OptionChain
from models.quotes import OptionQuoteRow


@dataclass(frozen=True)
class IronCondorLegs:
    short_put: OptionQuoteRow
    long_put: OptionQuoteRow
    short_call: OptionQuoteRow
    long_call: OptionQuoteRow


def estimate_probability_of_profit(
    condor: IronCondorLegs,
    spot_price: float,
    risk_free_rate: float = 0.045,
    default_vol: float = 0.30,
    bias_pct: float = 0.0,
) -> float:
    if condor.short_put.strike is None or condor.short_call.strike is None:
        return 0.0
    dte_values = [
        condor.short_put.dte,
        condor.long_put.dte,
        condor.short_call.dte,
        condor.long_call.dte,
    ]
    dte_values = [value for value in dte_values if value is not None]
    if not dte_values:
        return 0.0
    dte = min(dte_values)
    if dte <= 0:
        return 0.0

    volatility = _mean_iv(condor, default_vol)
    biased_spot = spot_price * (1 + bias_pct)
    time_to_exp = max(dte / 365.0, 1e-4)
    prob_below_call = _probability_below_strike(
        biased_spot,
        condor.short_call.strike,
        time_to_exp,
        volatility,
        risk_free_rate,
    )
    prob_below_put = _probability_below_strike(
        biased_spot,
        condor.short_put.strike,
        time_to_exp,
        volatility,
        risk_free_rate,
    )
    return max(0.0, min(1.0, prob_below_call - prob_below_put))


def build_iron_condor_candidates(option_chain: OptionChain) -> list[IronCondorLegs]:
    calls, puts = _split_chain(option_chain.quotes)
    if len(calls) < 2 or len(puts) < 2:
        return []

    candidates: list[IronCondorLegs] = []
    for short_put_index, short_put in enumerate(puts):
        for long_put in puts[:short_put_index]:
            for short_call_index, short_call in enumerate(calls):
                if short_put.strike is None or short_call.strike is None:
                    continue
                if short_put.strike >= short_call.strike:
                    continue
                for long_call in calls[short_call_index + 1 :]:
                    if long_call.strike is None:
                        continue
                    candidates.append(
                        IronCondorLegs(
                            short_put=short_put,
                            long_put=long_put,
                            short_call=short_call,
                            long_call=long_call,
                        )
                    )
    return candidates


def _split_chain(quotes: Sequence[OptionQuoteRow]) -> tuple[list[OptionQuoteRow], list[OptionQuoteRow]]:
    calls = [quote for quote in quotes if quote.right == "C" and quote.strike is not None]
    puts = [quote for quote in quotes if quote.right == "P" and quote.strike is not None]
    calls.sort(key=lambda quote: quote.strike or 0.0)
    puts.sort(key=lambda quote: quote.strike or 0.0)
    return calls, puts


def _mean_iv(condor: IronCondorLegs, fallback: float) -> float:
    iv_values = [
        condor.short_put.iv,
        condor.long_put.iv,
        condor.short_call.iv,
        condor.long_call.iv,
    ]
    iv_values = [value for value in iv_values if value is not None and value > 0]
    if not iv_values:
        return fallback
    return float(sum(iv_values) / len(iv_values))


def _probability_below_strike(
    spot: float,
    strike: float,
    time_to_exp: float,
    volatility: float,
    risk_free_rate: float,
) -> float:
    spot = max(spot, 0.01)
    strike = max(strike, 0.01)
    volatility = min(max(volatility, 0.05), 3.0)

    try:
        d2 = (math.log(spot / strike) + (risk_free_rate - 0.5 * volatility**2) * time_to_exp) / (
            volatility * math.sqrt(time_to_exp)
        )
    except (ValueError, ZeroDivisionError):
        return 0.0

    cdf = 0.5 * (1 + math.erf(-d2 / math.sqrt(2)))
    return max(0.0, min(1.0, cdf))
