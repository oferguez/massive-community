from __future__ import annotations

from models.iron_condor import estimate_probability_of_profit
from utils.condor_helpers import select_samples


def build_condor_samples(
    condor_candidates: list[object],
    price: float,
    pop_bias_pct: float,
    bearish_sample_size: int,
    bullish_sample_size: int,
    pop_sample_size: int,
) -> tuple[list[object], list[object], list[object], object, object, object]:
    def bearish_distance(condor: object) -> float:
        return (condor.short_call.strike or 0.0) - price

    def bullish_distance(condor: object) -> float:
        return price - (condor.short_put.strike or 0.0)

    def pop_value(condor: object) -> float:
        return estimate_probability_of_profit(condor, price, bias_pct=pop_bias_pct)

    bearish_candidates = [condor for condor in condor_candidates if bearish_distance(condor) >= 0]
    bullish_candidates = [condor for condor in condor_candidates if bullish_distance(condor) >= 0]
    bearish_samples = select_samples(
        bearish_candidates,
        bearish_sample_size,
        lambda condor: (bearish_distance(condor), -pop_value(condor)),
    )
    bullish_samples = select_samples(
        bullish_candidates,
        bullish_sample_size,
        lambda condor: (bullish_distance(condor), -pop_value(condor)),
    )
    pop_samples = select_samples(
        condor_candidates,
        pop_sample_size,
        lambda condor: (-pop_value(condor),),
    )
    return bearish_samples, bullish_samples, pop_samples, bearish_distance, bullish_distance, pop_value
