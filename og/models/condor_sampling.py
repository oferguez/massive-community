from __future__ import annotations

from dataclasses import dataclass

from models.iron_condor import IronCondorLegs, estimate_probability_of_profit
from utils.condor_helpers import select_samples


@dataclass(frozen=True)
class CondorSampleMetrics:
    price: float
    pop_bias_pct: float
    bearish_samples: list[IronCondorLegs]
    bullish_samples: list[IronCondorLegs]
    pop_samples: list[IronCondorLegs]

    def bearish_distance(self, condor: IronCondorLegs) -> float:
        return (condor.short_call.strike or 0.0) - self.price

    def bullish_distance(self, condor: IronCondorLegs) -> float:
        return self.price - (condor.short_put.strike or 0.0)

    def pop_value(self, condor: IronCondorLegs) -> float:
        return estimate_probability_of_profit(condor, self.price, bias_pct=self.pop_bias_pct)


def build_condor_samples(
    condor_candidates: list[IronCondorLegs],
    price: float,
    pop_bias_pct: float,
    bearish_sample_size: int,
    bullish_sample_size: int,
    pop_sample_size: int,
) -> CondorSampleMetrics:
    metrics = CondorSampleMetrics(
        price=price,
        pop_bias_pct=pop_bias_pct,
        bearish_samples=[],
        bullish_samples=[],
        pop_samples=[],
    )
    bearish_candidates = [condor for condor in condor_candidates if metrics.bearish_distance(condor) >= 0]
    bullish_candidates = [condor for condor in condor_candidates if metrics.bullish_distance(condor) >= 0]
    bearish_samples = select_samples(
        bearish_candidates,
        bearish_sample_size,
        lambda condor: (metrics.bearish_distance(condor), -metrics.pop_value(condor)),
    )
    bullish_samples = select_samples(
        bullish_candidates,
        bullish_sample_size,
        lambda condor: (metrics.bullish_distance(condor), -metrics.pop_value(condor)),
    )
    pop_samples = select_samples(
        condor_candidates,
        pop_sample_size,
        lambda condor: (-metrics.pop_value(condor),),
    )
    return CondorSampleMetrics(
        price=price,
        pop_bias_pct=pop_bias_pct,
        bearish_samples=bearish_samples,
        bullish_samples=bullish_samples,
        pop_samples=pop_samples,
    )
