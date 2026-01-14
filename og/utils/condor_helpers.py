from __future__ import annotations

from models.iron_condor import IronCondorLegs
from models.quotes import OptionQuoteRow
from utils.formatting import format_float


def quote_mid(quote: OptionQuoteRow) -> float | None:
    if quote.mid is not None:
        return quote.mid
    if quote.bid is not None and quote.ask is not None:
        return (quote.bid + quote.ask) / 2
    return None


def format_condor(condor: IronCondorLegs) -> str:
    expiration = condor.min_expiration()
    return (
        "["
        f"long_put={format_float(condor.long_put.strike)}@{format_float(quote_mid(condor.long_put))} "
        f"short_put={format_float(condor.short_put.strike)}@{format_float(quote_mid(condor.short_put))} "
        f"short_call={format_float(condor.short_call.strike)}@{format_float(quote_mid(condor.short_call))} "
        f"long_call={format_float(condor.long_call.strike)}@{format_float(quote_mid(condor.long_call))} "
        f"exp={expiration} "
        "]"
    )


def select_samples(
    candidates: list[IronCondorLegs],
    size: int,
    key_fn,
) -> list[IronCondorLegs]:
    samples: list[IronCondorLegs] = []
    seen: set[
        tuple[float | None, float | None, float | None, float | None, int | None, object | None, object | None]
    ] = set()
    for condor in sorted(candidates, key=key_fn):
        signature = condor.signature()
        if signature in seen:
            continue
        seen.add(signature)
        samples.append(condor)
        if len(samples) >= size:
            break
    return samples


def condor_net_credit(
    condor: IronCondorLegs,
    lookup: dict[tuple[object, object, object, object], OptionQuoteRow] | None = None,
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
