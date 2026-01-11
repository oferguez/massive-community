from __future__ import annotations

from datetime import date
from typing import Iterable, Dict, Optional

from fetchers.option_quotes import IOptionQuoteFetcher
from fetchers.prices import IPriceFetcher
from models.market_data import InstrumentSnapshot, MarketSnapshot, OptionChain
from models.prices import InstrumentPriceRow
from utils.conversions import Utils


class MarketSnapshotBuilder:
    def __init__(self, price_fetcher: IPriceFetcher, option_quote_fetcher: IOptionQuoteFetcher) -> None:
        self.price_fetcher = price_fetcher
        self.option_quote_fetcher = option_quote_fetcher

    def build_for_date(
        self,
        quote_date: date | str,
        symbols: Iterable[str],
        min_dte: int,
        max_dte: int,
        strike_distance_pct: float | None = None,
        max_spread_pct: float | None = None,
    ) -> MarketSnapshot:
        target_date = Utils.to_date(quote_date)
        date_str = Utils.to_yyyy_mm_dd(target_date) or target_date.isoformat()

        instruments: Dict[str, InstrumentSnapshot] = {}
        for symbol in symbols:
            price = self._pick_price(symbol, date_str, target_date)
            option_chain = self._build_option_chain(
                symbol,
                date_str,
                target_date,
                min_dte,
                max_dte,
                strike_distance_pct,
                max_spread_pct,
                price,
            )

            instruments[symbol] = InstrumentSnapshot(
                symbol=symbol,
                quote_date=target_date,
                price=price,
                option_chain=option_chain,
            )

        return MarketSnapshot(quote_date=target_date, instruments=instruments)

    def _pick_price(self, symbol: str, date_str: str, target_date: date) -> Optional[InstrumentPriceRow]:
        rows = self.price_fetcher.fetch_prices(ticker=symbol, date_from=date_str, date_to=date_str)
        for row in rows:
            if row.date == target_date:
                return row
        return rows[0] if rows else None

    def _build_option_chain(
        self,
        symbol: str,
        date_str: str,
        target_date: date,
        min_dte: int,
        max_dte: int,
        strike_distance_pct: float | None,
        max_spread_pct: float | None,
        price: Optional[InstrumentPriceRow],
    ) -> Optional[OptionChain]:
        quotes = self.option_quote_fetcher.fetch_option_quotes(
            symbol=symbol,
            date_from=date_str,
            date_to=date_str,
            min_dte=min_dte,
            max_dte=max_dte,
        )
        filtered = [quote for quote in quotes if quote.quote_date == target_date]
        if strike_distance_pct is not None and price and price.close is not None:
            min_strike = price.close * (1 - strike_distance_pct)
            max_strike = price.close * (1 + strike_distance_pct)
            filtered = [
                quote
                for quote in filtered
                if quote.strike is not None and min_strike <= quote.strike <= max_strike
            ]
        if max_spread_pct is not None:
            filtered = [
                quote
                for quote in filtered
                if quote.spread_pct is None or quote.spread_pct <= max_spread_pct
            ]
        if not filtered:
            return None

        as_of = max((quote.as_of for quote in filtered if quote.as_of is not None), default=None)
        return OptionChain(symbol=symbol, quote_date=target_date, as_of=as_of, quotes=filtered)
