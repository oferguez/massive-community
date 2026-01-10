from __future__ import annotations

from pathlib import Path
from typing import List

import duckdb

from fetchers.option_quotes import IOptionQuoteFetcher
from models.quotes import OptionQuoteRow
from utils.conversions import Utils


class DuckDbOptionQuoteFetcher(IOptionQuoteFetcher):
    def __init__(self, db_path: Path, table_name: str = "option_quotes") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        super().__init__()

    def fetch_option_quotes(self, symbol: str, date_from: str, date_to: str, min_dte: int, max_dte: int) -> List[OptionQuoteRow]:
        start = Utils.to_date(date_from)
        end = Utils.to_date(date_to)
        if start > end:
            raise ValueError(f"date_from must be <= date_to; got {date_from} > {date_to}")

        query = f"""
            SELECT
                symbol,
                quote_date,
                as_of,
                expiration,
                dte,
                "right",
                strike,
                bid,
                ask,
                last,
                iv,
                volume,
                size_raw,
                delta,
                gamma,
                vega,
                theta,
                rho,
                bid_size,
                ask_size,
                mid,
                spread,
                spread_pct,
                is_missing_market,
                is_crossed
            FROM {self.table_name}
            WHERE symbol = ?
              AND quote_date BETWEEN ? AND ?
              AND dte >= ?
              AND dte < ?
            ORDER BY quote_date, expiration, strike, "right"
        """

        with duckdb.connect(self.db_path, read_only=True) as con:
            rows = con.execute(query, [symbol, start, end, min_dte, max_dte]).fetchall()

        return [
            OptionQuoteRow(
                symbol=row[0],
                quote_date=row[1],
                as_of=row[2],
                expiration=row[3],
                dte=Utils.to_int(row[4], alert_on_none=False),
                right=row[5],
                strike=Utils.to_float(row[6], alert_on_none=False),
                bid=Utils.to_float(row[7], alert_on_none=False),
                ask=Utils.to_float(row[8], alert_on_none=False),
                last=Utils.to_float(row[9], alert_on_none=False),
                iv=Utils.to_float(row[10], alert_on_none=False),
                volume=Utils.to_int(row[11], alert_on_none=False),
                size_raw=row[12],
                delta=Utils.to_float(row[13], alert_on_none=False),
                gamma=Utils.to_float(row[14], alert_on_none=False),
                vega=Utils.to_float(row[15], alert_on_none=False),
                theta=Utils.to_float(row[16], alert_on_none=False),
                rho=Utils.to_float(row[17], alert_on_none=False),
                bid_size=Utils.to_int(row[18], alert_on_none=False),
                ask_size=Utils.to_int(row[19], alert_on_none=False),
                mid=Utils.to_float(row[20], alert_on_none=False),
                spread=Utils.to_float(row[21], alert_on_none=False),
                spread_pct=Utils.to_float(row[22], alert_on_none=False),
                is_missing_market=bool(row[23]) if row[23] is not None else None,
                is_crossed=bool(row[24]) if row[24] is not None else None,
                raw=None,
            )
            for row in rows
        ]
