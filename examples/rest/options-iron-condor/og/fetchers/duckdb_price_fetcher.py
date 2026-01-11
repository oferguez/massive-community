from __future__ import annotations

from pathlib import Path
import logging
from typing import List

import duckdb

from fetchers.prices import IPriceFetcher
from models.prices import InstrumentPriceRow
from utils.conversions import Utils
from utils.sql_tools import format_query


class DuckDbPriceFetcher(IPriceFetcher):
    def __init__(self, db_path: Path, table_name: str = "underlying_eod") -> None:
        self.db_path = Path(db_path)
        self.table_name = table_name
        super().__init__()

    def fetch_prices(self, ticker: str, date_from: str, date_to: str) -> List[InstrumentPriceRow]:
        start = Utils.to_date(date_from)
        end = Utils.to_date(date_to)
        if start > end:
            raise ValueError(f"date_from must be <= date_to; got {date_from} > {date_to}")

        query = f"""
            SELECT symbol, quote_date, close, as_of
            FROM {self.table_name}
            WHERE symbol = ?
              AND quote_date BETWEEN ? AND ?
            ORDER BY quote_date
        """

        logger = logging.getLogger(__name__)
        params = [ticker, start, end]
        logger.debug("DuckDB query (prices): %s", format_query(query, params))
        logger.debug("DuckDB params (prices): %s", params)

        with duckdb.connect(self.db_path, read_only=True) as con:
            rows = con.execute(query, params).fetchall()

        return [
            InstrumentPriceRow(
                ticker=row[0],
                date=row[1],
                price=Utils.to_float(row[2], alert_on_none=False),
                close=Utils.to_float(row[2], alert_on_none=False),
                as_of=row[3],
                raw=None,
            )
            for row in rows
        ]
