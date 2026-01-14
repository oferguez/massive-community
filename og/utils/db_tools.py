from __future__ import annotations

import logging
from pathlib import Path
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

def describe_db(db_path:Path) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.expand_frame_repr", False)

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    con = duckdb.connect(database=str(db_path), read_only=True)
    df = con.execute("SELECT * FROM option_quotes LIMIT(10)").fetchdf()
    logger.info("Sample:\n%s", df.describe().to_string())
    