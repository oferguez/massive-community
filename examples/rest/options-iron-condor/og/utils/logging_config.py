from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging


def configure_logging(log_dir: str | Path = "logs") -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"contract_fetcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=str(log_file),
        filemode="w",
    )

    logging.getLogger(__name__).info("starting")
    return log_file
