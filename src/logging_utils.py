"""Logging configuration shared by the notebooks and the pipeline modules.

Replaces the ``utils/logger.py`` class in the legacy project with the standard
library equivalent: modules create their own ``logging.getLogger(__name__)`` and
the entry point (a notebook, usually) calls :func:`configure_logging` once.
"""

import logging
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Send log records to the console and to a date-stamped file.

    Calling this more than once replaces the handlers rather than adding to
    them, so re-running a notebook cell does not duplicate every message.

    Args:
        log_dir: Directory for the log file. Created if it does not exist.
        level: Minimum level to emit.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d}_log.txt"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)
