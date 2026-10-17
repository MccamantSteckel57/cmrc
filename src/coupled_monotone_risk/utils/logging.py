"""Process-wide logging with a single idempotent handler."""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED: set[str] = set()
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("CMRC_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


def configure(log_level: str) -> None:
    for name in list(_CONFIGURED):
        logging.getLogger(name).setLevel(log_level.upper())
