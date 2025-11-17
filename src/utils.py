"""Shared utility helpers for the AlphaPredict project."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

import numpy as np
import yaml


_LOGGER_NAME = "alphapredict"


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> None:
    """Configure root logger with console and optional file handlers.

    Parameters
    ----------
    level : str, optional
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Defaults to environment variable LOG_LEVEL or INFO.
    log_file : Path, optional
        Path to log file. If provided, logs are also written to file.
    """
    # Get log level from parameter or environment
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Clear existing handlers
    root_logger.handlers = [console_handler]

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str, optional
        Optional logger name. Defaults to the project logger hierarchy.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    The logger is automatically added to the root logger's hierarchy.
    Log level can be controlled via LOG_LEVEL environment variable.
    """

    logger_name = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
    logger = logging.getLogger(logger_name)
    if not logger.handlers and logger_name == _LOGGER_NAME:
        # Configure root logger only once
        configure_logging()
    return logger


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


@contextlib.contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """Measure elapsed time for a code block."""

    logger = get_logger("timer")
    start = time.perf_counter()
    logger.info("Started %s", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Finished %s in %.2fs", name, elapsed)


def read_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents."""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialize `payload` to JSON, supporting dataclasses out of the box."""

    if is_dataclass(payload):
        payload = asdict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, sort_keys=True)


def ensure_columns(df, required_columns: Iterable[str]) -> None:
    """Validate that `df` contains the required columns."""

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")


__all__ = [
    "configure_logging",
    "get_logger",
    "set_seed",
    "timer",
    "read_yaml",
    "write_json",
    "ensure_columns",
]
