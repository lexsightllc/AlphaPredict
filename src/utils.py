"""Shared utility helpers for the AlphaPredict project."""
from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

import numpy as np
import yaml


_LOGGER_NAME = "alphapredict"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Optional logger name. Defaults to the project logger hierarchy.
    """

    logger_name = f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
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
    "get_logger",
    "set_seed",
    "timer",
    "read_yaml",
    "write_json",
    "ensure_columns",
]
