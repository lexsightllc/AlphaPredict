"""Data loading utilities for AlphaPredict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import PathConfig
from .utils import ensure_columns, get_logger

LOGGER = get_logger("data_loader")


@dataclass
class DataSchema:
    required_columns: Iterable[str]
    date_column: str = "date"


def _load_csv(path: Path, *, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    LOGGER.info("Loading %s", path)
    return pd.read_csv(path, parse_dates=parse_dates)


def load_train_dataset(paths: PathConfig, schema: DataSchema) -> pd.DataFrame:
    """Load the training dataset and validate schema."""

    df = _load_csv(paths.external_data / "train.csv", parse_dates=[schema.date_column])
    ensure_columns(df, schema.required_columns)
    return df


def load_test_dataset(paths: PathConfig, schema: DataSchema) -> pd.DataFrame:
    """Load the test dataset used to mirror Kaggle inference."""

    df = _load_csv(paths.external_data / "test.csv", parse_dates=[schema.date_column])
    ensure_columns(df, schema.required_columns)
    return df


def time_series_split(df: pd.DataFrame, *, n_splits: int, test_size: int, gap: int,
                      date_column: str) -> list[tuple[pd.Index, pd.Index]]:
    """Create custom walk-forward splits with a gap between train and validation."""

    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    unique_dates = df[date_column].sort_values().unique()
    splits: list[tuple[pd.Index, pd.Index]] = []
    for split_idx in range(n_splits):
        end = len(unique_dates) - (n_splits - split_idx - 1) * test_size
        start = end - test_size
        if start - gap <= 0:
            break
        train_dates = unique_dates[: start - gap]
        valid_dates = unique_dates[start:end]
        train_idx = df.index[df[date_column].isin(train_dates)]
        valid_idx = df.index[df[date_column].isin(valid_dates)]
        splits.append((train_idx, valid_idx))
    if not splits:
        raise ValueError("Unable to create time-series splits with given parameters")
    return splits


__all__ = ["DataSchema", "load_train_dataset", "load_test_dataset", "time_series_split"]
