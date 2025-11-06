"""Configuration management for AlphaPredict."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import read_yaml


@dataclass
class PathConfig:
    data_root: Path
    external_data: Path
    processed_data: Path
    artifacts_root: Path
    model_artifact: Path
    scaler_artifact: Path
    feature_metadata: Path
    statistics_metadata: Path


@dataclass
class PreprocessingConfig:
    lags: list[int]
    rolling_windows: list[int]
    winsor_limits: Dict[str, float]
    imputation_strategy: str
    scaler: str


@dataclass
class TrainingConfig:
    target: str
    validation_splits: int
    validation_gap: int
    max_train_days: Optional[int]
    random_seed: int


@dataclass
class ModelConfig:
    type: str
    params: Dict[str, Any]


@dataclass
class StrategyConfig:
    leverage_min: float
    leverage_max: float
    turnover_penalty: float
    volatility_cap: float


@dataclass
class ServingConfig:
    max_batch_size: int
    inference_timeout_ms: int
    feature_window_days: int


@dataclass
class AppConfig:
    paths: PathConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model: ModelConfig
    strategy: StrategyConfig
    serving: ServingConfig


def _as_path_config(payload: Dict[str, Any], *, root: Path) -> PathConfig:
    return PathConfig(
        data_root=root / payload["data_root"],
        external_data=root / payload["external_data"],
        processed_data=root / payload["processed_data"],
        artifacts_root=root / payload["artifacts_root"],
        model_artifact=root / payload["model_artifact"],
        scaler_artifact=root / payload["scaler_artifact"],
        feature_metadata=root / payload["feature_metadata"],
        statistics_metadata=root / payload["statistics_metadata"],
    )


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""

    payload = read_yaml(path)
    root = path.parent
    return AppConfig(
        paths=_as_path_config(payload["paths"], root=root),
        preprocessing=PreprocessingConfig(**payload["preprocessing"]),
        training=TrainingConfig(**payload["training"]),
        model=ModelConfig(**payload["model"]),
        strategy=StrategyConfig(**payload["strategy"]),
        serving=ServingConfig(**payload["serving"]),
    )


__all__ = [
    "AppConfig",
    "PathConfig",
    "PreprocessingConfig",
    "TrainingConfig",
    "ModelConfig",
    "StrategyConfig",
    "ServingConfig",
    "load_config",
]
