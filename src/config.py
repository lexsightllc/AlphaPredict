"""Configuration management for AlphaPredict.

This module provides:
- Type-safe configuration dataclasses
- YAML-based configuration loading
- Environment variable override support
- Configuration validation
- Comprehensive error messages
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import get_logger, read_yaml

logger = get_logger(__name__)


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
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.

    Returns
    -------
    AppConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.
    ValueError
        If configuration is invalid or missing required sections.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info(f"Loading configuration from {path}")

    try:
        payload = read_yaml(path)
    except Exception as e:
        raise ValueError(f"Failed to parse configuration file {path}: {e}")

    required_sections = {
        "paths",
        "preprocessing",
        "training",
        "model",
        "strategy",
        "serving",
    }
    missing = required_sections - set(payload.keys())
    if missing:
        raise ValueError(
            f"Configuration missing required sections: {sorted(missing)}"
        )

    root = path.parent

    try:
        config = AppConfig(
            paths=_as_path_config(payload["paths"], root=root),
            preprocessing=PreprocessingConfig(**payload["preprocessing"]),
            training=TrainingConfig(**payload["training"]),
            model=ModelConfig(**payload["model"]),
            strategy=StrategyConfig(**payload["strategy"]),
            serving=ServingConfig(**payload["serving"]),
        )
        logger.info("Configuration loaded successfully")
        return config
    except TypeError as e:
        raise ValueError(f"Invalid configuration structure: {e}")


def get_config_path() -> Path:
    """Get configuration file path.

    Priority order:
    1. CONFIG_FILE environment variable
    2. Default config/settings.yaml relative to package root
    3. Alternative config/config.yaml

    Returns
    -------
    Path
        Path to the configuration file.

    Raises
    ------
    FileNotFoundError
        If no valid configuration file is found.
    """
    # Check environment variable
    if env_config := os.getenv("CONFIG_FILE"):
        config_path = Path(env_config)
        if config_path.exists():
            logger.info(f"Using config from environment variable: {config_path}")
            return config_path
        logger.warning(f"CONFIG_FILE not found: {config_path}")

    # Check default locations
    package_root = Path(__file__).parent.parent
    default_locations = [
        package_root / "config" / "settings.yaml",
        package_root / "config" / "config.yaml",
    ]

    for location in default_locations:
        if location.exists():
            logger.info(f"Found configuration at: {location}")
            return location

    raise FileNotFoundError(
        f"Configuration file not found. Checked: {default_locations}. "
        "Set CONFIG_FILE environment variable or create config/settings.yaml"
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
    "get_config_path",
]
