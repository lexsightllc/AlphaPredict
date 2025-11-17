"""Unit tests for configuration module."""
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AppConfig,
    ModelConfig,
    PathConfig,
    PreprocessingConfig,
    ServingConfig,
    StrategyConfig,
    TrainingConfig,
    load_config,
)


class TestPathConfig:
    """Test PathConfig dataclass."""

    def test_path_config_initialization(self, tmp_path):
        """Test PathConfig creates valid paths."""
        config = PathConfig(
            data_root=tmp_path / "data",
            external_data=tmp_path / "external",
            processed_data=tmp_path / "processed",
            artifacts_root=tmp_path / "artifacts",
            model_artifact=tmp_path / "model.pkl",
            scaler_artifact=tmp_path / "scaler.pkl",
            feature_metadata=tmp_path / "features.json",
            statistics_metadata=tmp_path / "stats.json",
        )
        assert isinstance(config.data_root, Path)
        assert isinstance(config.external_data, Path)

    def test_path_config_can_be_created(self):
        """Test PathConfig can be instantiated."""
        config = PathConfig(
            data_root=Path("/tmp/data"),
            external_data=Path("/tmp/external"),
            processed_data=Path("/tmp/processed"),
            artifacts_root=Path("/tmp/artifacts"),
            model_artifact=Path("/tmp/model.pkl"),
            scaler_artifact=Path("/tmp/scaler.pkl"),
            feature_metadata=Path("/tmp/features.json"),
            statistics_metadata=Path("/tmp/stats.json"),
        )
        assert config is not None


class TestPreprocessingConfig:
    """Test PreprocessingConfig dataclass."""

    def test_preprocessing_defaults(self):
        """Test preprocessing configuration."""
        config = PreprocessingConfig(
            lags=[1, 5, 21],
            rolling_windows=[5, 21, 63],
            winsor_limits={"lower": 0.01, "upper": 0.99},
            imputation_strategy="forward_fill",
            scaler="standard",
        )
        assert config.lags == [1, 5, 21]
        assert config.rolling_windows == [5, 21, 63]

    def test_preprocessing_custom_values(self):
        """Test custom preprocessing configuration."""
        config = PreprocessingConfig(
            lags=[1, 2, 3],
            rolling_windows=[5, 10],
            winsor_limits={"lower": 0.05, "upper": 0.95},
            imputation_strategy="mean",
            scaler="minmax",
        )
        assert config.lags == [1, 2, 3]
        assert config.rolling_windows == [5, 10]
        assert config.scaler == "minmax"


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_elasticnet(self):
        """Test model configuration for ElasticNet."""
        config = ModelConfig(
            type="elasticnet",
            params={"alpha": 0.5, "l1_ratio": 0.5},
        )
        assert config.type == "elasticnet"
        assert config.params["alpha"] == 0.5

    def test_model_config_lightgbm(self):
        """Test model configuration for LightGBM."""
        config = ModelConfig(
            type="lightgbm",
            params={"n_estimators": 100, "learning_rate": 0.01},
        )
        assert config.type == "lightgbm"
        assert config.params["n_estimators"] == 100


class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_strategy_config_defaults(self):
        """Test default strategy configuration."""
        config = StrategyConfig(
            leverage_min=0.0,
            leverage_max=2.0,
            turnover_penalty=0.05,
            volatility_cap=0.03,
        )
        assert config.leverage_min == 0.0
        assert config.leverage_max == 2.0
        assert config.turnover_penalty == 0.05

    def test_strategy_config_bounds(self):
        """Test strategy leverage bounds are reasonable."""
        config = StrategyConfig(
            leverage_min=0.5,
            leverage_max=1.5,
            turnover_penalty=0.05,
            volatility_cap=0.03,
        )
        assert config.leverage_min < config.leverage_max


class TestServingConfig:
    """Test ServingConfig dataclass."""

    def test_serving_config_defaults(self):
        """Test serving configuration."""
        config = ServingConfig(
            max_batch_size=1024,
            inference_timeout_ms=50,
            feature_window_days=60,
        )
        assert config.max_batch_size == 1024
        assert config.inference_timeout_ms == 50
        assert config.feature_window_days == 60


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_creation(self):
        """Test training configuration."""
        config = TrainingConfig(
            target="returns",
            validation_splits=5,
            validation_gap=7,
            max_train_days=None,
            random_seed=42,
        )
        assert config.random_seed == 42
        assert config.validation_splits == 5


class TestConfigLoading:
    """Integration tests for loading configuration from YAML."""

    def test_config_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        # Create a minimal config YAML file
        config_data = {
            "paths": {
                "data_root": "data",
                "external_data": "data/External",
                "processed_data": "data/Processed",
                "artifacts_root": "artifacts",
                "model_artifact": "artifacts/models/model.pkl",
                "scaler_artifact": "artifacts/scalers/scaler.pkl",
                "feature_metadata": "artifacts/metadata/features.json",
                "statistics_metadata": "artifacts/metadata/statistics.json",
            },
            "preprocessing": {
                "lags": [1, 5, 21],
                "rolling_windows": [5, 21, 63],
                "winsor_limits": {"lower": 0.01, "upper": 0.99},
                "imputation_strategy": "forward_fill",
                "scaler": "standard",
            },
            "training": {
                "target": "returns",
                "validation_splits": 5,
                "validation_gap": 7,
                "max_train_days": None,
                "random_seed": 42,
            },
            "model": {
                "type": "elasticnet",
                "params": {"alpha": 0.5, "l1_ratio": 0.5},
            },
            "strategy": {
                "leverage_min": 0.0,
                "leverage_max": 2.0,
                "turnover_penalty": 0.05,
                "volatility_cap": 0.03,
            },
            "serving": {
                "max_batch_size": 1024,
                "inference_timeout_ms": 50,
                "feature_window_days": 60,
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config is not None
        assert isinstance(config, AppConfig)

    def test_config_all_sections_present(self, tmp_path):
        """Test all configuration sections are loaded."""
        config_data = {
            "paths": {
                "data_root": "data",
                "external_data": "data/External",
                "processed_data": "data/Processed",
                "artifacts_root": "artifacts",
                "model_artifact": "artifacts/models/model.pkl",
                "scaler_artifact": "artifacts/scalers/scaler.pkl",
                "feature_metadata": "artifacts/metadata/features.json",
                "statistics_metadata": "artifacts/metadata/statistics.json",
            },
            "preprocessing": {
                "lags": [1, 5, 21],
                "rolling_windows": [5, 21, 63],
                "winsor_limits": {"lower": 0.01, "upper": 0.99},
                "imputation_strategy": "forward_fill",
                "scaler": "standard",
            },
            "training": {
                "target": "returns",
                "validation_splits": 5,
                "validation_gap": 7,
                "max_train_days": None,
                "random_seed": 42,
            },
            "model": {
                "type": "elasticnet",
                "params": {"alpha": 0.5, "l1_ratio": 0.5},
            },
            "strategy": {
                "leverage_min": 0.0,
                "leverage_max": 2.0,
                "turnover_penalty": 0.05,
                "volatility_cap": 0.03,
            },
            "serving": {
                "max_batch_size": 1024,
                "inference_timeout_ms": 50,
                "feature_window_days": 60,
            },
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_file)
        assert config.paths is not None
        assert config.preprocessing is not None
        assert config.model is not None
        assert config.strategy is not None
        assert config.serving is not None
        assert config.training is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
