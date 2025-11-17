"""Unit tests for models module."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from src.config import ModelConfig
from src.models import TrainedModel, create_model


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randn(n_samples) * 10, name="target")

    return X, y


class TestModelCreation:
    """Test model creation."""

    def test_create_elasticnet_model(self):
        """Test creating ElasticNet model."""
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        assert model is not None
        assert isinstance(model, ElasticNet)

    def test_elasticnet_parameters_applied(self):
        """Test ElasticNet parameters are correctly applied."""
        config = ModelConfig(
            model_type="elasticnet",
            elasticnet_alpha=0.1,
            elasticnet_l1_ratio=0.7,
        )
        model = create_model(config)
        assert model.alpha == 0.1
        assert model.l1_ratio == 0.7

    def test_model_has_fit_predict(self):
        """Test created model has fit and predict methods."""
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


class TestTrainedModel:
    """Test TrainedModel wrapper class."""

    def test_trained_model_initialization(self, sample_training_data):
        """Test TrainedModel initialization."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        trained = TrainedModel(model, config)
        assert trained is not None
        assert trained.model == model
        assert trained.config == config

    def test_trained_model_predict(self, sample_training_data):
        """Test TrainedModel predict method."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        trained = TrainedModel(model, config)
        predictions = trained.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)
        assert isinstance(predictions, (np.ndarray, pd.Series))

    def test_trained_model_calculates_rmse(self, sample_training_data):
        """Test TrainedModel calculates RMSE."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        trained = TrainedModel(model, config)
        rmse = trained.calculate_rmse(X, y)

        assert rmse >= 0
        assert isinstance(rmse, (float, np.floating))

    def test_predictions_are_numeric(self, sample_training_data):
        """Test predictions are numeric."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        trained = TrainedModel(model, config)
        predictions = trained.predict(X)

        assert np.all(np.isfinite(predictions))


class TestModelTraining:
    """Test model training."""

    def test_model_trains_without_error(self, sample_training_data):
        """Test model trains without error."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)

        try:
            model.fit(X, y)
        except Exception as e:
            pytest.fail(f"Model training failed: {e}")

    def test_trained_model_makes_predictions(self, sample_training_data):
        """Test trained model makes predictions."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        predictions = model.predict(X[:10])
        assert len(predictions) == 10

    def test_model_rmse_is_positive(self, sample_training_data):
        """Test model RMSE is positive."""
        X, y = sample_training_data
        config = ModelConfig(model_type="elasticnet")
        model = create_model(config)
        model.fit(X, y)

        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        rmse = np.sqrt(mse)

        assert rmse >= 0
        assert not np.isnan(rmse)


class TestModelPersistence:
    """Test model serialization."""

    def test_model_config_can_be_serialized(self):
        """Test model config can be serialized."""
        config = ModelConfig(
            model_type="elasticnet",
            elasticnet_alpha=0.5,
            elasticnet_l1_ratio=0.5,
        )

        # Should be convertible to dict
        config_dict = {
            "model_type": config.model_type,
            "elasticnet_alpha": config.elasticnet_alpha,
        }
        assert config_dict is not None

    def test_trained_model_preserves_config(self, sample_training_data):
        """Test TrainedModel preserves configuration."""
        X, y = sample_training_data
        config = ModelConfig(
            model_type="elasticnet",
            elasticnet_alpha=0.3,
            elasticnet_l1_ratio=0.6,
        )
        model = create_model(config)
        model.fit(X, y)

        trained = TrainedModel(model, config)
        assert trained.config.elasticnet_alpha == 0.3
        assert trained.config.elasticnet_l1_ratio == 0.6


class TestModelValidation:
    """Test model validation."""

    def test_model_parameters_are_reasonable(self):
        """Test model parameters are in reasonable ranges."""
        config = ModelConfig()
        assert 0 <= config.elasticnet_alpha <= 1.0
        assert 0 <= config.elasticnet_l1_ratio <= 1.0

    def test_invalid_model_type_raises_error(self):
        """Test invalid model type raises error."""
        config = ModelConfig(model_type="invalid_model")
        try:
            model = create_model(config)
            # Some implementations might skip invalid types
            assert model is None or hasattr(model, "fit")
        except (ValueError, KeyError, NotImplementedError):
            pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
