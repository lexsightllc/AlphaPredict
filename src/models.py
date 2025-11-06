"""Model interfaces for AlphaPredict."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .utils import get_logger

LOGGER = get_logger("models")


class Model(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class ModelArtifacts:
    estimator: Model
    metrics: Dict[str, float]


def build_model(config: ModelConfig) -> Model:
    """Instantiate a model based on configuration."""

    model_type = config.type.lower()
    if model_type == "elastic_net":
        alpha = config.params.get("alpha", 0.5)
        l1_ratio = config.params.get("l1_ratio", 0.5)
        LOGGER.info("Building ElasticNet model alpha=%s l1_ratio=%s", alpha, l1_ratio)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("model", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)),
            ]
        )
    if model_type == "lightgbm":  # pragma: no cover - optional dependency
        try:
            import lightgbm as lgb
        except ModuleNotFoundError as exc:
            raise RuntimeError("LightGBM is not installed") from exc
        LOGGER.info("Building LightGBM model with params %s", config.params)
        return lgb.LGBMRegressor(**config.params)
    raise ValueError(f"Unsupported model type: {config.type}")


def train_model(model: Model, X: np.ndarray, y: np.ndarray) -> ModelArtifacts:
    LOGGER.info("Training model on %s samples", X.shape[0])
    model.fit(X, y)
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    LOGGER.info("Training RMSE: %.4f", rmse)
    return ModelArtifacts(estimator=model, metrics={"rmse": rmse})


__all__ = ["Model", "ModelArtifacts", "build_model", "train_model"]
