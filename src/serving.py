"""Inference API implementation for AlphaPredict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from joblib import load

from .config import AppConfig, load_config
from .preprocessing import FeaturePreprocessor, align_for_inference
from .utils import get_logger

LOGGER = get_logger("serving")

CONFIG_PATH = Path("config/settings.yaml")


class InferenceService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.fitted = False

    def load_artifacts(self) -> None:
        LOGGER.info("Loading artifacts from %s", self.config.paths.model_artifact)
        if not self.config.paths.model_artifact.exists():
            raise FileNotFoundError("Model artifact is missing. Train the model first.")
        self.model = load(self.config.paths.model_artifact)
        self.preprocessor = load(self.config.paths.scaler_artifact)
        self.fitted = True

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.fitted or self.model is None or self.preprocessor is None:
            raise RuntimeError("InferenceService is not initialized. Call load_artifacts().")

        df = pd.DataFrame(payload)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df = align_for_inference(
            df,
            feature_window_days=self.config.serving.feature_window_days,
            date_column="date",
        )
        processed = self.preprocessor.transform(
            df,
            date_column="date",
            group_column="symbol",
            feature_columns=list(self.preprocessor.base_features),
        )
        features = processed[self.preprocessor.fitted_columns]
        predictions = self.model.predict(features)
        return {"predictions": predictions.tolist()}


service = InferenceService(load_config(CONFIG_PATH))
app = FastAPI(title="AlphaPredict Inference API")


@app.on_event("startup")
def startup_event() -> None:  # pragma: no cover - runtime hook
    try:
        service.load_artifacts()
    except FileNotFoundError as exc:
        LOGGER.warning("Skipping artifact load: %s", exc)


@app.post("/predict")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return service.predict(payload)
    except Exception as exc:  # pragma: no cover - FastAPI handles logging
        raise HTTPException(status_code=400, detail=str(exc))


__all__ = ["app", "service", "InferenceService"]
