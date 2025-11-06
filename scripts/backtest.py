"""Run historical backtests for AlphaPredict."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.config import load_config
from src.data_loader import DataSchema, load_test_dataset
from src.evaluation import evaluate_strategy
from src.strategy import position_sizer
from src.utils import get_logger

LOGGER = get_logger("scripts.backtest")
CONFIG_PATH = Path("config/settings.yaml")


def main() -> None:
    config = load_config(CONFIG_PATH)
    schema = DataSchema(required_columns=[config.training.target, "date", "symbol"])
    test_df = load_test_dataset(config.paths, schema)

    model = joblib.load(config.paths.model_artifact)
    preprocessor = joblib.load(config.paths.scaler_artifact)

    processed = preprocessor.transform(
        test_df,
        date_column=schema.date_column,
        group_column="symbol",
        feature_columns=list(preprocessor.base_features),
    )

    raw_predictions = model.predict(processed[preprocessor.fitted_columns])
    sized_positions = position_sizer(raw_predictions, config=config.strategy)
    metrics = evaluate_strategy(sized_positions, processed[config.training.target].to_numpy(),
                                config=config.strategy)

    LOGGER.info("Backtest metrics: %s", metrics)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
