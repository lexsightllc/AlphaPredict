"""Train the AlphaPredict model and persist artifacts."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.config import load_config
from src.data_loader import DataSchema, load_train_dataset
from src.models import build_model, train_model
from src.preprocessing import FeaturePreprocessor
from src.utils import get_logger, set_seed, write_json

LOGGER = get_logger("scripts.train")
CONFIG_PATH = Path("config/settings.yaml")


def main() -> None:
    config = load_config(CONFIG_PATH)
    set_seed(config.training.random_seed)

    schema = DataSchema(required_columns=[config.training.target, "date", "symbol"])
    train_df = load_train_dataset(config.paths, schema)

    feature_cols = [col for col in train_df.columns if col not in {schema.date_column, "symbol", config.training.target}]

    preprocessor = FeaturePreprocessor(config.preprocessing)
    processed = preprocessor.fit_transform(
        train_df,
        date_column=schema.date_column,
        group_column="symbol",
        feature_columns=feature_cols,
    )

    processed_path = config.paths.processed_data / "feature_matrix.parquet"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(processed_path, index=False)

    cleaned_path = config.paths.processed_data / "cleaned_train.parquet"
    train_df.to_parquet(cleaned_path, index=False)

    X = processed[preprocessor.fitted_columns].to_numpy()
    y = processed[config.training.target].to_numpy()

    model = build_model(config.model)
    artifacts = train_model(model, X, y)

    config.paths.model_artifact.parent.mkdir(parents=True, exist_ok=True)
    config.paths.scaler_artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.estimator, config.paths.model_artifact)
    joblib.dump(preprocessor, config.paths.scaler_artifact)

    metrics_path = config.paths.statistics_metadata
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, {"training": artifacts.metrics})

    feature_meta = {"features": preprocessor.fitted_columns}
    config.paths.feature_metadata.parent.mkdir(parents=True, exist_ok=True)
    write_json(config.paths.feature_metadata, feature_meta)

    LOGGER.info("Training complete. Artifacts saved to %s", config.paths.artifacts_root)


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
