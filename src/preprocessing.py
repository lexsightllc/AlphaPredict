"""Preprocessing utilities for AlphaPredict."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import PreprocessingConfig
from .utils import get_logger

LOGGER = get_logger("preprocessing")


class FeaturePreprocessor:
    """Constructs the feature matrix used for training and inference."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.imputer = SimpleImputer(strategy=config.imputation_strategy)
        self.scaler = StandardScaler() if config.scaler == "standard" else None
        self.base_features: list[str] = []
        self.fitted_columns: list[str] = []

    def _winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        lower = df.quantile(self.config.winsor_limits["lower"])
        upper = df.quantile(self.config.winsor_limits["upper"])
        return df.clip(lower=lower, upper=upper, axis=1)

    def _add_lags(self, df: pd.DataFrame, *, group_col: str, target_cols: Iterable[str]) -> list[str]:
        created: list[str] = []
        for lag in self.config.lags:
            for col in target_cols:
                name = f"{col}_lag_{lag}"
                df[name] = df.groupby(group_col)[col].shift(lag)
                created.append(name)
        return created

    def _add_roll_stats(self, df: pd.DataFrame, *, group_col: str, target_cols: Iterable[str]) -> list[str]:
        created: list[str] = []
        for window in self.config.rolling_windows:
            roll = df.groupby(group_col)[list(target_cols)].rolling(window, min_periods=1)
            means = roll.mean().reset_index(level=0, drop=True)
            stds = roll.std().reset_index(level=0, drop=True)
            for col in target_cols:
                mean_name = f"{col}_roll_mean_{window}"
                std_name = f"{col}_roll_std_{window}"
                df[mean_name] = means[col]
                df[std_name] = stds[col]
                created.extend([mean_name, std_name])
        return created

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        date_column: str,
        group_column: str,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        self.base_features = list(feature_columns)
        df = df.sort_values(date_column).copy()
        df = self._winsorize(df)
        lag_cols = self._add_lags(df, group_col=group_column, target_cols=self.base_features)
        roll_cols = self._add_roll_stats(df, group_col=group_column, target_cols=self.base_features)
        all_features = list(self.base_features) + lag_cols + roll_cols

        features = df[all_features]
        LOGGER.info("Imputing %d features", features.shape[1])
        imputed = pd.DataFrame(
            self.imputer.fit_transform(features),
            index=df.index,
            columns=all_features,
        )
        if self.scaler is not None:
            LOGGER.info("Scaling features using %s", self.config.scaler)
            scaled = pd.DataFrame(
                self.scaler.fit_transform(imputed),
                index=df.index,
                columns=all_features,
            )
        else:
            scaled = imputed

        self.fitted_columns = list(scaled.columns)
        df[self.fitted_columns] = scaled
        return df

    def transform(
        self,
        df: pd.DataFrame,
        *,
        date_column: str,
        group_column: str,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.fitted_columns:
            raise RuntimeError("Preprocessor must be fit before calling transform")
        base_features = list(feature_columns) if feature_columns is not None else self.base_features
        if not base_features:
            raise RuntimeError("Base features are undefined. Call fit_transform first.")

        df = df.sort_values(date_column).copy()
        df = self._winsorize(df)
        lag_cols = self._add_lags(df, group_col=group_column, target_cols=base_features)
        roll_cols = self._add_roll_stats(df, group_col=group_column, target_cols=base_features)
        all_features = list(base_features) + lag_cols + roll_cols

        features = df[all_features]
        imputed = pd.DataFrame(
            self.imputer.transform(features),
            index=df.index,
            columns=all_features,
        )
        if self.scaler is not None:
            scaled = pd.DataFrame(
                self.scaler.transform(imputed),
                index=df.index,
                columns=all_features,
            )
        else:
            scaled = imputed
        df[self.fitted_columns] = scaled[self.fitted_columns]
        return df


def align_for_inference(df: pd.DataFrame, *, feature_window_days: int, date_column: str) -> pd.DataFrame:
    """Keep only rows with the latest available feature window for inference."""

    cutoff = df[date_column].max() - np.timedelta64(feature_window_days, "D")
    return df[df[date_column] >= cutoff].copy()


__all__ = ["FeaturePreprocessor", "align_for_inference"]
