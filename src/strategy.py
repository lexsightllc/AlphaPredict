"""Position sizing logic for AlphaPredict."""
from __future__ import annotations

import numpy as np

from .config import StrategyConfig


def position_sizer(predictions: np.ndarray, *, config: StrategyConfig,
                   previous_position: float | None = None) -> np.ndarray:
    """Translate raw model predictions into valid leverage exposures."""

    clipped = np.clip(predictions, config.leverage_min, config.leverage_max)
    if previous_position is None:
        return clipped

    turnover = np.abs(clipped - previous_position)
    penalty = np.exp(-config.turnover_penalty * turnover)
    adjusted = previous_position + (clipped - previous_position) * penalty
    return np.clip(adjusted, config.leverage_min, config.leverage_max)


__all__ = ["position_sizer"]
