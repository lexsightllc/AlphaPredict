"""Custom evaluation metrics for AlphaPredict."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def competition_sharpe(predictions: np.ndarray, targets: np.ndarray, *,
                       volatility_penalty: float = 0.01) -> float:
    """Approximate competition Sharpe metric with volatility penalty."""

    returns = predictions * targets
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    if np.isclose(volatility, 0.0):
        return 0.0
    sharpe = mean_return / volatility
    return sharpe - volatility_penalty * volatility


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Compute drawdown from an equity curve."""

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return drawdown


def evaluate_strategy(predictions: np.ndarray, targets: np.ndarray,
                      *, config: StrategyConfig) -> dict[str, float]:
    pnl = predictions * targets
    cumulative = np.cumsum(pnl)
    drawdown = compute_drawdown(pd.Series(cumulative))
    return {
        "competition_sharpe": competition_sharpe(predictions, targets,
                                                 volatility_penalty=config.volatility_cap),
        "total_return": float(cumulative[-1]),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((pnl > 0).mean()),
    }


__all__ = ["competition_sharpe", "compute_drawdown", "evaluate_strategy"]
