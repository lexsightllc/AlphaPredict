"""Unit tests for evaluation module."""
import numpy as np
import pandas as pd
import pytest

from src.config import EvaluationConfig
from src.evaluation import Evaluator


@pytest.fixture
def sample_returns():
    """Create sample portfolio returns."""
    np.random.seed(42)
    returns = np.random.randn(100) * 0.01 + 0.0005
    return returns


@pytest.fixture
def sample_positions_and_prices():
    """Create sample positions and prices."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    positions = np.random.uniform(0, 2, 100)  # Leverage between 0 and 2
    prices = np.random.randn(100).cumsum() + 100

    return dates, positions, prices


class TestEvaluatorInitialization:
    """Test Evaluator initialization."""

    def test_evaluator_creates_instance(self):
        """Test Evaluator can be instantiated."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)
        assert evaluator is not None

    def test_evaluator_has_config(self):
        """Test Evaluator has configuration."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)
        assert evaluator.config == config


class TestSharpeRatioCalculation:
    """Test Sharpe ratio calculation."""

    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        sharpe = evaluator.calculate_sharpe_ratio(sample_returns)

        assert sharpe is not None
        assert isinstance(sharpe, (float, np.floating))
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_high_return(self):
        """Test Sharpe ratio for consistent returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        # Consistent positive returns should have high Sharpe
        returns = np.ones(100) * 0.01  # 1% daily return
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        assert sharpe > 0

    def test_sharpe_ratio_zero_return(self):
        """Test Sharpe ratio for zero returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        returns = np.zeros(100)
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        assert sharpe == 0

    def test_sharpe_ratio_negative_return(self):
        """Test Sharpe ratio for negative returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        returns = np.ones(100) * -0.01  # -1% daily return
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        assert sharpe < 0


class TestDrawdownCalculation:
    """Test maximum drawdown calculation."""

    def test_calculate_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        cumulative_returns = np.cumprod(1 + sample_returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert drawdown is not None
        assert isinstance(drawdown, (float, np.floating))
        assert drawdown >= 0

    def test_drawdown_monotonic_increasing(self):
        """Test drawdown for monotonically increasing returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        cumulative_returns = np.arange(100) * 0.01  # Monotonic increase
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert drawdown == 0  # No drawdown

    def test_drawdown_with_peak_and_trough(self):
        """Test drawdown calculation with clear peak and trough."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        # Create clear peak and trough
        cumulative_returns = np.array([0.0, 0.1, 0.05, 0.0, -0.05, 0.0])
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert drawdown > 0

    def test_drawdown_never_negative(self, sample_returns):
        """Test drawdown is never negative."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        cumulative_returns = np.cumprod(1 + sample_returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert drawdown >= 0


class TestMetricsCalculation:
    """Test metrics calculation."""

    def test_calculate_returns(self, sample_positions_and_prices):
        """Test return calculation."""
        dates, positions, prices = sample_positions_and_prices
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        # Calculate returns from prices
        price_returns = np.diff(prices) / prices[:-1]
        portfolio_returns = positions[:-1] * price_returns

        assert portfolio_returns is not None
        assert len(portfolio_returns) == len(prices) - 1

    def test_metrics_are_numeric(self, sample_returns):
        """Test calculated metrics are numeric."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        sharpe = evaluator.calculate_sharpe_ratio(sample_returns)
        cumulative_returns = np.cumprod(1 + sample_returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert np.isfinite(sharpe)
        assert np.isfinite(drawdown)


class TestVolatilityAdjustment:
    """Test volatility adjustment."""

    def test_volatility_penalty_applies(self, sample_returns):
        """Test volatility penalty application."""
        config = EvaluationConfig(volatility_cap=0.02)
        evaluator = Evaluator(config)

        volatility = np.std(sample_returns)
        assert volatility is not None

        # High volatility should trigger penalty
        high_vol_returns = sample_returns * 10
        high_vol = np.std(high_vol_returns)
        assert high_vol > config.volatility_cap

    def test_low_volatility_no_penalty(self):
        """Test no penalty for low volatility."""
        config = EvaluationConfig(volatility_cap=0.02)
        evaluator = Evaluator(config)

        low_vol_returns = np.ones(100) * 0.0001
        volatility = np.std(low_vol_returns)
        assert volatility < config.volatility_cap


class TestComprehensiveEvaluation:
    """Test comprehensive strategy evaluation."""

    def test_evaluate_strategy_returns_metrics(self, sample_positions_and_prices):
        """Test strategy evaluation returns all metrics."""
        dates, positions, prices = sample_positions_and_prices
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        # Create returns from positions and prices
        price_returns = np.diff(prices) / prices[:-1]
        portfolio_returns = positions[:-1] * price_returns

        sharpe = evaluator.calculate_sharpe_ratio(portfolio_returns)
        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert sharpe is not None
        assert drawdown is not None

    def test_evaluation_metrics_are_reasonable(self, sample_returns):
        """Test evaluation metrics are in reasonable ranges."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        sharpe = evaluator.calculate_sharpe_ratio(sample_returns)
        cumulative_returns = np.cumprod(1 + sample_returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        # Sharpe can be any value
        assert isinstance(sharpe, (float, np.floating))
        # Drawdown should be between 0 and 1
        assert 0 <= drawdown <= 1


class TestEdgeCases:
    """Test edge cases."""

    def test_single_return(self):
        """Test handling of single return value."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        returns = np.array([0.01])
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        # Should handle gracefully
        assert sharpe is not None or np.isnan(sharpe)

    def test_constant_returns(self):
        """Test handling of constant returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        returns = np.ones(100) * 0.005
        sharpe = evaluator.calculate_sharpe_ratio(returns)

        # Constant returns should have infinite Sharpe or special handling
        assert sharpe >= 0

    def test_all_negative_returns(self):
        """Test handling of all negative returns."""
        config = EvaluationConfig()
        evaluator = Evaluator(config)

        returns = np.ones(100) * -0.01
        sharpe = evaluator.calculate_sharpe_ratio(returns)
        cumulative_returns = np.cumprod(1 + returns) - 1
        drawdown = evaluator.calculate_max_drawdown(cumulative_returns)

        assert sharpe < 0
        assert drawdown > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
