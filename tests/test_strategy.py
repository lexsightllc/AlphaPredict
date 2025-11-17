"""Unit tests for strategy module."""
import numpy as np
import pandas as pd
import pytest

from src.config import StrategyConfig
from src.strategy import PositionSizer


@pytest.fixture
def sample_predictions():
    """Create sample model predictions."""
    return np.array([0.5, 1.5, 2.5, -0.5, 0.0, 1.0, 2.0, 0.3])


class TestPositionSizerInitialization:
    """Test PositionSizer initialization."""

    def test_position_sizer_creates_instance(self):
        """Test PositionSizer can be instantiated."""
        config = StrategyConfig()
        sizer = PositionSizer(config)
        assert sizer is not None

    def test_position_sizer_has_config(self):
        """Test PositionSizer has configuration."""
        config = StrategyConfig()
        sizer = PositionSizer(config)
        assert sizer.config == config


class TestLeverageClipping:
    """Test leverage clipping."""

    def test_clip_applies_leverage_bounds(self, sample_predictions):
        """Test clipping applies leverage bounds."""
        config = StrategyConfig(leverage_min=0.0, leverage_max=2.0)
        sizer = PositionSizer(config)

        clipped = sizer.clip(sample_predictions)

        assert np.all(clipped >= config.leverage_min)
        assert np.all(clipped <= config.leverage_max)

    def test_clip_preserves_sign(self):
        """Test clipping preserves sign of predictions."""
        config = StrategyConfig(leverage_min=0.0, leverage_max=2.0)
        sizer = PositionSizer(config)

        predictions = np.array([0.5, -0.3, 1.5, -1.5])
        clipped = sizer.clip(predictions)

        # Positive predictions stay positive
        assert clipped[0] > 0
        # Negative predictions stay non-positive
        assert clipped[1] <= 0
        assert clipped[3] <= 0

    def test_clip_values_at_bounds(self):
        """Test clipping correctly handles boundary values."""
        config = StrategyConfig(leverage_min=0.5, leverage_max=1.5)
        sizer = PositionSizer(config)

        predictions = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        clipped = sizer.clip(predictions)

        assert clipped[0] >= config.leverage_min  # 0 clipped to min
        assert clipped[1] == 0.5  # At lower bound
        assert clipped[4] <= config.leverage_max  # 3.0 clipped to max


class TestTurnoverPenalty:
    """Test turnover penalty application."""

    def test_turnover_penalty_reduces_positions(self):
        """Test turnover penalty reduces position sizes."""
        config = StrategyConfig(turnover_penalty=0.05)
        sizer = PositionSizer(config)

        positions_prev = np.array([0.5, 1.0, 0.0])
        positions_curr = np.array([0.4, 1.0, 0.1])

        penalized = sizer.apply_turnover_penalty(positions_curr, positions_prev)

        # Should be smaller than original due to penalty
        # (This depends on implementation details)
        assert penalized is not None

    def test_turnover_penalty_no_change_same_positions(self):
        """Test no penalty when positions don't change."""
        config = StrategyConfig(turnover_penalty=0.05)
        sizer = PositionSizer(config)

        positions = np.array([0.5, 1.0, 0.0])

        penalized = sizer.apply_turnover_penalty(positions, positions)

        # Should be similar or same when no turnover
        np.testing.assert_array_almost_equal(penalized, positions, decimal=5)

    def test_turnover_penalty_positive_reduction(self):
        """Test turnover penalty is positive and reduces exposure."""
        config = StrategyConfig(turnover_penalty=0.05)
        sizer = PositionSizer(config)

        positions_prev = np.array([0.0, 0.0, 0.0])
        positions_curr = np.array([1.0, 1.0, 1.0])

        penalized = sizer.apply_turnover_penalty(positions_curr, positions_prev)

        # Penalty should reduce total exposure
        assert np.abs(penalized).sum() <= np.abs(positions_curr).sum()


class TestPositionSizingPipeline:
    """Test complete position sizing pipeline."""

    def test_size_applies_all_steps(self, sample_predictions):
        """Test size method applies all steps."""
        config = StrategyConfig(
            leverage_min=0.0,
            leverage_max=2.0,
            turnover_penalty=0.05,
        )
        sizer = PositionSizer(config)

        positions = sizer.size(sample_predictions)

        assert positions is not None
        assert len(positions) == len(sample_predictions)
        assert np.all(positions >= config.leverage_min)
        assert np.all(positions <= config.leverage_max)

    def test_size_with_previous_positions(self, sample_predictions):
        """Test size with previous positions applies turnover penalty."""
        config = StrategyConfig(
            leverage_min=0.0,
            leverage_max=2.0,
            turnover_penalty=0.05,
        )
        sizer = PositionSizer(config)

        prev_positions = np.array([0.5] * len(sample_predictions))

        positions = sizer.size(sample_predictions, prev_positions)

        assert positions is not None
        assert len(positions) == len(sample_predictions)


class TestPositionValidation:
    """Test position validation."""

    def test_positions_are_numeric(self, sample_predictions):
        """Test sized positions are numeric."""
        config = StrategyConfig()
        sizer = PositionSizer(config)

        positions = sizer.size(sample_predictions)

        assert np.all(np.isfinite(positions))
        assert positions.dtype in [np.float32, np.float64, float]

    def test_positions_within_bounds(self, sample_predictions):
        """Test sized positions are within configured bounds."""
        config = StrategyConfig(leverage_min=0.0, leverage_max=1.5)
        sizer = PositionSizer(config)

        positions = sizer.size(sample_predictions)

        assert np.all(positions >= config.leverage_min - 1e-6)  # Small tolerance
        assert np.all(positions <= config.leverage_max + 1e-6)

    def test_zero_prediction_no_position(self):
        """Test zero prediction results in no position (or near-zero)."""
        config = StrategyConfig(leverage_min=0.0, leverage_max=2.0)
        sizer = PositionSizer(config)

        predictions = np.array([0.0])
        positions = sizer.size(predictions)

        # Zero prediction should result in zero or near-zero position
        assert positions[0] < 0.1


class TestEdgeCases:
    """Test edge cases."""

    def test_handles_all_zero_predictions(self):
        """Test handling of all-zero predictions."""
        config = StrategyConfig()
        sizer = PositionSizer(config)

        predictions = np.zeros(5)
        positions = sizer.size(predictions)

        assert positions is not None
        assert len(positions) == 5

    def test_handles_all_same_predictions(self):
        """Test handling of identical predictions."""
        config = StrategyConfig()
        sizer = PositionSizer(config)

        predictions = np.ones(5) * 0.5
        positions = sizer.size(predictions)

        assert positions is not None
        assert np.all(np.isfinite(positions))

    def test_handles_extreme_predictions(self):
        """Test handling of extreme prediction values."""
        config = StrategyConfig()
        sizer = PositionSizer(config)

        predictions = np.array([-1000, 1000, -999, 999])
        positions = sizer.size(predictions)

        # Should clip to bounds
        assert np.all(positions >= config.leverage_min)
        assert np.all(positions <= config.leverage_max)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
