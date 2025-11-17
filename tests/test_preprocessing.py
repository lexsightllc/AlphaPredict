"""Unit tests for preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessingConfig
from src.preprocessing import Preprocessor


@pytest.fixture
def sample_data():
    """Create sample data for preprocessing tests."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "symbol": ["BTC"] * 100,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
    }
    return pd.DataFrame(data)


class TestPreprocessorInitialization:
    """Test Preprocessor initialization."""

    def test_preprocessor_creates_instance(self):
        """Test Preprocessor can be instantiated."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)
        assert preprocessor is not None

    def test_preprocessor_has_config(self):
        """Test Preprocessor has configuration."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)
        assert preprocessor.config == config


class TestWinsorization:
    """Test winsorization functionality."""

    def test_winsorization_removes_extremes(self, sample_data):
        """Test winsorization removes extreme values."""
        config = PreprocessingConfig(winsorize_limits=(0.01, 0.99))
        preprocessor = Preprocessor(config)

        data = sample_data.copy()
        # Create extreme outliers
        data.loc[0, "close"] = 10000
        data.loc[1, "close"] = -10000

        processed = preprocessor._winsorize(data)
        # Extreme values should be reduced
        assert processed["close"].max() < 10000
        assert processed["close"].min() > -10000

    def test_winsorization_preserves_data_shape(self, sample_data):
        """Test winsorization preserves data shape."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        original_shape = sample_data.shape
        processed = preprocessor._winsorize(sample_data)
        assert processed.shape == original_shape

    def test_winsorization_is_idempotent(self, sample_data):
        """Test applying winsorization twice gives same result."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        once = preprocessor._winsorize(sample_data)
        twice = preprocessor._winsorize(once)
        pd.testing.assert_frame_equal(once, twice)


class TestLagCreation:
    """Test lag feature creation."""

    def test_create_lags_adds_columns(self, sample_data):
        """Test lag creation adds new columns."""
        config = PreprocessingConfig(lags=[1, 5])
        preprocessor = Preprocessor(config)

        original_cols = len(sample_data.columns)
        processed = preprocessor._create_lags(sample_data)
        assert len(processed.columns) > original_cols

    def test_lag_values_are_correct(self, sample_data):
        """Test lag values are correctly calculated."""
        config = PreprocessingConfig(lags=[1, 2])
        preprocessor = Preprocessor(config)

        processed = preprocessor._create_lags(sample_data)
        # Lag 1 should be previous value
        assert processed["close_lag_1"].iloc[1] == sample_data["close"].iloc[0]

    def test_first_lag_rows_have_nans(self, sample_data):
        """Test first rows are NaN due to lag."""
        config = PreprocessingConfig(lags=[5])
        preprocessor = Preprocessor(config)

        processed = preprocessor._create_lags(sample_data)
        # First 5 rows should have NaNs in lag features
        assert processed["close_lag_5"].isna().sum() >= 5

    def test_lag_names_follow_convention(self, sample_data):
        """Test lag column names follow naming convention."""
        config = PreprocessingConfig(lags=[1, 5, 21])
        preprocessor = Preprocessor(config)

        processed = preprocessor._create_lags(sample_data)
        expected_names = ["close_lag_1", "close_lag_5", "close_lag_21"]
        for name in expected_names:
            assert any(name in col for col in processed.columns)


class TestRollingStatistics:
    """Test rolling statistics functionality."""

    def test_rolling_stats_adds_columns(self, sample_data):
        """Test rolling stats add new columns."""
        config = PreprocessingConfig(rolling_windows=[5])
        preprocessor = Preprocessor(config)

        original_cols = len(sample_data.columns)
        processed = preprocessor._add_rolling_stats(sample_data)
        assert len(processed.columns) > original_cols

    def test_rolling_mean_is_correct(self, sample_data):
        """Test rolling mean is correctly calculated."""
        config = PreprocessingConfig(rolling_windows=[5])
        preprocessor = Preprocessor(config)

        processed = preprocessor._add_rolling_stats(sample_data)
        # Rolling mean at index 5 should equal mean of first 5 values
        expected = sample_data["close"].iloc[:5].mean()
        actual = processed.loc[4, "close_rolling_mean_5"]
        assert np.isclose(expected, actual, rtol=1e-5)

    def test_rolling_stats_create_std_dev(self, sample_data):
        """Test rolling stats include standard deviation."""
        config = PreprocessingConfig(rolling_windows=[5])
        preprocessor = Preprocessor(config)

        processed = preprocessor._add_rolling_stats(sample_data)
        assert "close_rolling_std_5" in processed.columns

    def test_rolling_windows_with_gaps(self, sample_data):
        """Test rolling statistics handle multiple window sizes."""
        config = PreprocessingConfig(rolling_windows=[5, 21, 63])
        preprocessor = Preprocessor(config)

        processed = preprocessor._add_rolling_stats(sample_data)
        assert "close_rolling_mean_5" in processed.columns
        assert "close_rolling_mean_21" in processed.columns
        assert "close_rolling_mean_63" in processed.columns


class TestImputation:
    """Test data imputation."""

    def test_imputation_handles_nans(self):
        """Test imputation handles NaN values."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        data = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "close": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        })

        processed = preprocessor._impute_missing(data)
        assert processed.isna().sum().sum() == 0

    def test_imputation_preserves_known_values(self):
        """Test imputation preserves known values."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        data = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=10),
            "close": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
        })

        processed = preprocessor._impute_missing(data)
        assert processed["close"].iloc[0] == 1
        assert processed["close"].iloc[3] == 4


class TestScaling:
    """Test feature scaling."""

    def test_scaler_is_created_on_fit(self, sample_data):
        """Test scaler is created during fit."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        preprocessor.fit(sample_data)
        assert preprocessor.scaler is not None

    def test_transform_scales_features(self, sample_data):
        """Test transform scales features."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        preprocessor.fit(sample_data)
        scaled = preprocessor.transform(sample_data)

        # Numeric columns should be scaled
        numeric_cols = scaled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["symbol"]:  # Skip non-numeric
                # Most values should be in [-3, 3] after standardization
                assert scaled[col].abs().max() <= 10  # Some tolerance for outliers

    def test_fit_transform_consistency(self, sample_data):
        """Test fit_transform is consistent with fit then transform."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        result1 = preprocessor.fit_transform(sample_data)

        preprocessor2 = Preprocessor(config)
        preprocessor2.fit(sample_data)
        result2 = preprocessor2.transform(sample_data)

        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)


class TestFullPipeline:
    """Test full preprocessing pipeline."""

    def test_full_fit_transform_pipeline(self, sample_data):
        """Test full preprocessing pipeline."""
        config = PreprocessingConfig(
            lags=[1, 5],
            rolling_windows=[5, 21],
            winsorize_limits=(0.01, 0.99),
        )
        preprocessor = Preprocessor(config)

        result = preprocessor.fit_transform(sample_data)

        # Check output properties
        assert result is not None
        assert len(result) > 0
        assert len(result) <= len(sample_data)  # May drop first rows
        assert result.isna().sum().sum() == 0  # No NaNs after processing

    def test_pipeline_creates_features(self, sample_data):
        """Test pipeline creates expected features."""
        config = PreprocessingConfig(
            lags=[1, 5],
            rolling_windows=[5],
        )
        preprocessor = Preprocessor(config)

        result = preprocessor.fit_transform(sample_data)

        # Should have lags, rolling stats
        assert any("lag" in col.lower() for col in result.columns)
        assert any("rolling" in col.lower() for col in result.columns)

    def test_pipeline_maintains_date_column(self, sample_data):
        """Test pipeline maintains date column for indexing."""
        config = PreprocessingConfig()
        preprocessor = Preprocessor(config)

        result = preprocessor.fit_transform(sample_data)
        assert "date" in result.columns or "symbol" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
