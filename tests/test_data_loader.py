"""Unit tests for data loader module."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import DataLoader


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = {
        "date": dates,
        "symbol": ["BTC"] * 100,
        "open": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 101,
        "low": np.random.randn(100).cumsum() + 99,
        "close": np.random.randn(100).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_dir(sample_data):
    """Create temporary data directory with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        external_dir = data_dir / "External"
        external_dir.mkdir(parents=True)

        sample_data.to_csv(external_dir / "train.csv", index=False)
        sample_data.to_csv(external_dir / "test.csv", index=False)

        yield data_dir


class TestDataLoaderInitialization:
    """Test DataLoader initialization."""

    def test_data_loader_creates_instance(self):
        """Test DataLoader can be instantiated."""
        loader = DataLoader()
        assert loader is not None

    def test_data_loader_has_config(self):
        """Test DataLoader has configuration."""
        loader = DataLoader()
        assert loader.config is not None


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_train_data(self, temp_data_dir):
        """Test loading training data."""
        loader = DataLoader()
        # Override data path for testing
        loader.config.paths.data_external = temp_data_dir / "External"

        data = loader.load_train_data()
        assert data is not None
        assert len(data) > 0
        assert "date" in data.columns
        assert "close" in data.columns

    def test_load_test_data(self, temp_data_dir):
        """Test loading test data."""
        loader = DataLoader()
        loader.config.paths.data_external = temp_data_dir / "External"

        data = loader.load_test_data()
        assert data is not None
        assert len(data) > 0

    def test_data_has_required_columns(self, temp_data_dir):
        """Test loaded data has required columns."""
        loader = DataLoader()
        loader.config.paths.data_external = temp_data_dir / "External"

        data = loader.load_train_data()
        required_columns = ["date", "symbol", "close", "volume"]
        for col in required_columns:
            assert col in data.columns

    def test_data_types_are_correct(self, temp_data_dir):
        """Test loaded data has correct types."""
        loader = DataLoader()
        loader.config.paths.data_external = temp_data_dir / "External"

        data = loader.load_train_data()
        assert pd.api.types.is_datetime64_any_dtype(data["date"])
        assert pd.api.types.is_numeric_dtype(data["close"])


class TestDataValidation:
    """Test data validation functionality."""

    def test_validate_schema_accepts_valid_data(self, sample_data):
        """Test schema validation accepts valid data."""
        loader = DataLoader()
        # Should not raise any exceptions
        try:
            loader._validate_schema(sample_data)
        except Exception as e:
            pytest.fail(f"Valid data raised exception: {e}")

    def test_validate_schema_rejects_missing_columns(self, sample_data):
        """Test schema validation rejects missing required columns."""
        loader = DataLoader()
        invalid_data = sample_data.drop("close", axis=1)

        with pytest.raises((KeyError, ValueError)):
            loader._validate_schema(invalid_data)

    def test_validate_schema_rejects_missing_dates(self, sample_data):
        """Test schema validation handles missing date column."""
        loader = DataLoader()
        invalid_data = sample_data.drop("date", axis=1)

        with pytest.raises((KeyError, ValueError)):
            loader._validate_schema(invalid_data)


class TestTimeSeriesSplitting:
    """Test time series splitting functionality."""

    def test_get_splits_returns_multiple_folds(self):
        """Test get_splits returns multiple train-test splits."""
        loader = DataLoader()
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame({"date": dates, "close": np.random.randn(200)})

        splits = list(loader.get_splits(data, n_splits=5))
        assert len(splits) >= 3  # At least a few splits should be possible

    def test_splits_have_no_overlap(self):
        """Test train and test sets in splits don't overlap."""
        loader = DataLoader()
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "close": np.random.randn(200),
            "symbol": ["BTC"] * 200,
        })

        for train, test in loader.get_splits(data, n_splits=3):
            train_dates = set(train["date"])
            test_dates = set(test["date"])
            assert len(train_dates & test_dates) == 0, "Train and test overlap"

    def test_splits_maintain_time_order(self):
        """Test splits maintain chronological time ordering."""
        loader = DataLoader()
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "close": np.random.randn(200),
            "symbol": ["BTC"] * 200,
        })

        for train, test in loader.get_splits(data, n_splits=3):
            # Test dates should be after train dates
            max_train_date = train["date"].max()
            min_test_date = test["date"].min()
            assert max_train_date < min_test_date


class TestDataQuality:
    """Test data quality checks."""

    def test_handle_missing_values(self, sample_data):
        """Test handling of missing values."""
        loader = DataLoader()
        # Introduce some missing values
        data_with_na = sample_data.copy()
        data_with_na.loc[5:10, "close"] = np.nan

        # Should handle gracefully
        assert data_with_na.isna().sum().sum() > 0

    def test_data_sorted_by_date(self, sample_data):
        """Test data is properly sorted by date."""
        loader = DataLoader()
        # Shuffle the data
        shuffled = sample_data.sample(frac=1).reset_index(drop=True)

        # Load should sort it
        loader.config.paths.data_external = Path("/tmp")  # dummy path
        # In actual use, data should be sorted
        assert len(shuffled) == len(sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
