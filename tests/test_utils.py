"""Unit tests for utilities module."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.utils import Timer, load_json, load_yaml, save_json, save_yaml, set_seed


class TestSeedManagement:
    """Test seed management for reproducibility."""

    def test_set_seed_initializes_numpy(self):
        """Test set_seed initializes numpy."""
        set_seed(42)
        val1 = np.random.randn()

        set_seed(42)
        val2 = np.random.randn()

        assert val1 == val2

    def test_seed_reproducibility(self):
        """Test reproducibility with same seed."""
        set_seed(42)
        array1 = np.random.randn(10)

        set_seed(42)
        array2 = np.random.randn(10)

        np.testing.assert_array_equal(array1, array2)

    def test_different_seeds_different_results(self):
        """Test different seeds produce different results."""
        set_seed(42)
        array1 = np.random.randn(10)

        set_seed(43)
        array2 = np.random.randn(10)

        # Should be different (with very high probability)
        assert not np.allclose(array1, array2)


class TestYamlOperations:
    """Test YAML file operations."""

    def test_save_yaml_creates_file(self):
        """Test save_yaml creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.yaml"
            data = {"key": "value", "number": 42}

            save_yaml(data, filepath)

            assert filepath.exists()

    def test_load_yaml_reads_file(self):
        """Test load_yaml reads saved file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.yaml"
            original_data = {"key": "value", "number": 42}

            save_yaml(original_data, filepath)
            loaded_data = load_yaml(filepath)

            assert loaded_data == original_data

    def test_yaml_preserves_types(self):
        """Test YAML operations preserve data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.yaml"
            data = {
                "string": "test",
                "integer": 42,
                "float": 3.14,
                "list": [1, 2, 3],
                "nested": {"key": "value"},
            }

            save_yaml(data, filepath)
            loaded = load_yaml(filepath)

            assert loaded["integer"] == 42
            assert loaded["float"] == 3.14
            assert loaded["list"] == [1, 2, 3]
            assert loaded["nested"]["key"] == "value"

    def test_yaml_handles_missing_file(self):
        """Test load_yaml handles missing file gracefully."""
        filepath = Path("/nonexistent/path/file.yaml")

        try:
            load_yaml(filepath)
            pytest.fail("Should raise error for missing file")
        except (FileNotFoundError, OSError):
            pass  # Expected


class TestJsonOperations:
    """Test JSON file operations."""

    def test_save_json_creates_file(self):
        """Test save_json creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            save_json(data, filepath)

            assert filepath.exists()

    def test_load_json_reads_file(self):
        """Test load_json reads saved file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            original_data = {"key": "value", "number": 42}

            save_json(original_data, filepath)
            loaded_data = load_json(filepath)

            assert loaded_data == original_data

    def test_json_preserves_types(self):
        """Test JSON operations preserve data types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            data = {
                "string": "test",
                "integer": 42,
                "float": 3.14,
                "list": [1, 2, 3],
                "nested": {"key": "value"},
            }

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded["integer"] == 42
            assert loaded["float"] == 3.14
            assert loaded["list"] == [1, 2, 3]
            assert loaded["nested"]["key"] == "value"

    def test_json_handles_missing_file(self):
        """Test load_json handles missing file gracefully."""
        filepath = Path("/nonexistent/path/file.json")

        try:
            load_json(filepath)
            pytest.fail("Should raise error for missing file")
        except (FileNotFoundError, OSError):
            pass  # Expected


class TestTimerContext:
    """Test Timer context manager."""

    def test_timer_measures_time(self):
        """Test Timer measures elapsed time."""
        import time

        with Timer("test"):
            time.sleep(0.01)

        # Timer should complete without error

    def test_timer_returns_elapsed_time(self):
        """Test Timer returns elapsed time."""
        import time

        timer = Timer("test")
        with timer:
            time.sleep(0.01)

        assert timer.elapsed_time > 0

    def test_timer_measures_multiple_blocks(self):
        """Test Timer can measure multiple blocks."""
        import time

        times = []
        for i in range(3):
            timer = Timer(f"block_{i}")
            with timer:
                time.sleep(0.01)
            times.append(timer.elapsed_time)

        assert all(t > 0 for t in times)


class TestYamlJsonInterop:
    """Test YAML and JSON interoperability."""

    def test_yaml_and_json_equivalent(self):
        """Test YAML and JSON store equivalent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "data.yaml"
            json_path = Path(tmpdir) / "data.json"

            data = {"key": "value", "number": 42, "list": [1, 2, 3]}

            save_yaml(data, yaml_path)
            save_json(data, json_path)

            yaml_loaded = load_yaml(yaml_path)
            json_loaded = load_json(json_path)

            assert yaml_loaded == json_loaded

    def test_yaml_from_json_file(self):
        """Test loading JSON file with YAML loader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "data.json"
            data = {"key": "value"}

            save_json(data, json_path)

            # YAML can read JSON
            yaml_loaded = load_yaml(json_path)
            assert yaml_loaded == data


class TestFileOperationErrors:
    """Test error handling in file operations."""

    def test_save_to_invalid_path(self):
        """Test save handles invalid path."""
        invalid_path = Path("/root/invalid/path/file.yaml")

        try:
            save_yaml({"test": "data"}, invalid_path)
            # May or may not raise depending on permissions
        except (PermissionError, OSError):
            pass  # Expected in some environments

    def test_corrupt_yaml_file(self):
        """Test load handles corrupt YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "corrupt.yaml"
            filepath.write_text("{ invalid yaml ][")

            try:
                load_yaml(filepath)
                pytest.fail("Should raise error for invalid YAML")
            except (yaml.YAMLError, Exception):
                pass  # Expected

    def test_corrupt_json_file(self):
        """Test load handles corrupt JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "corrupt.json"
            filepath.write_text("{ invalid json ][")

            try:
                load_json(filepath)
                pytest.fail("Should raise error for invalid JSON")
            except (json.JSONDecodeError, Exception):
                pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
