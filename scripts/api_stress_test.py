"""Stress test the inference API for latency compliance."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import httpx
import pandas as pd

from src.config import load_config
from src.utils import get_logger

LOGGER = get_logger("scripts.api_stress_test")
CONFIG_PATH = Path("config/settings.yaml")


def load_payload(sample_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(sample_path)
    return df.to_dict(orient="list")


def main(host: str = "http://localhost:8000") -> None:
    config = load_config(CONFIG_PATH)
    processed_sample = config.paths.processed_data / "feature_matrix.parquet"
    if not processed_sample.exists():
        raise FileNotFoundError("Processed feature matrix not found. Run scripts/train.py first.")

    payload = load_payload(processed_sample)
    timeout = config.serving.inference_timeout_ms / 1000
    with httpx.Client(timeout=timeout) as client:
        start = time.perf_counter()
        response = client.post(f"{host}/predict", json=payload)
        latency_ms = (time.perf_counter() - start) * 1_000
        response.raise_for_status()
    LOGGER.info("Latency %.2f ms, response keys: %s", latency_ms, list(response.json().keys()))


if __name__ == "__main__":  # pragma: no cover - entry point
    main()
