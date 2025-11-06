# AlphaPredict: Hull Tactical Market Prediction Pipeline

An opinionated end-to-end machine learning stack built to compete in Kaggle's [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) challenge. The system targets daily excess returns of the S&P 500 index using a curated blend of financial and macroeconomic indicators while respecting the competition's strict real-time inference and data leakage requirements.

This repository serves as both the production code base and the accompanying research narrative. It is intentionally organized to balance experiment agility with the reproducibility demands of a regulated trading workflow.

## 📚 Documentation Highlights

The primary documentation lives in [`README.md`](README.md) and the rendered research manuscript [`reports/final_report.md`](reports/final_report.md). The README motivates the tactical allocation problem, revisits the Efficient Market Hypothesis, and discusses how disciplined feature engineering and validation can expose exploitable market structure under realistic volatility constraints.

## 🗂️ Project Layout

```
project_root/
├── README.md
├── data/
│   ├── External/
│   │   ├── train.csv
│   │   └── test.csv
│   └── Processed/
│       ├── cleaned_train.parquet
│       └── feature_matrix.parquet
├── notebooks/
│   ├── eda.ipynb
│   ├── model_development.ipynb
│   └── api_submission_template.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── strategy.py
│   ├── evaluation.py
│   ├── serving.py
│   └── utils.py
├── reports/
│   ├── figures/
│   └── final_report.md
├── scripts/
│   ├── train.py
│   ├── backtest.py
│   └── api_stress_test.py
├── artifacts/
│   ├── models/final_model.pkl
│   ├── scalers/trained_scaler.pkl
│   └── metadata/
│       ├── feature_list.json
│       └── training_statistics.json
└── config/settings.yaml
```

> **Note:** Raw datasets and trained artifacts are excluded from version control. Populate the placeholders above by downloading Kaggle competition data or restoring artifacts from your secure storage.

## 🔧 Core Modules

- **`src/config.py`** – Centralized configuration definitions. Exposes strongly typed data classes for file system paths, preprocessing parameters, validation schedules, and model hyperparameters. Utility helpers load YAML configurations from [`config/settings.yaml`](config/settings.yaml).
- **`src/data_loader.py`** – Data access layer that loads training and inference datasets, enforces schemas, and constructs time-aware cross-validation folds.
- **`src/preprocessing.py`** – Full preprocessing pipeline: missing value handling, winsorization, lag creation, rolling statistics, alignment to the competition's latency window, and feature scaling.
- **`src/models.py`** – Model abstractions unifying gradient boosted trees, regularized linear models, deep tabular networks, and ensemble stacks under a shared `fit`/`predict` interface.
- **`src/strategy.py`** – Maps model predictions to valid position sizes in `[0, 2]` while honoring leverage caps, turnover throttles, and execution realism.
- **`src/evaluation.py`** – Implements the competition-specific Sharpe metric with volatility penalties plus custom diagnostics such as drawdown, hit rate, and tail-risk exposure.
- **`src/serving.py`** – Production inference surface compatible with the competition's real-time API. Ensures no forward-looking leakage, handles request batching, and streams predictions within latency constraints.
- **`src/utils.py`** – Shared utilities including seed management, structured logging, configuration validation, and instrumentation helpers.

## 🧪 Workflow Overview

1. **Data Preparation** – Use `scripts/train.py` to load the external datasets, build lagged features with [`src/preprocessing.py`](src/preprocessing.py), and persist processed matrices to `data/Processed/`.
2. **Model Development** – Experiment interactively inside [`notebooks/model_development.ipynb`](notebooks/model_development.ipynb). The notebook leverages the production preprocessing utilities to ensure parity between experiments and deployment.
3. **Training & Backtesting** – Run `scripts/train.py` for production training. Validate robustness using `scripts/backtest.py`, which simulates execution frictions and drawdowns on out-of-sample periods.
4. **Serving** – Package the trained model plus feature metadata into `artifacts/`. Use `scripts/api_stress_test.py` to assert that the [`src/serving.py`](src/serving.py) predict function satisfies latency limits under realistic loads.

## 🚀 Getting Started

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train and evaluate
python scripts/train.py
python scripts/backtest.py

# Launch local inference server
uvicorn src.serving:app --host 0.0.0.0 --port 8000
```

## 📈 Interpretability & Reporting

The pipeline logs experiment metadata and stores model explainability assets (feature importances, SHAP summaries, rolling Sharpe charts) under `reports/figures/`. The rendered [`reports/final_report.md`](reports/final_report.md) consolidates these insights into a narrative suitable for investment committees or compliance review.

## 🤝 Contributing

1. Fork the repository and create a feature branch.
2. Run the formatting and linting checks defined in `pyproject.toml`.
3. Submit a pull request describing the motivation, methodology, and validation for your contribution.

## 📄 License

This project is released under the [Mozilla Public License 2.0](LICENSE). Please review the license before distributing derivative works.

## 📬 Contact

Questions, bug reports, or collaboration requests are welcome via GitHub Issues.
