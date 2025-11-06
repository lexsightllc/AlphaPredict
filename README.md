# AlphaPredict: S&P 500 Tactical Forecasting System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end machine learning pipeline for predicting S&P 500 excess returns, designed for the Hull Tactical Market Prediction Kaggle competition. This project implements a robust system for tactical asset allocation using machine learning, with strict adherence to competition constraints including real-time inference requirements and prevention of data leakage.

## 🌟 Features

- **End-to-End ML Pipeline**: Complete system from data ingestion to real-time predictions
- **Advanced Feature Engineering**: Technical indicators, statistical features, and macroeconomic factors
- **Multiple Model Architectures**: Support for gradient boosted trees, neural networks, and ensemble methods
- **Robust Backtesting**: Comprehensive evaluation framework with walk-forward validation
- **Production-Ready API**: High-performance inference endpoint with strict latency requirements

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lexsightllc/AlphaPredict.git
   cd AlphaPredict
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-minimal.txt  # For basic functionality
   # or
   pip install -r requirements.txt  # For full feature set
   ```

## 🏗️ Project Structure

```
AlphaPredict/
├── config/               # Configuration files
│   └── config.yaml      # Main configuration
├── data/                 # Data storage
│   ├── raw/             # Raw data files
│   ├── processed/       # Processed datasets
│   └── features/        # Feature sets
├── notebooks/           # Jupyter notebooks for exploration
├── models/              # Trained model artifacts
├── reports/             # Analysis reports and visualizations
└── src/                 # Source code
    ├── data/           # Data loading and processing
    ├── features/       # Feature engineering
    ├── models/         # Model definitions
    └── api/            # Inference API
```

## 🛠️ Usage

### Data Preparation
```python
from src.data.make_dataset import load_and_preprocess_data

# Load and preprocess data
df = load_and_preprocess_data('data/raw/sp500_data.csv')
```

### Feature Engineering
```python
from src.features.build_features import create_feature_pipeline

# Create features
features = create_feature_pipeline(df)
```

### Model Training
```python
from src.models.train import train_model

# Train model
model = train_model(X_train, y_train)
```

### Run Analysis
```bash
python run_analysis.py
```

### Start API Server
```bash
uvicorn src.api.main:app --reload
```

## 📊 Example Analysis

Run the example analysis script to see the system in action:

```bash
python run_analysis.py
```

This will generate visualizations of:
- Price with moving averages and Bollinger Bands
- Relative Strength Index (RSI)
- Trading signals and performance metrics

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Mozilla Public License 2.0 (MPL-2.0) - see the [LICENSE](LICENSE) file for details.

### Key Points About MPL-2.0:
- You may use, copy, modify, and distribute the software under the terms of the MPL-2.0
- Modifications to MPL-2.0 licensed files must be made available under the same license
- You may combine this software with proprietary code in a larger work
- The license includes a patent grant from contributors

For the full license text, see [LICENSE](LICENSE).

### Copyright

Copyright © 2025 Augusto 'Guto' Ochoa Ughini. All rights reserved.

## 📧 Contact

For questions or feedback, please open an issue or contact the maintainers.

## 📚 Resources

- [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction)
- [pandas-ta Documentation](https://github.com/twopirllc/pandas-ta)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
