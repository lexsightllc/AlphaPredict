# AlphaPredict: S&P 500 Tactical Forecasting System

An end-to-end machine learning pipeline for predicting S&P 500 excess returns, designed for the Hull Tactical Market Prediction Kaggle competition.

## Project Structure

```
AlphaPredict/
├── config/               # Configuration files
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

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Usage

### Data Preparation
```python
from src.data.make_dataset import load_and_preprocess_data

df = load_and_preprocess_data('data/raw/sp500_data.csv')
```

### Model Training
```python
from src.models.train import train_model

model = train_model(X_train, y_train)
```

### API
```bash
uvicorn src.api.main:app --reload
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
