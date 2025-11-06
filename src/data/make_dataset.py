"""
Data loading and preprocessing module for the AlphaPredict project.
Handles data ingestion, cleaning, and preparation for model training.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw market data from a CSV file.
    
    Args:
        file_path: Path to the CSV file containing raw market data
        
    Returns:
        pd.DataFrame: Raw market data
    """
    return pd.read_csv(file_path, parse_dates=['date'], index_col='date')

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns for a price series."""
    return np.log(prices / prices.shift(periods))

def preprocess_data(
    df: pd.DataFrame, 
    target_col: str = 'close',
    lookahead_days: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess market data and create features and targets.
    
    Args:
        df: Raw market data
        target_col: Column name for the target variable
        lookahead_days: Number of days ahead to predict
        
    Returns:
        Tuple containing features (X) and target (y)
    """
    # Calculate target: future log returns
    prices = df[target_col]
    y = calculate_returns(prices, -lookahead_days).shift(-lookahead_days)
    
    # Basic feature engineering
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['log_return_1d'] = calculate_returns(prices, 1)
    features['log_return_5d'] = calculate_returns(prices, 5)
    features['log_return_21d'] = calculate_returns(prices, 21)
    
    # Volume features
    if 'volume' in df.columns:
        features['volume_ma5'] = df['volume'].rolling(5).mean()
        features['volume_ma21'] = df['volume'].rolling(21).mean()
    
    # Remove rows with missing values
    valid_idx = y.notna() & features.notna().all(axis=1)
    
    return features[valid_idx], y[valid_idx]

def train_test_split(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.2,
    time_based: bool = True
) -> tuple:
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data to use for testing
        time_based: If True, split based on time (last test_size% of data)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if time_based:
        n = len(X)
        split_idx = int(n * (1 - test_size))
        return (
            X.iloc[:split_idx], 
            X.iloc[split_idx:], 
            y.iloc[:split_idx], 
            y.iloc[split_idx:]
        )
    else:
        from sklearn.model_selection import train_test_split as tts
        return tts(X, y, test_size=test_size, shuffle=False)

def load_and_preprocess_data(
    file_path: str, 
    test_size: float = 0.2,
    lookahead_days: int = 1
) -> dict:
    """
    Complete data loading and preprocessing pipeline.
    
    Args:
        file_path: Path to the raw data file
        test_size: Proportion of data to use for testing
        lookahead_days: Number of days ahead to predict
        
    Returns:
        Dictionary containing train/test splits and metadata
    """
    # Load data
    df = load_raw_data(file_path)
    
    # Preprocess
    X, y = preprocess_data(df, lookahead_days=lookahead_days)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, time_based=True
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'lookahead_days': lookahead_days
    }
