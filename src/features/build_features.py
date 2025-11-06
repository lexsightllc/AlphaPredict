"""
Feature engineering module for the AlphaPredict project.
Contains functions to create technical indicators and other financial features.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pandas_ta as ta

def add_technical_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters for technical indicators
        
    Returns:
        DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Ensure required columns are present
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # RSI
    for window in kwargs.get('rsi_windows', [14, 21]):
        df[f'rsi_{window}'] = ta.rsi(df['close'], length=window)
    
    # MACD
    macd = ta.macd(
        df['close'],
        fast=kwargs.get('macd_fast', 12),
        slow=kwargs.get('macd_slow', 26),
        signal=kwargs.get('macd_signal', 9)
    )
    df = pd.concat([df, macd], axis=1)
    
    # Bollinger Bands
    for window in kwargs.get('bb_windows', [20, 50]):
        bb = ta.bbands(df['close'], length=window, std=2)
        df = pd.concat([df, bb.add_prefix(f'bb_{window}_')], axis=1)
    
    # ATR (Average True Range)
    for window in kwargs.get('atr_windows', [14, 21]):
        df[f'atr_{window}'] = ta.atr(
            high=df['high'], 
            low=df['low'], 
            close=df['close'], 
            length=window
        )
    
    # Volume indicators if volume data is available
    if 'volume' in df.columns:
        # Volume Moving Averages
        for window in kwargs.get('volume_windows', [5, 21]):
            df[f'volume_ma{window}'] = df['volume'].rolling(window=window).mean()
            
        # On-Balance Volume
        df['obv'] = ta.obv(df['close'], df['volume'])
    
    # Drop any rows with NaN values that were created by the indicators
    df = df.dropna()
    
    return df

def add_volatility_features(df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
    """
    Add volatility-related features to the dataframe.
    
    Args:
        df: Input DataFrame with price data
        windows: List of lookback windows for volatility calculation
        
    Returns:
        DataFrame with added volatility features
    """
    if windows is None:
        windows = [5, 21, 63]  # Default lookback periods
    
    df = df.copy()
    
    # Calculate returns if not already present
    if 'log_return' not in df.columns:
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility (standard deviation of returns)
    for window in windows:
        df[f'volatility_{window}d'] = df['log_return'].rolling(window).std() * np.sqrt(252)  # Annualized
    
    # Realized volatility (historical volatility)
    for window in windows:
        df[f'realized_vol_{window}d'] = df['log_return'].rolling(window).std()
    
    # Parkinson volatility (uses high/low prices)
    if all(col in df.columns for col in ['high', 'low']):
        for window in windows:
            log_hl = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{window}d'] = np.sqrt(
                (1 / (4 * window * np.log(2))) * 
                (log_hl ** 2).rolling(window).sum()
            )
    
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features to the dataframe.
    
    Args:
        df: Input DataFrame with a datetime index
        
    Returns:
        DataFrame with added time features
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    df = df.copy()
    
    # Basic time features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Cyclical encoding for periodic features
    def encode_cyclical(df, col, max_val):
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        return df
    
    df = encode_cyclical(df, 'day_of_week', 7)
    df = encode_cyclical(df, 'month', 12)
    
    # Market regime indicators (simple moving average crossover)
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['trend'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    return df

def add_lag_features(
    df: pd.DataFrame, 
    columns: List[str], 
    lags: List[int],
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods
        fill_method: Method to fill NaN values ('ffill' or 'bfill')
        
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Fill NaN values
    if fill_method == 'ffill':
        df = df.ffill()
    elif fill_method == 'bfill':
        df = df.bfill()
    
    return df

def create_feature_pipeline(
    df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw OHLCV DataFrame
        config: Configuration dictionary with feature engineering parameters
        
    Returns:
        Feature matrix with all engineered features
    """
    if config is None:
        config = {}
    
    # Ensure we have a copy to avoid modifying the original
    df = df.copy()
    
    # Add technical indicators if enabled
    if config.get('use_technical_indicators', True):
        df = add_technical_indicators(
            df,
            rsi_windows=config.get('rsi_windows', [14, 21]),
            macd_fast=config.get('macd_fast', 12),
            macd_slow=config.get('macd_slow', 26),
            macd_signal=config.get('macd_signal', 9),
            bb_windows=config.get('bb_windows', [20, 50]),
            atr_windows=config.get('atr_windows', [14, 21]),
            volume_windows=config.get('volume_windows', [5, 21])
        )
    
    # Add volatility features if enabled
    if config.get('use_volatility_features', True):
        df = add_volatility_features(
            df,
            windows=config.get('volatility_windows', [5, 21, 63])
        )
    
    # Add time features if enabled
    if config.get('use_time_features', True) and isinstance(df.index, pd.DatetimeIndex):
        df = add_time_features(df)
    
    # Add lag features if enabled
    if config.get('use_lag_features', True):
        price_cols = ['close', 'volume'] if 'volume' in df.columns else ['close']
        df = add_lag_features(
            df,
            columns=price_cols,
            lags=config.get('lag_periods', [1, 2, 3, 5, 10]),
            fill_method=config.get('fill_method', 'ffill')
        )
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    return df
