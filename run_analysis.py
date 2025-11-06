"""
Minimal analysis script for AlphaPredict
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_sample_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Generate random walk for prices
    returns = np.random.normal(0.0005, 0.01, n)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': np.random.lognormal(10, 1, n)
    })
    df.set_index('date', inplace=True)
    return df

def calculate_technical_indicators(df):
    """Calculate simple technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def plot_results(df):
    """Plot the results"""
    plt.figure(figsize=(14, 10))
    
    # Plot price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Close Price', alpha=0.7)
    plt.plot(df.index, df['SMA_20'], label='20-day SMA', alpha=0.7)
    plt.plot(df.index, df['SMA_50'], label='50-day SMA', alpha=0.7)
    plt.plot(df.index, df['BB_upper'], 'r--', alpha=0.5, label='Bollinger Bands')
    plt.plot(df.index, df['BB_lower'], 'r--', alpha=0.5)
    plt.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='red', alpha=0.1)
    plt.title('Price and Moving Averages')
    plt.legend()
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['RSI'], 'g-', alpha=0.7)
    plt.axhline(70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(30, color='r', linestyle='--', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

def main():
    print("Generating sample data...")
    df = generate_sample_data()
    
    print("Calculating technical indicators...")
    df = calculate_technical_indicators(df)
    
    print("Plotting results...")
    plot_results(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
