import sys
import pandas as pd
import numpy as np
import yfinance as yf
import talib


def compute_features(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    # Initialize a DataFrame to hold the features
    features = pd.DataFrame(index=stock_data.index)

    # 1. Rolling Asset Statistical Features
    features = pd.concat([features, compute_rolling_asset_statistical_features(stock_data, window_size)], axis=1)

    # 2. Rolling Asset Technical Indicators
    # Here you would calculate technical indicators using window_size // 2

    # 3. Rolling Asset-to-Market Index Features
    # Here you would calculate features like Alpha, Beta, etc. using the window_size

    # 4. Rolling Asset Trend Features
    # Here you would calculate trend features using a fixed window of 2200 (or 1000 if clarified)

    return features

def compute_rolling_asset_statistical_features(stock_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling asset statistical features for given stock data.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data with columns 'Open', 'High', 'Low', 'Close', 'Volume'.
    window_size (int): Look-back window size (L) for calculating rolling features.

    Returns:
    pd.DataFrame: DataFrame with rolling asset statistical features.
    """

    # Calculate daily returns
    stock_data['Returns'] = stock_data['Close'].pct_change()

    # Calculate rolling volatility (standard deviation of returns)
    stock_data['Volatility'] = stock_data['Returns'].rolling(window=window_size).std()

    # Calculate maximum drawdown and drawup within the look-back window
    # Drawdown: Max peak to trough decline within the window
    # Drawup: Max trough to peak rise within the window
    rolling_max = stock_data['Close'].rolling(window=window_size, min_periods=1).max()
    rolling_min = stock_data['Close'].rolling(window=window_size, min_periods=1).min()
    stock_data['Max_Drawdown'] = stock_data['Close'] / rolling_max - 1.0
    stock_data['Max_Drawup'] = stock_data['Close'] / rolling_min - 1.0

    # Extract the 'Volume' feature as it is, assuming it's already in the right format
    # If any normalization or scaling is needed, it should be applied here
    stock_data['Volume_Change'] = stock_data['Volume'].diff()

    # Select only the relevant columns for output
    statistical_features = stock_data[['Returns', 'Volatility', 'Max_Drawdown', 'Max_Drawup', 'Volume_Change']]

    return statistical_features


def compute_rolling_asset_technical_indicators(stock_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling asset technical indicators for given stock data using TA-Lib.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data.
    window_size (int): Look-back window size for calculating technical indicators.

    Returns:
    pd.DataFrame: DataFrame with rolling asset technical indicators.
    """

    # Technical Indicators
    # Using TA-Lib to calculate each technical indicator.
    technical_indicators = pd.DataFrame(index=stock_data.index)

    # Average Directional Index (ADX)
    technical_indicators['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            timeperiod=window_size)

    # Absolute Price Oscillator (APO)
    technical_indicators['APO'] = talib.APO(stock_data['Close'], fastperiod=window_size // 2, slowperiod=window_size,
                                            matype=0)

    # Commodity Channel Index (CCI)
    technical_indicators['CCI'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            timeperiod=window_size)

    # Directional Movement Index (DX)
    technical_indicators['DX'] = talib.DX(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                          timeperiod=window_size)

    # Money Flow Index (MFI)
    technical_indicators['MFI'] = talib.MFI(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            stock_data['Volume'], timeperiod=window_size)

    # Relative Strength Index (RSI)
    technical_indicators['RSI'] = talib.RSI(stock_data['Close'], timeperiod=window_size)

    # Ultimate Oscillator (ULTOSC)
    technical_indicators['ULTOSC'] = talib.ULTOSC(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                                  timeperiod1=window_size // 2, timeperiod2=window_size,
                                                  timeperiod3=window_size * 2)

    # Williams' %R (WILLR)
    technical_indicators['WILLR'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                                timeperiod=window_size)

    # Normalized Average True Range (NATR)
    technical_indicators['NATR'] = talib.NATR(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                              timeperiod=window_size)

    return technical_indicators


if __name__ == '__main__':
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    stock_data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    market_data = yf.download('SPY', start_date, end_date, interval='1d')

    stock_data.index = stock_data.index.tz_localize(None)
    market_data.index = market_data.index.tz_localize(None)

    features = compute_features(stock_data, market_data, window_size=60)

    print(features.tail())
