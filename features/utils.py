import sys
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import statsmodels.api as sm


def compute_features(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    # Initialize a DataFrame to hold the features
    features = pd.DataFrame(index=stock_data.index)

    # 1. Rolling Asset Statistical Features
    asf = compute_rolling_asset_statistical_features(stock_data, window_size)

    # 2. Rolling Asset Technical Indicators
    ati = compute_rolling_asset_technical_indicators(stock_data, window_size/2)

    # 3. Rolling Asset-to-Market Index Features
    atm = compute_rolling_asset_to_market_index_features(stock_data, market_data, window_size)

    # 4. Time Features
    tf = compute_time_features(stock_data)

    # 5. Rolling Asset Trend Features
    # Here you would calculate trend features using a fixed window of 2200
    atf = compute_rolling_asset_trend_features(stock_data, market_data, window_size=2200)

    features = pd.concat([features, asf, ati, atm, atf, tf], axis=1)

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


def compute_rolling_asset_to_market_index_features(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling asset-to-market index features for given stock data and market data.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data with 'Close' prices.
    market_data (pd.DataFrame): DataFrame containing market index data with 'Close' prices.
    window_size (int): Look-back window size (L) for calculating features.

    Returns:
    pd.DataFrame: DataFrame with rolling asset-to-market index features.
    """

    # Calculate returns for both stock and market index
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()

    # Initialize DataFrame to hold the features
    asset_to_market_features = pd.DataFrame(index=stock_data.index)

    # Asset-To-Index Alpha and Beta (Capital Asset Pricing Model parameters)
    # Rolling window regression to calculate alpha and beta
    for i in range(window_size, len(stock_data)):
        # Get the windowed data
        stock_window = stock_returns[i-window_size+1:i+1]
        market_window = market_returns[i-window_size+1:i+1]

        if len(stock_window) < window_size:  # Skip if there isn't enough data
            continue

        # Add a constant for OLS regression
        market_window = sm.add_constant(market_window)

        # Perform the regression
        model = sm.OLS(stock_window, market_window).fit()

        # Store the alpha and beta
        asset_to_market_features.at[stock_data.index[i], 'Alpha'] = model.params['const']
        asset_to_market_features.at[stock_data.index[i], 'Beta'] = model.params['Close']

    # Index Returns
    asset_to_market_features['Index_Returns'] = market_returns

    # Index Volatility: Standard deviation of market returns
    asset_to_market_features['Index_Volatility'] = market_returns.rolling(window=window_size).std()

    # Pearson Correlation Coefficient between stock and market returns
    asset_to_market_features['Correlation'] = stock_returns.rolling(window=window_size).corr(market_returns)

    # Return Distribution Dependencies: Rolling Covariance and Variance
    asset_to_market_features['Covariance'] = stock_returns.rolling(window=window_size).cov(market_returns)
    market_variance = market_returns.rolling(window=window_size).var()

    # Adjusted Close for both stock and market to compute Beta without using rolling regression
    adjusted_stock_close = stock_data['Close'] / stock_data['Close'].iloc[0]
    adjusted_market_close = market_data['Close'] / market_data['Close'].iloc[0]
    asset_to_market_features['Beta_Direct'] = adjusted_stock_close.rolling(window=window_size).cov(adjusted_market_close) / market_variance

    return asset_to_market_features


def compute_time_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time features for given stock data.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data with a datetime index.

    Returns:
    pd.DataFrame: DataFrame with time features.
    """

    # Ensure the index is a datetime index
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)

    # Initialize DataFrame to hold the features
    time_features = pd.DataFrame(index=stock_data.index)

    # Extract time features
    time_features['sin_Day_of_Week'] = np.sin(2 * np.pi * stock_data.index.dayofweek / 7)
    time_features['cos_Day_of_Week'] = np.cos(2 * np.pi * stock_data.index.dayofweek / 7)
    time_features['sin_Day_of_Month'] = np.sin(2 * np.pi * stock_data.index.day / 31)
    time_features['cos_Day_of_Month'] = np.cos(2 * np.pi * stock_data.index.day / 31)
    time_features['sin_Day_of_Year'] = np.sin(2 * np.pi * stock_data.index.dayofyear / 365)
    time_features['cos_Day_of_Year'] = np.cos(2 * np.pi * stock_data.index.dayofyear / 365)

    return time_features


def compute_rolling_asset_trend_features(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int=2200) -> pd.DataFrame:
    """
    Calculate rolling asset trend features for given stock data and market data.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data with 'Close' prices.
    market_data (pd.DataFrame): DataFrame containing market index data with 'Close' prices.
    window_size (int): Look-back window size (L) for calculating trend features.

    Returns:
    pd.DataFrame: DataFrame with rolling asset trend features.
    """

    # Calculate returns for both stock and market index
    stock_returns = stock_data['Close'].pct_change()
    market_returns = market_data['Close'].pct_change()

    # Initialize DataFrame to hold the features
    trend_features = pd.DataFrame(index=stock_data.index)

    # Maximum Drawdown and Maximum Drawup within the look-back window
    rolling_max = stock_data['Close'].rolling(window=window_size, min_periods=1).max()
    rolling_min = stock_data['Close'].rolling(window=window_size, min_periods=1).min()
    trend_features['TF_Max_Drawdown'] = stock_data['Close'] / rolling_max - 1.0
    trend_features['TF_Max_Drawup'] = stock_data['Close'] / rolling_min - 1.0

    # Skewness and Kurtosis of returns over the look-back window
    trend_features['Return_Skewness'] = stock_returns.rolling(window=window_size).skew()
    trend_features['Return_Kurtosis'] = stock_returns.rolling(window=window_size).kurt()

    # Rolling window regression to calculate Asset-To-Index Alpha and Beta
    y = stock_returns.rolling(window=window_size).apply(lambda x: sm.add_constant(market_returns.loc[x.index]).pipe(lambda y: sm.OLS(x, y).fit().params['const']), raw=False)
    x = stock_returns.rolling(window=window_size).apply(lambda x: sm.add_constant(market_returns.loc[x.index]).pipe(lambda y: sm.OLS(x, y).fit().params['Close']), raw=False)
    trend_features['Asset_To_Index_Alpha'] = y
    trend_features['Asset_To_Index_Beta'] = x

    return trend_features


def compute_and_save_features():
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    stock_data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    market_data = yf.download('SPY', start_date, end_date, interval='1d')

    stock_data.index = stock_data.index.tz_localize(None)
    market_data.index = market_data.index.tz_localize(None)

    # Reindex market_data to match stock_data's index, forward-filling missing values
    market_data = market_data.reindex(stock_data.index, method='ffill')
    # Now, drop rows from market_data where the index is not present in stock_data
    market_data = market_data[market_data.index.isin(stock_data.index)]

    features = compute_features(stock_data, market_data, window_size=60)

    features.dropna(inplace=True)

    features.to_csv('test_features.csv')

def find_nan_intervals(df: pd.DataFrame) -> dict:
    """
    Find columns with NaN values in a DataFrame and determine the first and last date of NaN values.
    If NaN values are divided into multiple intervals, returns first and last date of each interval.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    dict: A dictionary with column names as keys and lists of tuples (start_date, end_date) of NaN intervals as values.
    """
    _df = df.reset_index(drop=True)
    nan_intervals = {}
    for column in _df.columns:
        if _df[column].isna().any():
            # Create a boolean series where True represents NaN
            is_nan = _df[column].isna()

            # Use cumsum to identify continuous NaN intervals
            intervals = is_nan.ne(is_nan.shift()).cumsum()

            # Filter out non-NaN intervals and group by the interval number
            nan_groups = _df[is_nan].groupby(intervals[is_nan])

            # Extract first and last indices (dates) for each interval
            nan_intervals[column] = [(group.index[0], group.index[-1]) for _, group in nan_groups]

    return nan_intervals

if __name__ == '__main__':
    compute_and_save_features()
    features = pd.read_csv('test_features.csv', index_col=0)

    print(features.head())
