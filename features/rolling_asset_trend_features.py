import pandas as pd
import statsmodels.api as sm


def compute(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int=2200) -> pd.DataFrame:
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