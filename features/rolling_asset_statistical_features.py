import pandas as pd


def compute(stock_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
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
