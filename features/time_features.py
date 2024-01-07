import sys
sys.path.insert(0, '../')

import numpy as np
import pandas as pd

from features import utils as feat_utils


def compute(stock_data: pd.DataFrame, data_frequency: str) -> pd.DataFrame:
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
    if data_frequency == feat_utils.MINUTE_FREQUENCY:
        # Calculate the total minutes in a day
        minutes_in_day = 24 * 60
        # Calculate the minute of the day
        minute_of_day = stock_data.index.hour * 60 + stock_data.index.minute
        # Apply sine and cosine transformations
        time_features['sin_time_of_day'] = np.sin(2 * np.pi * minute_of_day / minutes_in_day)
        time_features['cos_time_of_day'] = np.cos(2 * np.pi * minute_of_day / minutes_in_day)

    return time_features