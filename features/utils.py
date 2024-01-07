import sys
sys.path.insert(0, '../')

import pandas as pd

from features import rolling_asset_statistical_features
from features import rolling_asset_technical_indicators
from features import rolling_asset_to_market_index_features
from features import time_features
from features import rolling_asset_trend_features


# Constants
MINUTE_FREQUENCY = '1m'
DAILY_FREQUENCY = '1d'


# TODO -  Rename window size variables
def compute_features(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int, trend_window_size: int) -> pd.DataFrame:
    # Detect data frequency
    data_frequency = detect_data_frequency(stock_data)

    # Initialize a DataFrame to hold the features
    features = pd.DataFrame(index=stock_data.index)

    # 1. Rolling Asset Statistical Features
    asf = rolling_asset_statistical_features.compute(stock_data, window_size)

    # 2. Rolling Asset Technical Indicators
    ati = rolling_asset_technical_indicators.compute(stock_data, window_size/2)

    # 3. Rolling Asset-to-Market Index Features
    atm = rolling_asset_to_market_index_features.compute(stock_data, market_data, window_size)

    # 4. Time Features
    tf = time_features.compute(stock_data, data_frequency)

    # 5. Rolling Asset Trend Features
    atf = rolling_asset_trend_features.compute(stock_data, market_data, trend_window_size)

    features = pd.concat([features, asf, ati, atm, atf, tf], axis=1)

    return features


def detect_data_frequency(stock_data: pd.DataFrame) -> str:
    # Calculate the minimum time difference between rows
    min_diff = stock_data.index.to_series().diff().min()

    # Determine if the data is daily or minute
    if min_diff > pd.Timedelta(minutes=1):
        return DAILY_FREQUENCY
    else:
        return MINUTE_FREQUENCY
