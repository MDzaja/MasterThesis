import sys
sys.path.insert(0, '../')

import pandas as pd
import yfinance as yf

from features import utils as feat_utils


# Parameters
TICKER_SYMBOL = 'GC=F'
START_DATE = '2000-01-01'
END_DATE = '2023-11-01'
DATA_FREQUENCY = feat_utils.DAILY_FREQUENCY
# TODO - Rename these variables
WINDOW_SIZE_1 = 60
WINDOW_SIZE_2 = 2200


if __name__ == '__main__':
    # Download stock data and market data using DATA_FREQUENCY for the interval
    stock_data = yf.download(TICKER_SYMBOL, START_DATE, END_DATE, interval=DATA_FREQUENCY)
    market_data = yf.download('SPY', START_DATE, END_DATE, interval=DATA_FREQUENCY)

    # Adjust timezone localization
    stock_data.index = stock_data.index.tz_localize(None)
    market_data.index = market_data.index.tz_localize(None)

    # Reindex and forward-fill market data
    market_data = market_data.reindex(stock_data.index, method='ffill')
    market_data = market_data[market_data.index.isin(stock_data.index)]

    # Compute features
    features = feat_utils.compute_features(stock_data, market_data, WINDOW_SIZE_1, WINDOW_SIZE_2)

    # Drop NaN values
    features.dropna(inplace=True)

    # Save features to CSV
    features.to_csv('test_features.csv')

    ##########################################################

    # Load features from CSV
    features = pd.read_csv('test_features.csv', index_col=0)

    print(features.head())