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
STATISTICAL_W = 60
TECHNICAL_W = 28
MARKET_W = 60
TREND_W = 2200
SAVE_PATH = '/home/mdzaja/MasterThesis/artifacts/features/features.csv'


if __name__ == '__main__':
    # Download stock data and market data using DATA_FREQUENCY for the interval
    stock_data = yf.download(TICKER_SYMBOL, START_DATE, END_DATE, interval=DATA_FREQUENCY)
    market_data = yf.download('SPY', START_DATE, END_DATE, interval=DATA_FREQUENCY)

    # Adjust timezone localization
    stock_data.index = stock_data.index.tz_convert('UTC')
    market_data.index = market_data.index.tz_convert('UTC')

    # Reindex data to common index
    common_index = stock_data.index.intersection(market_data.index)
    stock_data = stock_data.reindex(common_index)
    market_data = market_data.reindex(common_index)

    # Compute features
    features = feat_utils.compute_features(stock_data, market_data, STATISTICAL_W, TECHNICAL_W, MARKET_W, TREND_W)

    # Drop NaN values
    features.dropna(inplace=True)

    # Save features to CSV
    features.to_csv(SAVE_PATH)

    ##########################################################

    # Load features from CSV
    features = pd.read_csv(SAVE_PATH, index_col=0)

    print(features.head())