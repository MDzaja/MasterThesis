import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

sys.path.append('../')
from label_algorithms import oracle


def get_X(features, window_size):
    # Normalize features to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features.values)

    # Create the 3D input data shape [samples, time_steps, features]
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, :])

    return np.array(X)


def get_Y(labels, window_size):
    return labels[window_size:].shift(-1)[:-1].values.astype(int).reshape(-1, 1)


def get_dummy_X_n_Y():
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    prices = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    prices.index = prices.index.tz_localize(None)

    fee = 0.0004
    labels = oracle.binary_trend_labels(prices['Close'], fee=fee)

    window_size = 40
    X = get_X(prices, window_size)[:-1]
    Y = get_Y(labels, window_size)
    
    return X, Y
