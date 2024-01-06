import os
import yfinance as yf
import pandas as pd
from functools import reduce
import numpy as np

from features import utils as feat_utils
from labels import ct_two_state as ct2
from labels import ct_three_state as ct3
from labels import fixed_time_horizon as fth
from labels import oracle
from labels import triple_barrier as tb
from weights import backward_looking as bl
from weights import forward_looking as fl
from weights import sequential_return as sr
from weights import trend_interval_return as tir


TICKER_SYMBOL = 'GC=F'
START_DATE = '2000-01-01'
END_DATE = '2024-01-01'

# 1/7 of data is used for testing, 6/7 for training
TRAIN_RATIO = 6 / 7

CT_TWO_STATE_PARAMS = {
    'tau': 0.00298
}
CT_THREE_STATE_PARAMS = {
    'tau': 0.00176,
    'w': 16
}
FIXED_TIME_HORIZON = {
    'tau': 0,
    'H': 1
}
ORACLE_PARAMS = {
    'fee': 0.0004
}
TRIPLE_BARRIER_PARAMS = {
    'pt': 0.1,
    'sl': 0.4,
    'vol_span': 40,
    'f1_window': 30,
}


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_date(date):
    return date.strftime('%Y-%m-%d')


if __name__ == '__main__':

    #########################################
    # DOWNLOAD DATA AND COMPUTE FEATURES
    #########################################
    stock_df = yf.download(TICKER_SYMBOL, START_DATE, END_DATE, interval='1d')
    market_df = yf.download('SPY', START_DATE, END_DATE, interval='1d')

    # Reindex market_data to match stock_data's index, forward-filling missing values
    market_df = market_df.reindex(stock_df.index, method='ffill')

    feat_df = feat_utils.compute_features(stock_df, market_df, window_size=60)
    feat_df.dropna(inplace=True)

    # Reindex stock_df to match feat_df's index, dropping rows where the index is not present in feat_df
    stock_df = stock_df.reindex(feat_df.index)

    total_rows = len(stock_df)
    split_index = int(total_rows * TRAIN_RATIO)

    train_raw_df = stock_df.iloc[:split_index]
    test_raw_df = stock_df.iloc[split_index:]
    train_feat_df = feat_df.iloc[:split_index]
    test_feat_df = feat_df.iloc[split_index:]

    #########################################
    # COMPUTE LABELS
    #########################################
    labels_dict = {}
    prices = stock_df['Close']

    labels_dict['ct_two_state'] = ct2.binary_trend_labels(prices, 
                                                          tau=CT_TWO_STATE_PARAMS['tau']).dropna()

    labels_dict['ct_three_state'] = ct3.binary_trend_labels(prices, 
                                                            tau=CT_THREE_STATE_PARAMS['tau'], 
                                                            w=CT_THREE_STATE_PARAMS['w']).dropna()

    labels_dict['fixed_time_horizon'] = fth.binary_trend_labels(prices, 
                                                                tau=FIXED_TIME_HORIZON['tau'], 
                                                                H=FIXED_TIME_HORIZON['H']).dropna()

    labels_dict['oracle'] = oracle.binary_trend_labels(prices, 
                                                       fee=ORACLE_PARAMS['fee']).dropna()

    tEvents = prices.index
    t1 = prices.index.searchsorted(tEvents + pd.Timedelta(days=TRIPLE_BARRIER_PARAMS['f1_window']))
    t1 = pd.Series((prices.index[i] if i < prices.shape[0] else pd.NaT for i in t1), index=tEvents)
    dayVol = tb.getDayVol(prices, span=TRIPLE_BARRIER_PARAMS['vol_span'])
    dayVol = dayVol.reindex(tEvents).loc[tEvents].fillna(method='bfill')
    labels_dict['triple_barrier'] = tb.binary_trend_labels(prices, 
                                                           tEvents, 
                                                           pt=TRIPLE_BARRIER_PARAMS['pt'], 
                                                           sl=TRIPLE_BARRIER_PARAMS['sl'], 
                                                           volatility=dayVol, 
                                                           minRet=0, 
                                                           t1=t1).dropna()

    train_labels_dict = {}
    test_labels_dict = {}

    for key in labels_dict.keys():
        train_labels_dict[key] = labels_dict[key].iloc[:split_index]
        test_labels_dict[key] = labels_dict[key].iloc[split_index:]

    #########################################
    # COMPUTE WEIGHTS
    #########################################
    weights_dict = {}

    for label_name, labels in labels_dict.items():
        weights_dict[label_name] = {}
        weights_dict[label_name]['backward_looking'] = bl.get_weights(prices, labels).dropna()
        weights_dict[label_name]['forward_looking'] = fl.get_weights(prices, labels).dropna()
        weights_dict[label_name]['sequential_return'] = sr.get_weights(prices, labels).dropna()
        weights_dict[label_name]['trend_interval_return'] = tir.get_weights(prices, labels).dropna()

    train_weights_dict = {}
    test_weights_dict = {}

    for label_name, weights in weights_dict.items():
        train_weights_dict[label_name] = {}
        test_weights_dict[label_name] = {}
        for weight_name, weight in weights.items():
            train_weights_dict[label_name][weight_name] = weight.iloc[:split_index]
            test_weights_dict[label_name][weight_name] = weight.iloc[split_index:]

    #########################################
    # ALIGN ALL DATA
    #########################################
    # Get the union of all indices
    intersection_train_index = reduce(
        lambda x, y: x.intersection(y),
        [train_feat_df.index] +
        [train_raw_df.index] +
        [df.index for df in train_labels_dict.values()] +
        [df.index for weights_dict in train_weights_dict.values() for df in weights_dict.values()]
    )
    intersection_test_index = reduce(
        lambda x, y: x.intersection(y),
        [test_feat_df.index] +
        [test_raw_df.index] +
        [df.index for df in test_labels_dict.values()] +
        [df.index for weights_dict in test_weights_dict.values() for df in weights_dict.values()]
    )
    
    # Reindex all dataframes to the union of indices
    train_raw_df = train_raw_df.reindex(intersection_train_index)
    test_raw_df = test_raw_df.reindex(intersection_test_index)

    train_feat_df = train_feat_df.reindex(intersection_train_index)
    test_feat_df = test_feat_df.reindex(intersection_test_index)

    for label_name, labels in train_labels_dict.items():
        train_labels_dict[label_name] = labels.reindex(intersection_train_index)
    for label_name, labels in test_labels_dict.items():
        test_labels_dict[label_name] = labels.reindex(intersection_test_index)
    
    for label_name, weights in train_weights_dict.items():
        for weight_name, weight in weights.items():
            train_weights_dict[label_name][weight_name] = weight.reindex(intersection_train_index)
    for label_name, weights in test_weights_dict.items():
        for weight_name, weight in weights.items():
            test_weights_dict[label_name][weight_name] = weight.reindex(intersection_test_index)

    #########################################
    # SAVE DATA
    #########################################
    base_path = f'artifacts/assets/{TICKER_SYMBOL}'

    ensure_directory_exists(f'{base_path}/data/raw')
    train_raw_df.to_csv(f'{base_path}/data/raw/train_{format_date(train_raw_df.index[0])}_{format_date(train_raw_df.index[-1])}.csv')
    test_raw_df.to_csv(f'{base_path}/data/raw/test_{format_date(test_raw_df.index[0])}_{format_date(test_raw_df.index[-1])}.csv')

    ensure_directory_exists(f'{base_path}/data/feat')
    train_feat_df.to_csv(f'{base_path}/data/feat/train_{format_date(train_feat_df.index[0])}_{format_date(train_feat_df.index[-1])}.csv')
    test_feat_df.to_csv(f'{base_path}/data/feat/test_{format_date(test_feat_df.index[0])}_{format_date(test_feat_df.index[-1])}.csv')

    ensure_directory_exists(f'{base_path}/labels')
    pd.to_pickle(train_labels_dict, f'{base_path}/labels/all_labels_train_{format_date(train_labels_dict["ct_two_state"].index[0])}_{format_date(train_labels_dict["ct_two_state"].index[-1])}.pkl')
    pd.to_pickle(test_labels_dict, f'{base_path}/labels/all_labels_test_{format_date(test_labels_dict["ct_two_state"].index[0])}_{format_date(test_labels_dict["ct_two_state"].index[-1])}.pkl')

    ensure_directory_exists(f'{base_path}/weights')
    pd.to_pickle(train_weights_dict, f'{base_path}/weights/all_weights_train_{format_date(train_weights_dict["ct_two_state"]["backward_looking"].index[0])}_{format_date(train_weights_dict["ct_two_state"]["backward_looking"].index[-1])}.pkl')
    pd.to_pickle(test_weights_dict, f'{base_path}/weights/all_weights_test_{format_date(test_weights_dict["ct_two_state"]["backward_looking"].index[0])}_{format_date(test_weights_dict["ct_two_state"]["backward_looking"].index[-1])}.pkl')
