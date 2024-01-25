import os
import yfinance as yf
import pandas as pd
from functools import reduce
import numpy as np
import copy

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


TICKER_SYMBOL = 'CL'
#START_DATE = None#'2000-01-01'
#END_DATE = None#'2024-01-01'
INTERVAL = '1m'
SAMPLE_NUM_BEFORE_PROCESSING = 210000 # 210k

# 1/7 of data is used for testing, 6/7 for training
#TRAIN_RATIO = 6 / 7

STATISTICAL_W = 60
TECHNICAL_W = 28
MARKET_W = 60
TREND_W = 2200

CT_TWO_STATE_PARAMS = {
    'tau': 0.00104
}
CT_THREE_STATE_PARAMS = {
    'tau': 0.002,
    'w': 19
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
    'vol_span': 2,
    'f1_window': 30,
}


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def format_date(date):
    return date.strftime('%Y-%m-%d')


def read_csvs_to_dataframe(directory):
    dataframes = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            dataframes.append(df)

    combined_df = pd.concat(dataframes)

    # Sort the DataFrame by the index (datetime in this case)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df.sort_index(inplace=True)

    # Filter the DataFrame to keep rows between 9:30 and 16:00 hours; trading hours
    combined_df = combined_df.between_time('09:30', '16:00')

    return combined_df


def custom_feature_computation(stock_data: pd.DataFrame, market_data: pd.DataFrame, statistical_w: int, technical_w: int, market_w: int, trend_w: int) -> pd.DataFrame:
    asf = pd.DataFrame(columns=['Returns', 'Volatility', 'Max_Drawdown', 'Max_Drawup', 'Volume_Change'])
    ati = pd.DataFrame(columns=['ADX', 'APO', 'CCI', 'DX', 'MFI', 'RSI', 'ULTOSC', 'WILLR', 'NATR'])
    atm = pd.DataFrame(columns=['Alpha', 'Beta', 'Index_Returns', 'Index_Volatility', 'Correlation', 'Covariance', 'Beta_Direct'])
    tf = pd.DataFrame(columns=['sin_Day_of_Week', 'cos_Day_of_Week', 'sin_Day_of_Month', 'cos_Day_of_Month', 'sin_Day_of_Year', 'cos_Day_of_Year', 'sin_time_of_day', 'cos_time_of_day'])	
    
    for date, group in stock_df.groupby(pd.Grouper(freq='D')):
        if group.empty:
            continue
        # Compute features for the current day
        asf = pd.concat([asf, feat_utils.rolling_asset_statistical_features.compute(copy.deepcopy(group), statistical_w)])
        ati = pd.concat([ati, feat_utils.rolling_asset_technical_indicators.compute(copy.deepcopy(group), technical_w)])
        atm = pd.concat([atm, feat_utils.rolling_asset_to_market_index_features.compute(copy.deepcopy(group), market_data.loc[group.index], market_w)])
        tf = pd.concat([tf, feat_utils.time_features.compute(copy.deepcopy(group), INTERVAL)])
    
    atf = feat_utils.rolling_asset_trend_features.compute(copy.deepcopy(stock_df), market_data, trend_w)

    # Concatenate all features into one dataframe and remove rows where any of the features are NaN
    feat_df = pd.concat([stock_data, asf, ati, atm, atf, tf], axis=1)
    feat_df.dropna(inplace=True)
    # Sort the DataFrame by the index (datetime in this case)
    feat_df.index = pd.to_datetime(feat_df.index)
    feat_df.sort_index(inplace=True)

    return feat_df


if __name__ == '__main__':

    #########################################
    # DOWNLOAD DATA AND COMPUTE FEATURES
    #########################################
    stock_df = read_csvs_to_dataframe(f'/home/mdzaja/MasterThesis/artifacts/unprocessed_data/{TICKER_SYMBOL}')[:SAMPLE_NUM_BEFORE_PROCESSING]
    original_stock_df = copy.deepcopy(stock_df)
    market_df = read_csvs_to_dataframe(f'/home/mdzaja/MasterThesis/artifacts/unprocessed_data/SPX').iloc[:SAMPLE_NUM_BEFORE_PROCESSING]
    print(f'Loaded {stock_df.shape} stock data and {market_df.shape} market data')

    # Adjust timezone localization
    # TODO For now, timezones have to be set to None because of the way the labels are computed (Triple Barrier); Didn't have time to fix it.
    stock_df.index = stock_df.index.tz_localize(None)
    market_df.index = market_df.index.tz_localize(None)

    #TODO kako ovo odradit, ocako intersection ili fillat market_df sa closest values za sve stock_df indexe
    # Reindex data to common index
    common_index = stock_df.index.intersection(market_df.index)
    stock_df = stock_df.reindex(common_index)
    stock_df = stock_df.groupby(stock_df.index.date).filter(lambda x: len(x) >= 380)
    market_df = market_df.reindex(stock_df.index)

    feat_df = custom_feature_computation(copy.deepcopy(stock_df), market_df, statistical_w=STATISTICAL_W, technical_w=TECHNICAL_W, market_w=MARKET_W, trend_w=TREND_W)
    feat_df.dropna(inplace=True)

    # Reindex raw and feature dataframes to common index before splitting
    common_index = stock_df.index.intersection(feat_df.index)
    stock_df = stock_df.reindex(common_index)
    feat_df = feat_df.reindex(common_index)

    # Split data into train and test sets where test set is the 10 working days
    unique_dates = stock_df.index.normalize().unique()
    last_10_working_days = unique_dates[-10:]
    train_mask = stock_df.index.normalize().isin(unique_dates[:-10])
    test_mask = stock_df.index.normalize().isin(last_10_working_days)

    # Split data into train and test sets
    train_raw_df = stock_df[train_mask]
    test_raw_df = stock_df[test_mask]
    train_feat_df = feat_df[train_mask]
    test_feat_df = feat_df[test_mask]

    #########################################
    # COMPUTE LABELS
    #########################################
    labels_dict = {
        'ct_two_state': pd.Series(dtype=np.int64),
        'ct_three_state': pd.Series(dtype=np.int64),
        'fixed_time_horizon': pd.Series(dtype=np.int64),
        'oracle': pd.Series(dtype=np.int64),
        'triple_barrier': pd.Series(dtype=np.int64)
    }
    prices = original_stock_df['Close']

    for date, group in prices.groupby(pd.Grouper(freq='D')):
        if group.empty:
            continue

        ct_two_state_lbl = ct2.binary_trend_labels(group, 
                                            tau=CT_TWO_STATE_PARAMS['tau']).dropna()
        labels_dict['ct_two_state'] = pd.concat([labels_dict['ct_two_state'], ct_two_state_lbl])

        ct_three_state_lbl = ct3.binary_trend_labels(group, 
                                                tau=CT_THREE_STATE_PARAMS['tau'], 
                                                w=CT_THREE_STATE_PARAMS['w']).dropna()
        labels_dict['ct_three_state'] = pd.concat([labels_dict['ct_three_state'], ct_three_state_lbl])

        fixed_time_horizon_lbl = fth.binary_trend_labels(group, 
                                                    tau=FIXED_TIME_HORIZON['tau'], 
                                                    H=FIXED_TIME_HORIZON['H']).dropna()
        labels_dict['fixed_time_horizon'] = pd.concat([labels_dict['fixed_time_horizon'], fixed_time_horizon_lbl])

        oracle_lbl = oracle.binary_trend_labels(group, 
                                            fee=ORACLE_PARAMS['fee']).dropna()
        labels_dict['oracle'] = pd.concat([labels_dict['oracle'], oracle_lbl])

        tEvents = group.index
        t1 = group.index.searchsorted(tEvents + pd.Timedelta(days=TRIPLE_BARRIER_PARAMS['f1_window']))
        t1 = pd.Series((group.index[i] if i < group.shape[0] else pd.NaT for i in t1), index=tEvents)
        volatility = tb.getVolatility(group, span=TRIPLE_BARRIER_PARAMS['vol_span'])
        volatility = volatility.reindex(tEvents).loc[tEvents].fillna(method='bfill')
        triple_barrier_lbl = tb.binary_trend_labels(group,
                                                tEvents,
                                                pt=TRIPLE_BARRIER_PARAMS['pt'], 
                                                sl=TRIPLE_BARRIER_PARAMS['sl'], 
                                                volatility=volatility, 
                                                minRet=0, 
                                                t1=t1).dropna()
        labels_dict['triple_barrier'] = pd.concat([labels_dict['triple_barrier'], triple_barrier_lbl])

    train_labels_dict = {}
    test_labels_dict = {}

    for key in labels_dict.keys():
        labels_dict[key].dropna(inplace=True)
        train_indices = labels_dict[key].index.intersection(train_raw_df.index)
        train_labels_dict[key] = labels_dict[key].loc[train_indices]
        test_indices = labels_dict[key].index.intersection(test_raw_df.index)
        test_labels_dict[key] = labels_dict[key].loc[test_indices]

    #########################################
    # COMPUTE WEIGHTS
    #########################################
    weights_dict = {}

    for label_name, labels in labels_dict.items():
        weights_dict[label_name] = {
            'backward_looking': pd.Series(dtype=np.float64),
            'forward_looking': pd.Series(dtype=np.float64),
            'sequential_return': pd.Series(dtype=np.float64),
            'trend_interval_return': pd.Series(dtype=np.float64)
        }
        for date, group in labels.groupby(pd.Grouper(freq='D')):
            if group.empty:
                continue
            weights_dict[label_name]['backward_looking'] = pd.concat([
                                            weights_dict[label_name]['backward_looking'], 
                                            bl.get_weights(prices.loc[group.index], group)])
            weights_dict[label_name]['forward_looking'] = pd.concat([
                                            weights_dict[label_name]['forward_looking'], 
                                            fl.get_weights(prices.loc[group.index], group)])
            weights_dict[label_name]['sequential_return'] = pd.concat([
                                            weights_dict[label_name]['sequential_return'], 
                                            sr.get_weights(prices.loc[group.index], group)])
            weights_dict[label_name]['trend_interval_return'] = pd.concat([
                                            weights_dict[label_name]['trend_interval_return'], 
                                            tir.get_weights(prices.loc[group.index], group)])

    train_weights_dict = {}
    test_weights_dict = {}

    for label_name, weights in weights_dict.items():
        train_weights_dict[label_name] = {}
        test_weights_dict[label_name] = {}
        for weight_name, weight in weights.items():
            weight.dropna(inplace=True)
            train_indices = weight.index.intersection(train_raw_df.index)
            train_weights_dict[label_name][weight_name] = weight.loc[train_indices]
            test_indices = weight.index.intersection(test_raw_df.index)
            test_weights_dict[label_name][weight_name] = weight.loc[test_indices]

    #########################################
    # ALIGN ALL DATA
    #########################################
    # Get the intersection of all indices
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
    
    # Reindex all dataframes to the common indices
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
    train_raw_df.to_csv(f'{base_path}/data/raw/train_{INTERVAL}_{format_date(train_raw_df.index[0])}_{format_date(train_raw_df.index[-1])}.csv')
    test_raw_df.to_csv(f'{base_path}/data/raw/test_{INTERVAL}_{format_date(test_raw_df.index[0])}_{format_date(test_raw_df.index[-1])}.csv')

    ensure_directory_exists(f'{base_path}/data/feat')
    train_feat_df.to_csv(f'{base_path}/data/feat/train_{INTERVAL}_{format_date(train_feat_df.index[0])}_{format_date(train_feat_df.index[-1])}.csv')
    test_feat_df.to_csv(f'{base_path}/data/feat/test_{INTERVAL}_{format_date(test_feat_df.index[0])}_{format_date(test_feat_df.index[-1])}.csv')

    ensure_directory_exists(f'{base_path}/labels')
    pd.to_pickle(train_labels_dict, f'{base_path}/labels/all_labels_train_{INTERVAL}_{format_date(train_labels_dict["ct_two_state"].index[0])}_{format_date(train_labels_dict["ct_two_state"].index[-1])}.pkl')
    pd.to_pickle(test_labels_dict, f'{base_path}/labels/all_labels_test_{INTERVAL}_{format_date(test_labels_dict["ct_two_state"].index[0])}_{format_date(test_labels_dict["ct_two_state"].index[-1])}.pkl')

    ensure_directory_exists(f'{base_path}/weights')
    pd.to_pickle(train_weights_dict, f'{base_path}/weights/all_weights_train_{INTERVAL}_{format_date(train_weights_dict["ct_two_state"]["backward_looking"].index[0])}_{format_date(train_weights_dict["ct_two_state"]["backward_looking"].index[-1])}.pkl')
    pd.to_pickle(test_weights_dict, f'{base_path}/weights/all_weights_test_{INTERVAL}_{format_date(test_weights_dict["ct_two_state"]["backward_looking"].index[0])}_{format_date(test_weights_dict["ct_two_state"]["backward_looking"].index[-1])}.pkl')
