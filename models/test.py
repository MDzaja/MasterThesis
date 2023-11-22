import sys
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import json

# Importing keras from tensorflow
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM, GRU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay


import LSTM as lstm_impl
import CNN_LSTM as cnn_lstm_impl
import utils as model_utils
import transformer as tr_impl

def test_models(labeling, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    model_configs = {
        'cnn_lstm': {
            'build_func': cnn_lstm_impl.build_model,
            'epochs': 1,
            'batch_size': 64,
            'patience': 1
        },
        'lstm': {
            'build_func': lstm_impl.build_model,
            'epochs': 1,
            'batch_size': 64,
            'patience': 1
        },
        'transformer': {
            'build_func': tr_impl.build_model,
            'epochs': 1,
            'batch_size': 64,
            'patience': 1
        }
    }

    results = {}

    for model_name, config in model_configs.items():
        _X_train = X_train
        _X_val = X_val
        _X_test = X_test
        if model_name == 'cnn_lstm':
            n_steps = int(_X_train.shape[1] / 6)
            n_steps = n_steps if n_steps != 0 else 1
            n_length = int(_X_train.shape[1] / n_steps)
            n_features = _X_train.shape[-1]
            _X_train = _X_train.reshape((_X_train.shape[0], n_steps, n_length, n_features))
            _X_val = _X_val.reshape((_X_val.shape[0], n_steps, n_length, n_features))
            _X_test = _X_test.reshape((_X_test.shape[0], n_steps, n_length, n_features))


        # Build and train the model
        model = config['build_func'](_X_train.shape[-2], _X_train.shape[-1])
        early_stopping = EarlyStopping(monitor='loss', patience=config['patience'])
        model.fit(_X_train, Y_train, epochs=config['epochs'], validation_data=(_X_val, Y_val), 
                  batch_size=config['batch_size'], class_weight=class_weight_dict, callbacks=[early_stopping])

        # Evaluate the model
        train_eval = model.evaluate(_X_train, Y_train)
        val_eval = model.evaluate(_X_val, Y_val)
        test_eval = model.evaluate(_X_test, Y_test)

        # Save the model
        model.save(f"test-logs/saved_models/{labeling}-{model_name}.keras")

        # Store results
        results[model_name] = {
            "train": {
                "loss": train_eval[0],
                "accuracy": train_eval[1],
                "precision": train_eval[2],
                "recall": train_eval[3],
                "auc": train_eval[4]
            },
            "validation": {
                "loss": val_eval[0],
                "accuracy": val_eval[1],
                "precision": val_eval[2],
                "recall": val_eval[3],
                "auc": val_eval[4]
            },
            "test": {
                "loss": test_eval[0],
                "accuracy": test_eval[1],
                "precision": test_eval[2],
                "recall": test_eval[3],
                "auc": test_eval[4]
            }
        }

    return results


if __name__ == '__main__':
    ticker_symbol = 'GC=F'
    start_date = '1995-01-01'
    end_date = '2023-11-01'
    raw_data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    raw_data.index = raw_data.index.tz_localize(None)

    features_df = pd.read_csv('../features/test_features.csv', index_col=0)

    with open('../label_algorithms/labels_dict.pkl', 'rb') as file:
        labels_dict = pickle.load(file)

    # Ensure indices are in the same format
    raw_data.index = pd.to_datetime(raw_data.index)
    features_df.index = pd.to_datetime(features_df.index)
    labels_dict = {k: pd.Series(v, index=pd.to_datetime(v.index)) for k, v in labels_dict.items()}

    # Find the common indices
    common_indices = raw_data.index.intersection(features_df.index)
    for label_series in labels_dict.values():
        common_indices = common_indices.intersection(label_series.index)

    # Reindex raw_data, features_df, and each Series in labels_dict
    raw_data = raw_data.reindex(common_indices)
    features_df = features_df.reindex(common_indices)
    labels_dict = {key: series.reindex(common_indices) for key, series in labels_dict.items()}

    window_size = 60
    raw_X = model_utils.get_X(raw_data, window_size)
    raw_Y_dict = {key: model_utils.get_Y(series, window_size) for key, series in labels_dict.items()}
    feat_X = model_utils.get_X(features_df, 1)[window_size-1:]
    feat_Y_dict = {key: model_utils.get_Y(series, 1)[window_size-1:] for key, series in labels_dict.items()}

    raw_X_train, raw_X_val, feat_X_train, feat_X_val = train_test_split(raw_X, feat_X, test_size=0.2, shuffle=False)
    raw_Y_train_dict, raw_Y_val_dict, feat_Y_train_dict, feat_Y_val_dict = {}, {}, {}, {}
    for key in raw_Y_dict.keys():
        raw_Y_train_dict[key], raw_Y_val_dict[key], feat_Y_train_dict[key], feat_Y_val_dict[key] = train_test_split(raw_Y_dict[key], feat_Y_dict[key], test_size=0.2, shuffle=False)

    raw_X_val, raw_X_test, feat_X_val, feat_X_test = train_test_split(raw_X_val, feat_X_val, test_size=0.25, shuffle=False)
    raw_Y_test_dict, feat_Y_test_dict = {}, {}
    for key in raw_Y_val_dict.keys():
        raw_Y_val_dict[key], raw_Y_test_dict[key], feat_Y_val_dict[key], feat_Y_test_dict[key] = train_test_split(raw_Y_val_dict[key], feat_Y_val_dict[key], test_size=0.25, shuffle=False)

    # Xs: raw_X_train, raw_X_val, raw_X_test, feat_X_train, feat_X_val, feat_X_test
    # Ys: raw_Y_train_dict, raw_Y_val_dict, raw_Y_test_dict, feat_Y_train_dict, feat_Y_val_dict, feat_Y_test_dict

    metrics = {}
    data_type = 'raw_data'
    metrics[data_type] = {}
    for labeling in labels_dict.keys():
        metrics[data_type][labeling] = test_models(labeling, 
                                                   raw_X_train, raw_X_val, raw_X_test, 
                                                   raw_Y_train_dict[labeling], raw_Y_val_dict[labeling], raw_Y_test_dict[labeling]
                                                   )
        break
    data_type = 'features'
    metrics[data_type] = {}
    for labeling in labels_dict.keys():
        metrics[data_type][labeling] = test_models(labeling, 
                                                   feat_X_train, feat_X_val, feat_X_test, 
                                                   feat_Y_train_dict[labeling], feat_Y_val_dict[labeling], feat_Y_test_dict[labeling]
                                                   )
        break
    # Write metrics to JSON file
    results_filename = f"test-logs/metrics.json"
    with open(results_filename, 'w') as file:
        json.dump(metrics, file, indent=6)
