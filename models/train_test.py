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

def test_models(labeling, data_type, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    model_configs = {
        'cnn_lstm': {
            'build_func': cnn_lstm_impl.build_model_raw if data_type == 'raw_data' else cnn_lstm_impl.build_model,
            'epochs': 1000,
            'batch_size': 64,
            'patience': 100
        },
        'lstm': {
            'build_func': lstm_impl.build_model_raw if data_type == 'raw_data' else cnn_lstm_impl.build_model,
            'epochs': 1000,
            'batch_size': 64,
            'patience': 100
        },
        'transformer': {
            'build_func': tr_impl.build_model_raw if data_type == 'raw_data' else cnn_lstm_impl.build_model,
            'epochs': 1000,
            'batch_size': 64,
            'patience': 100
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
        model.save(f"test_logs/test2/saved_models/{data_type}-{labeling}-{model_name}.keras")

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
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl()

    labels_dict = labels_dict['ct_three_state', 'ct_two_state', 'fixed_time_horizon', 'oracle']

    window_size = 60
    raw_X = model_utils.get_X(raw_data, window_size)
    feat_X = model_utils.get_X(features_df, window_size)
    Y_dict = {key: model_utils.get_Y(series, window_size) for key, series in labels_dict.items()}

    raw_X_train, raw_X_val, feat_X_train, feat_X_val = train_test_split(raw_X, feat_X, test_size=0.2, shuffle=False)
    raw_X_val, raw_X_test, feat_X_val, feat_X_test = train_test_split(raw_X_val, feat_X_val, test_size=0.25, shuffle=False)

    Y_train_dict, Y_val_dict = {}, {}
    for key in Y_dict.keys():
        Y_train_dict[key], Y_val_dict[key] = train_test_split(Y_dict[key], test_size=0.2, shuffle=False)
    Y_test_dict, feat_Y_test_dict = {}, {}
    for key in Y_val_dict.keys():
        Y_val_dict[key], Y_test_dict[key] = train_test_split(Y_val_dict[key], test_size=0.25, shuffle=False)

    # Xs: raw_X_train, raw_X_val, raw_X_test, feat_X_train, feat_X_val, feat_X_test
    # Ys: raw_Y_train_dict, raw_Y_val_dict, raw_Y_test_dict, feat_Y_train_dict, feat_Y_val_dict, feat_Y_test_dict

    metrics = {}
    data_type = 'raw_data'
    metrics[data_type] = {}
    for labeling in labels_dict.keys():
        metrics[data_type][labeling] = test_models(labeling, data_type,
                                                   raw_X_train, raw_X_val, raw_X_test, 
                                                   Y_train_dict[labeling], Y_val_dict[labeling], Y_test_dict[labeling]
                                                   )
    data_type = 'features'
    metrics[data_type] = {}
    for labeling in labels_dict.keys():
        metrics[data_type][labeling] = test_models(labeling, data_type,
                                                   feat_X_train, feat_X_val, feat_X_test, 
                                                   Y_train_dict[labeling], Y_val_dict[labeling], Y_test_dict[labeling]
                                                   )

    # Write metrics to JSON file
    results_filename = f"test_logs/test2/metrics.json"
    with open(results_filename, 'w') as file:
        json.dump(metrics, file, indent=6)
