# Run example: python run.py 0 --use_sample_weights path/to/config.json path/to/results_directory

import sys
sys.path.insert(0, '../../')

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import json
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import yaml

from models import LSTM as lstm_impl
from models import CNN_LSTM as cnn_lstm_impl
from models import transformer as tr_impl
from models import utils as model_utils

dir_path = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run models based on a configuration file.")
    parser.add_argument('gpu_id', type=str, choices=['0', '1'], help='GPU ID')
    parser.add_argument('--use_sample_weights', action='store_true', help='Use sample weights')
    parser.add_argument('config_path', type=str, help='Path to the configuration JSON file')
    parser.add_argument('directory', type=str, help='Path to result directory')
    args = parser.parse_args()
    return args


def test_models(labeling, data_type, X_train, X_val, X_test, Y_train, Y_val, Y_test, 
                sample_weights=None, cnn_lstm=False, lstm=False, transformer=False):
    if sample_weights is None:
        sample_weights = {}
        sample_weights['train'] = np.ones(Y_train.shape[0])
        sample_weights['val'] = np.ones(Y_val.shape[0])
        sample_weights['test'] = np.ones(Y_test.shape[0])

    class_weight_dict = model_utils.calculate_class_weight_dict(Y_train)
    adjusted_sample_weights = {}
    adjusted_sample_weights['train'] = model_utils.adjust_sample_weights(Y_train, class_weight_dict, sample_weights['train'])
    adjusted_sample_weights['val'] = model_utils.adjust_sample_weights(Y_val, class_weight_dict, sample_weights['val'])

    model_configs = {}
    if cnn_lstm:
        model_configs['cnn_lstm'] = {
            'build_func': cnn_lstm_impl.build_model_raw if data_type == 'raw_data' else cnn_lstm_impl.build_model_feat,
            'epochs': 500,
            'batch_size': 64,
            'patience': 20
        }
    if lstm:
        model_configs['lstm'] = {
            'build_func': lstm_impl.build_model_raw if data_type == 'raw_data' else lstm_impl.build_model_feat,
            'epochs': 500,
            'batch_size': 64,
            'patience': 20
        }
    if transformer:
        model_configs['transformer'] = {
            'build_func': tr_impl.build_model_raw if data_type == 'raw_data' else tr_impl.build_model_feat,
            'epochs': 500,
            'batch_size': 64,
            'patience': 20
        }

    results = {}

    for model_name, config in model_configs.items():
        _X_train = X_train
        _X_val = X_val
        _X_test = X_test
        if model_name == 'cnn_lstm':
            n_length = 10
            n_steps = _X_train.shape[1] // n_length
            n_features = _X_train.shape[-1]
            _X_train = _X_train.reshape((_X_train.shape[0], n_steps, n_length, n_features))
            _X_val = _X_val.reshape((_X_val.shape[0], n_steps, n_length, n_features))
            _X_test = _X_test.reshape((_X_test.shape[0], n_steps, n_length, n_features))

        print(f"Training {model_name} on {data_type} data with {labeling} labeling...", flush=True)
        model = config['build_func'](_X_train.shape[-2], _X_train.shape[-1])
        model = model_utils.train_model(model, _X_train, Y_train, _X_val, Y_val, 
                                        adjusted_sample_weights['train'], adjusted_sample_weights['val'],
                                        config['batch_size'], config['epochs'], config['patience'], dir_path)
        print(f"Finished training {model_name} on {data_type} data with {labeling} labeling.", flush=True)

        # Save the model
        # if not os.path.exists(f"{dir_path}/saved_models"):
        #     os.makedirs(f"{dir_path}/saved_models")
        # model.save(f"{dir_path}/saved_models/{data_type}-{labeling}-{model_name}.keras")

        # Evaluate the model
        train_eval = model.evaluate(_X_train, Y_train, sample_weight=sample_weights['train'])
        val_eval = model.evaluate(_X_val, Y_val, sample_weight=sample_weights['val'])
        test_eval = model.evaluate(_X_test, Y_test, sample_weight=sample_weights['test'])

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


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def prepare_data(get_weights=False):
    # Load your raw data, features, and labels
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl(
        '../../artifacts/features/features_2009-06-22_2023-10-30.csv',
        '../../artifacts/labels/labels_dict_2000-2023_w_23y_params.pkl'
    )

    # Process raw data
    raw_X = model_utils.get_X(raw_data, 60)
    raw_Y_dict = {key: model_utils.get_Y(series, 60) for key, series in labels_dict.items()}

    # Process feature data
    feat_X = model_utils.get_X(features_df, 30)[30:]
    feat_Y_dict = {key: model_utils.get_Y(series, 30)[30:] for key, series in labels_dict.items()}

    # Get sample weights
    if get_weights:
        weights_path = '../../artifacts/weights/weights_2000-2023_w_lbl_23y_params.pkl'
        with open(weights_path, 'rb') as file:
                sample_weights = pickle.load(file)
        for label, weights_dict in sample_weights.items():
            for weight_alg, weights in weights_dict.items():
                sample_weights[label][weight_alg] = pd.Series(weights, index=pd.to_datetime(weights.index))
                sample_weights[label][weight_alg] = sample_weights[label][weight_alg].reindex(feat_Y_dict['oracle'].index)

    # Initialize dictionaries to hold split data
    X = {'raw_data': {}, 'features': {}}
    Y = {'raw_data': {}, 'features': {}}

    # Calculate split boundaries
    total_samples = raw_X.shape[0]
    train_end = int(total_samples * 0.75)
    val_end = int(total_samples * 0.90)

    # Split X data
    X['raw_data']['train'], X['raw_data']['val'], X['raw_data']['test'] = raw_X[:train_end], raw_X[train_end:val_end], raw_X[val_end:]
    X['features']['train'], X['features']['val'], X['features']['test'] = feat_X[:train_end], feat_X[train_end:val_end], feat_X[val_end:]

    # Split Y data
    for i, label in enumerate(labels_dict.keys()):
        Y['raw_data'][label] = {}
        Y['features'][label] = {}
        Y['raw_data'][label]['train'], Y['raw_data'][label]['val'], Y['raw_data'][label]['test'] = np.array(raw_Y_dict[label].iloc[:train_end]), np.array(raw_Y_dict[label].iloc[train_end:val_end]), np.array(raw_Y_dict[label].iloc[val_end:])
        Y['features'][label]['train'], Y['features'][label]['val'], Y['features'][label]['test'] = np.array(feat_Y_dict[label].iloc[:train_end]), np.array(feat_Y_dict[label].iloc[train_end:val_end]), np.array(feat_Y_dict[label].iloc[val_end:])

    # Split sample weights
    splitted_sample_weights = None
    if get_weights:
        splitted_sample_weights = {}
        for label, weights_dict in sample_weights.items():
            for weight_alg, weights in weights_dict.items():
                splitted_sample_weights[label] = splitted_sample_weights.get(label, {})
                splitted_sample_weights[label][weight_alg] = {}
                splitted_sample_weights[label][weight_alg]['train'] = np.array(weights.iloc[:train_end]).reshape(-1)
                splitted_sample_weights[label][weight_alg]['val'] = np.array(weights.iloc[train_end:val_end]).reshape(-1)
                splitted_sample_weights[label][weight_alg]['test'] = np.array(weights.iloc[val_end:]).reshape(-1)

    return X, Y, splitted_sample_weights


def run_models(config, X, Y, sample_weight_dict):
    metrics = {}
    for labeling, data_types in config.items():
        for data_type, models in data_types.items():
            metrics[data_type] = metrics.get(data_type, {})
            metrics[data_type][labeling] = test_models(labeling, data_type,
                                                       X[data_type]['train'], X[data_type]['val'], X[data_type]['test'],
                                                       Y[data_type][labeling]['train'], Y[data_type][labeling]['val'], Y[data_type][labeling]['test'],
                                                       sample_weights=sample_weight_dict[labeling]['trend_interval_return'] if sample_weight_dict else None, #TODO use all weight algorithms
                                                       cnn_lstm="cnn_lstm" in models,
                                                       lstm="lstm" in models,
                                                       transformer="transformer" in models)
    return metrics


def setup_logging(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = open(f'{directory}/output.log', 'a')
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def save_results(metrics, direcotry):
    with open(f'{direcotry}/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=6, default=model_utils.convert_types)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    log_file = setup_logging(args.directory)
    config = load_config(args.config_path)
    dir_path = args.directory

    X, Y, sample_weight_dict = prepare_data(args.use_sample_weights)
    metrics = run_models(config, X, Y, sample_weight_dict)

    save_results(metrics, args.directory)