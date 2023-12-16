# Run example: python run.py 0 path/to/config.json path/to/results_directory

import sys
sys.path.insert(0, '../../')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import json
import argparse
import os

import LSTM as lstm_impl
import CNN_LSTM as cnn_lstm_impl
import utils as model_utils
import transformer as tr_impl

dir_path = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run models based on a configuration file.")
    parser.add_argument('gpu_id', type=str, choices=['0', '1'], help='GPU ID')
    parser.add_argument('config_path', type=str, help='Path to the configuration JSON file')
    parser.add_argument('directory', type=str, help='Path to result directory')
    args = parser.parse_args()
    return args


def test_models(labeling, data_type, X_train, X_val, X_test, Y_train, Y_val, Y_test, cnn_lstm=False, lstm=False, transformer=False):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

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


        # Build and train the model
        model = config['build_func'](_X_train.shape[-2], _X_train.shape[-1])
        early_stopping = EarlyStopping(monitor=model_utils.get_default_monitor_metric(), patience=config['patience'])
        model.fit(_X_train, Y_train, epochs=config['epochs'], validation_data=(_X_val, Y_val), 
                  batch_size=config['batch_size'], class_weight=class_weight_dict, callbacks=[early_stopping])

        # Evaluate the model
        train_eval = model.evaluate(_X_train, Y_train)
        val_eval = model.evaluate(_X_val, Y_val)
        test_eval = model.evaluate(_X_test, Y_test)

        # Save the model
        model.save(f"{dir_path}/saved_models/{data_type}-{labeling}-{model_name}.keras")

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


def prepare_data():
    # Load your raw data, features, and labels
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl()

    # Process raw data
    raw_X = model_utils.get_X(raw_data, 60)
    raw_Y_dict = {key: model_utils.get_Y(series, 60) for key, series in labels_dict.items()}

    # Process feature data
    feat_X = model_utils.get_X(features_df, 30)[30:]
    feat_Y_dict = {key: model_utils.get_Y(series, 30)[30:] for key, series in labels_dict.items()}

    # Initialize dictionaries to hold split data
    X = {'raw_data': {}, 'features': {}}
    Y = {'raw_data': {}, 'features': {}}

    # Split raw data
    X['raw_data']['train'], X['raw_data']['val'], raw_Y_train, raw_Y_val = train_test_split(raw_X, list(raw_Y_dict.values()), test_size=0.2, shuffle=False)
    X['raw_data']['val'], X['raw_data']['test'], raw_Y_val, raw_Y_test = train_test_split(X['raw_data']['val'], raw_Y_val, test_size=0.25, shuffle=False)

    # Split feature data
    X['features']['train'], X['features']['val'], feat_Y_train, feat_Y_val = train_test_split(feat_X, list(feat_Y_dict.values()), test_size=0.2, shuffle=False)
    X['features']['val'], X['features']['test'], feat_Y_val, feat_Y_test = train_test_split(X['features']['val'], feat_Y_val, test_size=0.25, shuffle=False)

    # Organize Y data into dictionaries
    for i, label in enumerate(labels_dict.keys()):
        Y['raw_data'][label] = {'train': raw_Y_train[i], 'val': raw_Y_val[i], 'test': raw_Y_test[i]}
        Y['features'][label] = {'train': feat_Y_train[i], 'val': feat_Y_val[i], 'test': feat_Y_test[i]}

    return X, Y


def run_models(config, X, Y):
    metrics = {}
    for labeling, data_types in config.items():
        for data_type, models in data_types.items():
            metrics[labeling] = metrics.get(labeling, {})
            metrics[labeling][data_type] = test_models(labeling, data_type,
                                                       X[data_type]['train'], X[data_type]['val'], X[data_type]['test'],
                                                       Y[data_type]['train'][labeling], Y[data_type]['val'][labeling], Y[data_type]['test'][labeling],
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
        json.dump(metrics, file, indent=6)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    log_file = setup_logging(args.directory)
    config = load_config(args.config_path)
    dir_path = args.directory

    X, Y = prepare_data()
    metrics = run_models(config, X, Y)

    save_results(metrics, args.directory)
