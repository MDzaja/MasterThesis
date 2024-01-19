import sys
sys.path.insert(0, '../../')

import os
import json
import argparse
import numpy as np
import copy
import pandas as pd
import datetime
from tensorflow.keras.models import load_model

from models import LSTM as lstm_impl
from models import CNN_LSTM as cnn_lstm_impl
from models import transformer as tr_impl
from models import utils as model_utils
from backtest import utils as backtest_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run models based on a configuration file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()
    return args


def save_results(metrics, train_probs, test_probs, directory):
    with open(f'{directory}/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=3, default=model_utils.convert_types)
    pd.to_pickle(train_probs, f'{directory}/train_probs.pkl')
    pd.to_pickle(test_probs, f'{directory}/test_probs.pkl')


def test_model(data_type, label_name, weight_name, model_name,
                data, Xs, Ys, Ws, hp_dict,
                use_class_balancing,
                batch_size, epochs, 
                es_patience, n_splits,
                directory, window):
    results = {}
    # Select the appropriate model building function based on model_name
    if model_name == 'cnn_lstm':
        build_func = cnn_lstm_impl.build_model
        # Additional reshaping for CNN-LSTM model
        for key in Xs:
            n_length = 10
            n_steps = Xs[key].shape[1] // n_length
            n_features = Xs[key].shape[-1]
            Xs[key] = Xs[key].reshape((Xs[key].shape[0], n_steps, n_length, n_features))
    elif model_name == 'lstm':
        build_func = lstm_impl.build_model
    elif model_name == 'transformer':
        build_func = tr_impl.build_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model_save_path = f"{directory}/saved_models/{data_type}-{label_name}-{'CB_' if use_class_balancing else ''}{weight_name}-{model_name}.keras"
    if os.path.exists(model_save_path):
        best_model = load_model(model_save_path)
        best_model.compile()
        print(f"Loaded {model_name} on {data_type} data with {label_name} labeling and {'CB_' if use_class_balancing else ''}{weight_name} weighting.", flush=True)
    else:
        print(f"Training {model_name} on {data_type} data with {label_name} labeling and {'CB_' if use_class_balancing else ''}{weight_name} weighting...", flush=True)
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_datetime}", flush=True)
        mean_metrics, best_model = model_utils.custom_cross_val(hp_dict, Xs['train'], Ys['train'], Ws['train'],
                                                                        build_func,
                                                                        n_splits, epochs, batch_size,
                                                                        es_patience, directory,
                                                                        useWeightsForEval=False,
                                                                        useClassBalance=use_class_balancing,
                                                                        verbosity_level=0)
        print(f"Finished training {model_name} on {data_type} data with {label_name} labeling and {'CB_' if use_class_balancing else ''}{weight_name} weighting.", flush=True)
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_datetime}", flush=True)

        # Save the model
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        best_model.save(model_save_path)

        # Save the mean metrics
        for stage in ['train', 'validation']:
            name = f'{stage}_cv_mean'
            metric_prefix = 'val_' if stage == 'validation' else 'train_'
            results[name] = {
                "loss": mean_metrics[f'{metric_prefix}loss'],
                "accuracy": mean_metrics[f'{metric_prefix}binary_accuracy'],
                "precision": mean_metrics[f'{metric_prefix}precision'],
                "recall": mean_metrics[f'{metric_prefix}recall'],
                "auc": mean_metrics[f'{metric_prefix}auc']
            }

    if not os.path.exists(f'{directory}/backtests'):
        os.makedirs(f'{directory}/backtests')

    # Evaluate the model
    probs_dict = {}
    for stage in ['train', 'test']:
        name = f'best_model_{stage}'
        eval_results = best_model.evaluate(Xs[stage], Ys[stage], verbose=0)
        save_backtest_plot_path = f"{directory}/backtests/{data_type}-{label_name}-{'CB_' if use_class_balancing else ''}{weight_name}-{model_name}-{stage}.html"

        probs_dict[stage] = {
            'label_name': label_name,
        }
        probs_arr = best_model.predict(Xs[stage], verbose=0).flatten()
        probs_s = pd.Series(probs_arr, index=Ys[stage].index)
        combined_index = probs_s.index.union(data[stage].index)
        probs_s = probs_s.reindex(combined_index, fill_value=0)
        probs_dict[stage]['probs'] = probs_s
        bt_result, threshold = backtest_utils.do_backtest_w_optimization(data[stage], probs_dict[stage]['probs'], save_backtest_plot_path)

        results[name] = {
            "loss": eval_results[0],
            "accuracy": eval_results[1],
            "precision": eval_results[2],
            "recall": eval_results[3],
            "auc": eval_results[4],
            "cumulative_return": bt_result['Return [%]'] / 100,
            "threshold": threshold
        }

    return results, probs_dict['train'], probs_dict['test']


def run_models(config):
    metrics = {}
    if os.path.exists(f'{config["directory"]}/metrics.json'):
        with open(f'{config["directory"]}/metrics.json', 'r') as file:
            metrics = json.load(file)
    probs_train = {}
    if os.path.exists(f'{config["directory"]}/train_probs.pkl'):
        probs_train = pd.read_pickle(f'{config["directory"]}/train_probs.pkl')
    probs_test = {}
    if os.path.exists(f'{config["directory"]}/test_probs.pkl'):
        probs_test = pd.read_pickle(f'{config["directory"]}/test_probs.pkl')
    window_size = config['window_size']

    for combination in config['combinations']:
        data_config = combination['data']
        label_config = combination['labels']
        weights_config = combination['weights']
        model_config = combination['model']

        # Handling 'all' in label and weight configurations
        if 'all' in label_config:
            all_label_names = model_utils.get_all_label_names()
            all_labels = {ln: label_config['all'] for ln in all_label_names}
            label_config = {**all_labels, **{k: v for k, v in label_config.items() if k != 'all'}}

        if 'all' in weights_config:
            all_weight_names = model_utils.get_all_weight_names()
            all_weights = {wn: weights_config['all'] for wn in all_weight_names}
            weights_config = {**all_weights, **{k: v for k, v in weights_config.items() if k != 'all'}}

        for data_type, data_paths in data_config.items():
            Xs = {}
            data = {}
            for data_stage, data_path in data_paths.items():
                data[data_stage] = model_utils.load_data(data_path)
                Xs[data_stage] = model_utils.get_X_day_separated(data[data_stage], window_size, data['train'])

            for label_name, label_paths in label_config.items():
                Ys = {}
                for label_stage, label_path in label_paths.items():
                    labels = model_utils.load_labels(label_path, label_name)
                    Ys[label_stage] = model_utils.get_Y_or_W_day_separated(labels, window_size)

                for weight_name, weight_props in weights_config.items():
                    if weight_props is not None and 'class_balance' in weight_props:
                        use_class_balancing = True
                    else:
                        use_class_balancing = False
                        
                    Ws = {}
                    if weight_name == 'none':
                        Ws = {key: pd.Series(np.ones_like(Ys[key]), index=Ys[key].index) for key in Ys}
                    else:
                        for weight_stage, weight_path in weight_props.items():
                            if weight_stage == 'class_balance':
                                continue
                            weights = model_utils.load_weights(weight_path, label_name, weight_name)
                            Ws[weight_stage] = model_utils.get_Y_or_W_day_separated(weights, window_size)

                    for model_name, model_params in model_config.items():
                        hp_dict = model_utils.load_hyperparameters(model_params['hyperparameters'])
                        
                        comb_name = f'D-{data_type};L-{label_name};W-{"CB_" if use_class_balancing else ""}{weight_name};M-{model_name}'
                        if comb_name in metrics.keys():
                            print(f"Skipping {comb_name} because it has already been processed.", flush=True)
                            continue

                        metrics[comb_name], probs_train[comb_name], probs_test[comb_name] = test_model(
                            data_type, label_name, weight_name, model_name,
                            copy.deepcopy(data),
                            copy.deepcopy(Xs), copy.deepcopy(Ys),
                            copy.deepcopy(Ws), copy.deepcopy(hp_dict),
                            use_class_balancing,
                            model_params['batch_size'], model_params['epochs'],
                            model_params['early_stopping_patience'], model_params['cv_splits'],
                            config['directory'], config['window_size']
                        )

                        # Save results
                        save_results(metrics, probs_train, probs_test, config['directory'])

    return metrics


def setup_logging(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = open(f'{directory}/output.log', 'a')
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


if __name__ == '__main__':
    args = parse_args()
    config = model_utils.load_yaml_config(args.config_path)

    # Create directory for saving results
    if not os.path.exists(config['directory']):
        os.makedirs(config['directory'])

    # Set GPU ID based on config
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])
    setup_logging(config['directory'])

    # Run models based on the config
    metrics = run_models(config)