import sys
sys.path.insert(0, '../../')

import os
import json
import argparse
import numpy as np
import copy
import pandas as pd

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


def save_results(metrics, direcotry):
    with open(f'{direcotry}/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=3, default=model_utils.convert_types)


def test_model(data_type, label_name, weight_name, model_name,
                data, Xs, Ys, Ws, hp_dict,
                batch_size, epochs, 
                es_patience, n_splits,
                directory, window):
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

    print(f"Training {model_name} on {data_type} data with {label_name} labeling and {weight_name} weighting...", flush=True)
    mean_metrics, best_model = model_utils.custom_cross_val(hp_dict, Xs['train'], Ys['train'], Ws['train'],
                                                                    build_func,
                                                                    n_splits, epochs, batch_size,
                                                                    es_patience, directory,
                                                                    adjustedWeightsForEval=False,
                                                                    verbosity_level=1)
    print(f"Finished training {model_name} on {data_type} data with {label_name} labeling and {weight_name} weighting.", flush=True)

    # Save the model
    model_save_path = f"{directory}/saved_models/{data_type}-{label_name}-{weight_name}-{model_name}.keras"
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    best_model.save(model_save_path)

    # Save the mean metrics
    results = {}
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
    # Evaluate the model
    for stage in ['train', 'test']:
        name = f'best_model_{stage}'
        eval_results = best_model.evaluate(Xs[stage], Ys[stage], sample_weight=Ws[stage])
        save_backtest_plot_path = f"{directory}/backtests/{data_type}-{label_name}-{weight_name}-{model_name}-{stage}.html"
        results[name] = {
            "loss": eval_results[0],
            "accuracy": eval_results[1],
            "precision": eval_results[2],
            "recall": eval_results[3],
            "auc": eval_results[4],
            "cumulative_return": backtest_utils.do_backtest(data[stage], best_model, window, save_backtest_plot_path)['Return [%]'] / 100
        }

    return results


def run_models(config):
    metrics = {}
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
            for data_stage, data_path in data_paths.items():
                data = model_utils.load_data(data_path)
                Xs[data_stage] = model_utils.get_X(data, window_size)

            for label_name, label_paths in label_config.items():
                Ys = {}
                for label_stage, label_path in label_paths.items():
                    labels = model_utils.load_labels(label_path, label_name)
                    Ys[label_stage] = model_utils.get_Y(labels, window_size)

                for weight_name, weight_paths in weights_config.items():
                    Ws = {}
                    if weight_name == 'none':
                        Ws = {key: pd.Series(np.ones_like(Ys[key]), index=Ys[key].index) for key in Ys}
                    else:
                        for weight_stage, weight_path in weight_paths.items():
                            Ws[weight_stage] = model_utils.load_weights(weight_path, label_name, weight_name)[window_size:]

                    for model_name, model_params in model_config.items():
                        hp_dict = model_utils.load_hyperparameters(model_params['hyperparameters'])
                        
                        comb_name = f'{data_type}-{label_name}-{weight_name}-{model_name}'
                        metrics[comb_name] = test_model(
                            data_type, label_name, weight_name, model_name,
                            copy.deepcopy(data),
                            copy.deepcopy(Xs), copy.deepcopy(Ys), 
                            copy.deepcopy(Ws), copy.deepcopy(hp_dict),
                            model_params['batch_size'], model_params['epochs'], 
                            model_params['early_stopping_patience'], model_params['cv_splits'],
                            config['directory'], config['window_size']
                        )

                        # Save results
                        save_results(metrics, config['directory'])

    return metrics


if __name__ == '__main__':
    args = parse_args()
    config = model_utils.load_yaml_config(args.config_path)

    # Create directory for saving results
    if not os.path.exists(config['directory']):
        os.makedirs(config['directory'])

    # Set GPU ID based on config
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])

    # Run models based on the config
    metrics = run_models(config)