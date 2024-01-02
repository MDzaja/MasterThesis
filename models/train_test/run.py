import sys
sys.path.insert(0, '../../')

import os
import json
import argparse
import numpy as np

from models import LSTM as lstm_impl
from models import CNN_LSTM as cnn_lstm_impl
from models import transformer as tr_impl
from models import utils as model_utils


GET_X_Y_WINDOW_SIZE = 60


def parse_args():
    parser = argparse.ArgumentParser(description="Run models based on a configuration file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()
    return args


def save_results(metrics, direcotry):
    with open(f'{direcotry}/metrics.json', 'w') as file:
        json.dump(metrics, file, indent=3, default=model_utils.convert_types)


def test_model(data_type, label_name, weight_name, model_name, 
                Xs, Ys, Ws, hp_dict,
                batch_size, epochs, es_patience,
                direcotry):
    # Select the appropriate model building function based on model_name
    if model_name == 'cnn_lstm':
        build_func = cnn_lstm_impl.build_model
        # Additional reshaping for CNN-LSTM model
        n_length = 10
        n_steps = Xs['train'].shape[1] // n_length
        n_features = Xs['train'].shape[-1]
        for key in Xs:
            Xs[key] = Xs[key].reshape((Xs[key].shape[0], n_steps, n_length, n_features))
    elif model_name == 'lstm':
        build_func = lstm_impl.build_model
    elif model_name == 'transformer':
        build_func = tr_impl.build_model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Calculate class weights and adjust sample weights
    class_weight_dict = model_utils.calculate_class_weight_dict(Ys['train'])
    adjusted_sample_weights = {
        key: model_utils.adjust_sample_weights(Ys[key], class_weight_dict, Ws[key])
        for key in Ws
    }

    print(f"Training {model_name} on {data_type} data with {label_name} labeling and {weight_name} weighting...", flush=True)
    model = build_func(**hp_dict, Xs['train'].shape[-2], Xs['train'].shape[-1])
    model = model_utils.train_model(model, Xs['train'], Ys['train'], Xs['validation'], Ys['validation'], 
                                    adjusted_sample_weights['train'], adjusted_sample_weights['validation'],
                                    batch_size, epochs, es_patience, direcotry)
    print(f"Finished training {model_name} on {data_type} data with {label_name} labeling and {weight_name} weighting.", flush=True)

    # Save the model
    model_save_path = f"{direcotry}/saved_models/{data_type}-{label_name}-{weight_name}-{model_name}.keras"
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    model.save(model_save_path)

    # Evaluate the model
    results = {}
    for stage in ['train', 'validation', 'test']:
        eval_results = model.evaluate(Xs[stage], Ys[stage], sample_weight=Ws[stage])
        results[stage] = {
            "loss": eval_results[0],
            "accuracy": eval_results[1],
            "precision": eval_results[2],
            "recall": eval_results[3],
            "auc": eval_results[4]
        }

    return results


def run_models(config):
    metrics = {}

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
                Xs[data_stage] = model_utils.get_X(data, GET_X_Y_WINDOW_SIZE)

            for label_name, label_paths in label_config.items():
                Ys = {}
                for label_stage, label_path in label_paths.items():
                    labels = model_utils.load_labels(label_path, label_name)
                    Ys[label_stage] = model_utils.get_Y(labels, GET_X_Y_WINDOW_SIZE)

                for weight_name, weight_paths in weights_config.items():
                    Ws = {}
                    if weight_name == 'none':
                        Ws = {key: np.ones_like(Ys[key]) for key in Ys}
                    else:
                        for weight_stage, weight_path in weight_paths.items():
                            Ws[weight_stage] = model_utils.load_weights(weight_path, label_name, weight_name)

                    for model_name, model_params in model_config.items():
                        hp_dict = model_utils.load_hyperparameters(model_params['hyperparameters'])
                        
                        comb_name = f'{data_type}_{label_name}_{weight_name}_{model_name}'
                        metrics[comb_name] = test_model(
                            data_type, label_name, weight_name, model_name,
                            Xs, Ys, Ws, hp_dict,
                            model_params['batch_size'], model_params['epochs'], model_params['early_stopping_patience']
                        )

                        # Save results
                        save_results(metrics, config['directory'])

    return metrics


if __name__ == '__main__':
    args = parse_args()
    config = model_utils.load_config(args.config_path)

    # Set GPU ID based on config
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])

    # Run models based on the config
    metrics = run_models(config)