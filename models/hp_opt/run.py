import sys
sys.path.insert(0, '../../')

import os
import argparse
import numpy as np
import pandas as pd

from models import CNN_LSTM as cnn_lstm
from models import LSTM as lstm
from models import transformer as tr
from models import utils as model_utils
from models.hp_opt import optimization as cv_opt
from models.xgboost import hp_opt as xgb_opt


def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization based on a configuration file.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()
    return args

def setup_logging(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = open(f'{directory}/output.log', 'a')
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file

def set_logging_back_to_normal():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def main(config):
    window_size = config['window_size']
    use_class_balancing = False

    for combination in config['combinations']:
        label_config = combination['labels']
        weights_config = combination['weights']
        data_config = combination['data']
        model_config = combination['model']

        # Handling 'all' in label and weight configurations
        if 'all' in label_config:
            all_label_names = model_utils.get_all_label_names()
            all_labels = {ln: label_config['all'] for ln in all_label_names}
            label_config = {**all_labels, **{k: v for k, v in label_config.items() if k != 'all'}}
        
        # TODO - Implement 'all' in weights_config, some changes to the code are required
        if 'all' in weights_config:
            all_weight_names = model_utils.get_all_weight_names()
            all_weights = {wn: weights_config['all'] for wn in all_weight_names}
            weights_config = {**all_weights, **{k: v for k, v in weights_config.items() if k != 'all'}}

        for data_type, data_props in data_config.items():
            # Load data
            data = model_utils.load_data(data_props['path'])
            if 'drop_duplicates' in data_props:
                data.drop_duplicates(inplace=True)
            if 'drop_zero_volume' in data_props:
                data = data[data['Volume'] != 0]

            X = model_utils.get_X_day_separated(data, window_size)

            for label_name, label_path in label_config.items():
                # Load labels
                labels = model_utils.load_labels(label_path['path'], label_name)
                Y = model_utils.get_Y_or_W_day_separated(labels, window_size)

                for weight_name, weight_props in weights_config.items():
                    if weight_props is not None and 'class_balance' in weight_props:
                        use_class_balancing = True
                    # Load weights
                    if weight_name == 'none':
                        W = pd.Series(np.ones(len(Y)), index=Y.index)
                    else:
                        weights = model_utils.load_weights(weight_props['path'], label_name, weight_name)
                        W = model_utils.get_Y_or_W_day_separated(weights, window_size)

                    for model_name, model_params in model_config.items():
                        # Determine the model and hyperparameter space
                        if model_name == 'cnn_lstm':
                            build_model_gp, search_space = cnn_lstm.build_model_gp, cnn_lstm.define_search_space()
                            n_length = 10
                            n_steps = X.shape[1] // n_length
                            n_features = X.shape[-1]
                            X = X.reshape((X.shape[0], n_steps, n_length, n_features))
                        elif model_name == 'lstm':
                            build_model_gp, search_space = lstm.build_model_gp, lstm.define_search_space()
                        elif model_name == 'transformer':
                            build_model_gp, search_space = tr.build_model_gp, tr.define_search_space()
                        elif model_name == 'xgboost':
                            pass
                        else:
                            raise ValueError(f"Unknown model name: {model_name}")

                        # Construct and create the combination-specific directory
                        combination_dir = f"{config['directory']}/{data_type}-{label_name}-{'CB_' if use_class_balancing else ''}{weight_name}-{model_name}"
                        if not os.path.exists(combination_dir):
                            os.makedirs(combination_dir)
                        log_file = setup_logging(combination_dir)

                        finished_file = os.path.join(combination_dir, '.finished')

                        # Check if the .finished file exists
                        if os.path.exists(finished_file):
                            print(f"Skipping optimization for {combination_dir} as it's already completed.")
                            continue
                        
                        if model_name == 'xgboost':
                            xgb_opt.xgb_hp_opt_cv(
                                X, Y, W, combination_dir, use_class_balancing,
                                trial_num=model_params['trial_num'],
                                n_splits=model_params['cv_splits']
                            )
                        else:
                            # Hyperparameter Optimization with Cross-Validation
                            cv_opt.hp_opt_cv(
                                build_model_gp, search_space, X, Y, W, 
                                combination_dir,
                                use_class_balancing=use_class_balancing,
                                trial_num=model_params['trial_num'],
                                initial_random_trials=model_params['initial_random_trials'],
                                early_stopping_patience=model_params['early_stopping_patience'],
                                epochs=model_params['epochs'],
                                batch_size=model_params['batch_size'],
                                n_splits=model_params['cv_splits']
                            )

                        # Create a .finished file after completion
                        with open(finished_file, 'w') as file:
                            file.write('Optimization completed.')
                        print(f"Optimization completed for {combination_dir}.")
                        log_file.close()
                        set_logging_back_to_normal()

if __name__ == '__main__':
    args = parse_args()
    config = model_utils.load_yaml_config(args.config_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_id'])

    main(config)