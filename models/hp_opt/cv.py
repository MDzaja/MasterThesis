import sys
sys.path.insert(0, '../../')

import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import load
from memory_profiler import profile
import numpy as np
import json
import os
import gc
from functools import partial

from models import utils as model_utils


def hp_opt_cv(build_model_gp, search_space, X, Y, W, directory, trial_num=100, initial_random_trials=10, 
              early_stopping_patience=20, epochs=100, batch_size=64, n_splits=5):
    
    try:
        with open(f'{directory}/metric_history.json', 'r') as file:
            metric_history = json.load(file)
    except FileNotFoundError:
        metric_history = []
    
    partial_objective = partial(objective_gp, X=X, Y=Y, W=W, build_model_gp=build_model_gp, epochs=epochs, 
                                batch_size=batch_size, n_splits=n_splits, early_stopping_patience=early_stopping_patience, 
                                metric_history=metric_history, directory=directory)

    # Load previous state if it exists
    checkpoint_path  = f'{directory}/optimization_state.pkl'
    checkpoint_saver  = CheckpointSaver(checkpoint_path)
    try:
        res = load(checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals
    except FileNotFoundError:
        x0 = y0 = None

    # Create a ProgressTracker callback
    tracker = ProgressTracker(f'{directory}/progress.txt',
                              n_total=trial_num,
                              n_init=len(x0) if x0 is not None else 0,
                              n_random=initial_random_trials)

    np.int = np.int64
    result = gp_minimize(func=partial_objective, dimensions=search_space,
                         n_calls=max(trial_num - len(x0) if x0 is not None else trial_num, 1),
                         n_initial_points=max(initial_random_trials - len(x0) if x0 is not None else trial_num, 0),
                         x0=x0, y0=y0, callback=[tracker, checkpoint_saver], verbose=True)

    # Process results for hyperparameter analysis
    hyperparam_analysis = process_hyperparams(result, search_space)
    with open(f'{directory}/hp_analysis.json', 'w') as file:
        json.dump(hyperparam_analysis, file, indent=4, default=model_utils.convert_types)

    # Process results for best and worst trials
    overall_info = process_trials(result, search_space, metric_history)
    with open(f'{directory}/overall_info.json', 'w') as file:
        json.dump(overall_info, file, indent=4, default=model_utils.convert_types)

    # Save the best hyperparameters
    best_hyperparams = result.x
    best_hp_dict = {search_space[i].name: best_hyperparams[i] for i in range(len(best_hyperparams))}
    with open(f'{directory}/best_hp.json', 'w') as file:
        json.dump(best_hp_dict, file, indent=2, default=model_utils.convert_types)


@profile
def objective_gp(params, X, Y, W, build_model_gp, epochs, batch_size, n_splits, early_stopping_patience, metric_history, directory):
    metrics = custom_cross_val_score(params, X, Y, W, 
                                     build_model_gp, 
                                     n_splits, epochs, batch_size, 
                                     early_stopping_patience, directory)

    metric_history.append(metrics)
    with open(f'{directory}/metric_history.json', 'w') as file:
        json.dump(metric_history, file, indent=4, default=model_utils.convert_types)

    return -metrics[model_utils.get_default_monitor_metric()]


@profile
def custom_cross_val_score(params, X, Y, W, build_model_gp, n_splits, epochs, batch_size, early_stopping_patience, directory):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = {}

    memory_log_path = os.path.join(directory, "memory_usage.log")#TODO

    for train_index, val_index in tscv.split(X):

        mem_before = model_utils.get_memory_usage()#TODO
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage before training split: {mem_before:.2f} MB\n")

        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        W_train, W_val = W[train_index], W[val_index]

        class_weight_dict = model_utils.calculate_class_weight_dict(Y_train)
        adjusted_W_train = model_utils.adjust_sample_weights(Y_train, class_weight_dict, W_train)
        adjusted_W_val = model_utils.adjust_sample_weights(Y_val, class_weight_dict, W_val)


        model = build_model_gp(params, X_train.shape[-2], X_train.shape[-1])
        try:
            model = model_utils.train_model(model, X_train, Y_train, X_val, Y_val, 
                                        adjusted_W_train, adjusted_W_val,
                                        batch_size, epochs, early_stopping_patience, directory)
        except tf.errors.InternalError or tf.errors.ResourceExhaustedError as e:
            print(f"Error: {e}")
            model_utils.restart_script()

        # Evaluate the model
        train_metrics = model.evaluate(X_train, Y_train, sample_weight=adjusted_W_train, verbose=0)
        val_metrics = model.evaluate(X_val, Y_val, sample_weight=adjusted_W_val, verbose=0)

        # Update the metrics dictionary
        for i, metric_name in enumerate(model.metrics_names):
            all_metrics.setdefault(f'train_{metric_name}', []).append(train_metrics[i])
            all_metrics.setdefault(f'val_{metric_name}', []).append(val_metrics[i])

        # Clear the model and Keras session to free memory
        del model, X_train, X_val, Y_train, Y_val, train_dataset, val_dataset
        tf.keras.backend.clear_session()
        gc.collect()

        mem_after = model_utils.get_memory_usage() #TODO
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage after training split: {mem_after:.2f} MB\n")
            mem_log.write(f"Memory usage difference: {mem_after - mem_before:.2f} MB\n\n")

    # Calculate and return mean metrics
    mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    return mean_metrics


def process_hyperparams(result, search_space):
    hp_stats = {}
    for i, dimension in enumerate(search_space):
        values = [x[i] for x in result.x_iters]
        hp_stats[dimension.name] = {
            'average': np.mean(values),
            'median': np.median(values),
            'max': np.max(values),
            'min': np.min(values),
            '90th_percentile': np.percentile(values, 90)
        }
    return hp_stats


def process_trials(result, search_space, metric_history):
    # Finding indices of the best and worst trials
    best_trial_index = np.argmin(result.func_vals)
    worst_trial_index = np.argmax(result.func_vals)

    # Metrics for the best trial
    best_trial_metrics = metric_history[best_trial_index]
    best_trial_params = {search_space[i].name: result.x[i] for i in range(len(search_space))}
    best_trial = {
        'trial_num': best_trial_index + 1,
        'hyperparameters': best_trial_params,
        'score': -result.fun,
        'metrics': best_trial_metrics
    }

    # Metrics for the worst trial
    worst_score = result.func_vals[worst_trial_index]
    worst_trial_metrics = metric_history[worst_trial_index]
    worst_trial_params = {search_space[i].name: result.x_iters[worst_trial_index][i] for i in range(len(search_space))}
    worst_trial = {
        'trial_num': worst_trial_index + 1,
        'hyperparameters': worst_trial_params,
        'score': -worst_score,
        'metrics': worst_trial_metrics
    }

    return {
        'best_trial': best_trial,
        'worst_trial': worst_trial
    }


class ProgressTracker(VerboseCallback):
    def __init__(self, log_file, n_total, n_init=0, n_random=0):
        super(ProgressTracker, self).__init__(n_total=n_total, n_init=n_init, n_random=n_random)
        self.log_file = log_file

    def __call__(self, result):
        # First call the base class's __call__, which increments the iteration number and prints to console
        super(ProgressTracker, self).__call__(result)
        # Then, add your custom file logging
        with open(self.log_file, 'a') as file:
            current_trial = len(result.func_vals)
            best_score = result.fun
            file.write(f"Trial {current_trial} finished: Best score = {best_score}\n")
