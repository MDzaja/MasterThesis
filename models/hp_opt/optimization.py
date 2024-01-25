import sys
sys.path.insert(0, '../../')

import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, VerboseCallback, EarlyStopper
from skopt import load
from memory_profiler import profile
import numpy as np
import json
import os
import gc
from functools import partial
import datetime

from models import utils as model_utils


def hp_opt_cv(build_model_gp, search_space, X, Y, W, directory, use_class_balancing, trial_num=100, initial_random_trials=10, 
              early_stopping_patience=20, epochs=100, batch_size=64, n_splits=5):
    
    try:
        with open(f'{directory}/metric_history.json', 'r') as file:
            metric_history = json.load(file)
    except FileNotFoundError:
        metric_history = []
    
    partial_objective = partial(objective_gp, X=X, Y=Y, W=W, use_class_balancing=use_class_balancing, build_model_gp=build_model_gp, 
                                epochs=epochs, batch_size=batch_size, n_splits=n_splits, early_stopping_patience=early_stopping_patience, 
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
    tracker = ProgressTracker(directory,
                              search_space=search_space,
                              n_total=trial_num,
                              n_init=len(x0) if x0 is not None else 0,
                              n_random=initial_random_trials,
                              )
    
    #early_stopper = CustomEarlyStopper(-0.8)

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
def objective_gp(params, X, Y, W, use_class_balancing, build_model_gp, epochs, batch_size, n_splits, early_stopping_patience, metric_history, directory):
    metrics, _ = model_utils.custom_cross_val(params, X, Y, W,
                                     build_model_gp,
                                     n_splits, epochs, batch_size,
                                     early_stopping_patience, directory,
                                     useWeightsForEval=True,
                                     useClassBalance=use_class_balancing,
                                     verbosity_level=0)

    metric_history.append(metrics)
    with open(f'{directory}/metric_history.json', 'w') as file:
        json.dump(metric_history, file, indent=4, default=model_utils.convert_types)

    return -metrics[model_utils.get_default_monitor_metric()]


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


def save_best_params(best_hyperparams, search_space, directory):
    best_hp_dict = {search_space[i].name: best_hyperparams[i] for i in range(len(best_hyperparams))}
    with open(f'{directory}/best_hp.json', 'w') as file:
        json.dump(best_hp_dict, file, indent=2, default=model_utils.convert_types)


class ProgressTracker(VerboseCallback):
    def __init__(self, directory, search_space, n_total, n_init=0, n_random=0):
        super(ProgressTracker, self).__init__(n_total=n_total, n_init=n_init, n_random=n_random)
        self.directory = directory
        self.search_space = search_space

    def __call__(self, result):
        # First call the base class's __call__, which increments the iteration number and prints to console
        super(ProgressTracker, self).__call__(result)
        # Then, add your custom file logging
        with open(f'{self.directory}/progress.txt', 'a') as file:
            current_trial = len(result.func_vals)
            best_score = result.fun
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"{current_datetime} - Trial {current_trial} finished: Best score = {best_score}\n")

        save_best_params(result.x, self.search_space, self.directory)

class CustomEarlyStopper(EarlyStopper):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def __call__(self, res):
        if res.fun < self.threshold:
            print(f"Stopping optimization: Objective function value {res.fun} is below the threshold {self.threshold}")
            return True
        return False