import sys
sys.path.insert(0, '../../')

from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import BayesianOptimization
import numpy as np
import json

from models import utils as model_utils


def hp_opt(build_model_func, X_train, Y_train, X_val, Y_val, directory, project_name, max_trials=100, executions_per_trial=1, early_stopping_patience=20, epochs=100, batch_size=64):
    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor=model_utils.get_default_monitor_metric(), patience=early_stopping_patience),

    # Bayesian Optimization tuner
    tuner = BayesianOptimization(
        lambda hp: build_model_func(hp, X_train.shape[-2], X_train.shape[-1]),
        objective=model_utils.get_default_monitor_metric(),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name
    )

    # Search for the best hyperparameters
    tuner.search(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[early_stopping])

    # Analyze hyperparameters
    save_hyperparameter_analysis(directory, tuner)

    # Analyze overall trials
    save_best_n_worst_trials(directory, tuner)


def save_hyperparameter_analysis(directory, tuner):
    hp_stats = {}
    for trial_id, trial in tuner.oracle.trials.items():
        for hp, value in trial.hyperparameters.values.items():
            if hp not in hp_stats:
                hp_stats[hp] = {'values': [], 'type': 'numerical' if isinstance(value, (int, float)) else 'categorical'}
            hp_stats[hp]['values'].append(value)

    for hp in hp_stats.keys():
        if hp_stats[hp]['type'] == 'numerical':
            values = np.array(hp_stats[hp]['values'])
            hp_stats[hp].update({
                'average': np.mean(values),
                'median': np.median(values),
                'max': np.max(values),
                'min': np.min(values),
                '90th_percentile': np.percentile(values, 90)
            })
            hp_stats[hp].pop('values')
        else:
            hp_stats[hp]['count'] = {val: hp_stats[hp]['values'].count(val) for val in set(hp_stats[hp]['values'])}

    with open(f'{directory}/hp_analysis.json', 'w') as f:
        json.dump(hp_stats, f, default=model_utils.convert_types, indent=4)


def save_best_n_worst_trials(directory, tuner):
    non_null_trials = [t for t in tuner.oracle.trials.values() if t.score is not None]
    sorted_trials = sorted_trials = sorted(non_null_trials, key=lambda t: t.score)
    best_trial = sorted_trials[0]
    worst_trial = sorted_trials[-1]

    overall_info = {
        'best_trial': {
            'trial_id': best_trial.trial_id,
            'hyperparameters': best_trial.hyperparameters.values,
            'score': best_trial.score,
            'metrics': extract_trial_metrics(best_trial)
        },
        'worst_trial': {
            'trial_id': worst_trial.trial_id,
            'hyperparameters': worst_trial.hyperparameters.values,
            'score': worst_trial.score,
            'metrics': extract_trial_metrics(worst_trial)
        }
    }

    with open(f'{directory}/overall_info.json', 'w') as f:
        json.dump(overall_info, f, default=model_utils.convert_types, indent=4)


def extract_trial_metrics(trial):
    # Extract all available metrics for a given trial
    metrics = {}
    for metric_name in trial.metrics.metrics.keys():
        metric_history = trial.metrics.get_history(metric_name)
        if metric_history:
            # Assuming the last entry in the metric history contains the metric value
            last_metric_observation = metric_history[-1]
            #if hasattr(last_metric_observation, 'value'):
                # If the MetricObservation has a 'value' attribute
            #    metrics[metric_name] = last_metric_observation.value
            #elif isinstance(last_metric_observation, dict) and 'value' in last_metric_observation:
                # If the MetricObservation is a dictionary with a 'value' key
            # TODO remove if unnecessary
            metrics[metric_name] = last_metric_observation['value']
    return metrics