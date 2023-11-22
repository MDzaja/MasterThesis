from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import json
import pandas as pd

sys.path.append('../')
from label_algorithms import oracle

def get_ft_n_Y(window_size=60):
    features_df = pd.read_csv('../features/test_features.csv', index_col=0)

    start_date = features_df.index[0]
    end_date = pd.to_datetime(features_df.index[-1]) + pd.Timedelta(days=1)
    close = yf.download('GC=F', start_date, end_date, interval='1d')['Close']
    fee = 0.0004
    labels = oracle.binary_trend_labels(close, fee=fee)

    X = get_X(features_df, window_size)[:-1]
    Y = get_Y(labels, window_size)

    return X, Y



def get_X(data, window_size):
    # Normalize features to range between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Create the 3D input data shape [samples, time_steps, features]
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, :])

    return np.array(X)


def get_Y(labels: pd.Series, window_size):
    return labels.shift(-1)[window_size:-1].astype(int).reshape(-1, 1)


def get_dummy_X_n_Y(window_size=60):
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    data.index = data.index.tz_localize(None)

    fee = 0.0004
    labels = oracle.binary_trend_labels(data['Close'], fee=fee)

    X = get_X(data, window_size)[:-1]
    Y = get_Y(labels, window_size)
    
    return X, Y

def train_model(build_model_func, X_train, Y_train, X_val, Y_val, early_stopping_patience=20, epochs=100):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    model = build_model_func(X_train.shape[-2], X_train.shape[-1])

    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), batch_size=64, class_weight=class_weight_dict, callbacks=[early_stopping])


def hyperparameter_optimization(build_model_func, X_train, Y_train, X_val, Y_val, directory, project_name, max_trials=100, executions_per_trial=1, early_stopping_patience=20, epochs=100, batch_size=64):
    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    # Bayesian Optimization tuner
    tuner = BayesianOptimization(
        lambda hp: build_model_func(hp, X_train.shape[-2], X_train.shape[-1]),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name
    )

    # Search for the best hyperparameters
    tuner.search(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), callbacks=[early_stopping])

    # Analyze hyperparameters
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
        json.dump(hp_stats, f, default=convert_types, indent=4)

    # Analyze overall trials
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
        json.dump(overall_info, f, default=convert_types, indent=4)


def extract_trial_metrics(trial):
    # Extract all available metrics for a given trial
    metrics = {}
    for metric_name in trial.metrics.metrics.keys():
        metric_history = trial.metrics.get_history(metric_name)
        if metric_history:
            # Assuming the last entry in the metric history contains the metric value
            last_metric_observation = metric_history[-1]
            if hasattr(last_metric_observation, 'value'):
                # If the MetricObservation has a 'value' attribute
                metrics[metric_name] = last_metric_observation.value
            elif isinstance(last_metric_observation, dict) and 'value' in last_metric_observation:
                # If the MetricObservation is a dictionary with a 'value' key
                metrics[metric_name] = last_metric_observation['value']
    return metrics


# Convert NumPy types to Python types for JSON serialization
def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
