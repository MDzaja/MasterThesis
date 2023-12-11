from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import BayesianOptimization, Objective
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from functools import partial
from skopt import gp_minimize

import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
import pickle
import os

sys.path.append('../')
from label_algorithms import oracle

def get_ft_n_Y(window_size=60):
    features_df = pd.read_csv('../features/test_features.csv', index_col=0)

    start_date = features_df.index[0]
    end_date = pd.to_datetime(features_df.index[-1]) + pd.Timedelta(days=1)
    close = yf.download('GC=F', start_date, end_date, interval='1d')['Close']
    fee = 0.0004
    labels = oracle.binary_trend_labels(close, fee=fee)

    X = get_X(features_df, window_size)
    Y = get_Y(labels, window_size)

    return X, Y


def get_aligned_raw_feat_lbl():
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'
    raw_data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    raw_data.index = raw_data.index.tz_localize(None)

    features_df = pd.read_csv('../features/test_features.csv', index_col=0)

    with open('../label_algorithms/labels_dict.pkl', 'rb') as file:
        labels_dict = pickle.load(file)

    # Ensure indices are in the same format
    raw_data.index = pd.to_datetime(raw_data.index)
    features_df.index = pd.to_datetime(features_df.index)
    labels_dict = {k: pd.Series(v, index=pd.to_datetime(v.index)) for k, v in labels_dict.items()}

    # Find the common indices
    common_indices = raw_data.index.intersection(features_df.index)
    for label_series in labels_dict.values():
        common_indices = common_indices.intersection(label_series.index)

    # Reindex raw_data, features_df, and each Series in labels_dict
    raw_data = raw_data.reindex(common_indices)
    features_df = features_df.reindex(common_indices)
    labels_dict = {key: series.reindex(common_indices) for key, series in labels_dict.items()}

    return raw_data, features_df, labels_dict


def get_X(data, window_size):
    # Normalize features to range between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    # Create the 3D input data shape [samples, time_steps, features]
    X = []

    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, :])

    # Convert to a 3D NumPy array
    return np.array(X)


def get_Y(labels: pd.Series, window_size) -> pd.Series:
    return labels[window_size:]


def get_dummy_X_n_Y(window_size=60):
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'

    data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    data.index = data.index.tz_localize(None)

    fee = 0.0004
    labels = oracle.binary_trend_labels(data['Close'], fee=fee)

    X = get_X(data, window_size)
    Y = get_Y(labels, window_size)
    
    return X, Y

def train_model(build_model_func, X_train, Y_train, X_val, Y_val, early_stopping_patience=20, epochs=100):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    early_stopping = EarlyStopping(monitor=get_default_monitor_metric(), patience=early_stopping_patience)

    model = build_model_func(X_train.shape[-2], X_train.shape[-1])

    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), batch_size=64, class_weight=class_weight_dict, callbacks=[early_stopping])


def hyperparameter_optimization(build_model_func, X_train, Y_train, X_val, Y_val, directory, project_name, max_trials=100, executions_per_trial=1, early_stopping_patience=20, epochs=100, batch_size=64):
    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor=get_default_monitor_metric(), patience=early_stopping_patience),

    # Bayesian Optimization tuner
    tuner = BayesianOptimization(
        lambda hp: build_model_func(hp, X_train.shape[-2], X_train.shape[-1]),
        objective=get_default_monitor_metric(),
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
        json.dump(hp_stats, f, default=convert_types, indent=4)


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
        json.dump(overall_info, f, default=convert_types, indent=4)


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


# Convert NumPy types to Python types for JSON serialization
def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def plot_acc_auc(file_path):
    # Load the JSON data into a dictionary
    with open(file_path, 'r') as file:
        metrics_dict = json.load(file)

    # Flatten the JSON data into a DataFrame directly
    def flatten_data(data, path=None, accumulator=None):
        if accumulator is None:
            accumulator = []
        if path is None:
            path = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                flatten_data(value, path + [key], accumulator)
        else:
            accumulator.append(path + [data])
        return accumulator

    flattened_data = flatten_data(metrics_dict)
    columns = ['features_used', 'labeling_algorithm', 'model_name', 'data_name', 'metric_name', 'value']
    metrics_df = pd.DataFrame(flattened_data, columns=columns)

    # Filter for accuracy and AUC metrics separately
    accuracy_df = metrics_df[metrics_df['metric_name'] == 'accuracy'].copy()
    auc_df = metrics_df[metrics_df['metric_name'] == 'auc'].copy()

    # Rename the 'value' column to match the metric they represent
    accuracy_df.rename(columns={'value': 'accuracy'}, inplace=True)
    auc_df.rename(columns={'value': 'auc'}, inplace=True)

    # Drop the 'metric_name' column as it is no longer needed
    accuracy_df.drop('metric_name', axis=1, inplace=True)
    auc_df.drop('metric_name', axis=1, inplace=True)

    # Merge the two DataFrames on the categorical columns
    merged_df = pd.merge(
        accuracy_df, 
        auc_df, 
        on=['features_used', 'labeling_algorithm', 'model_name', 'data_name'],
        how='outer'
    )
    merged_df['lbl_model'] = merged_df['labeling_algorithm'] + ' - ' + merged_df['model_name']

    # Introduce jitter to the accuracy and auc values
    merged_df['accuracy'] = merged_df['accuracy'] + np.random.uniform(-0.01, 0.01, size=len(merged_df))
    merged_df['auc'] = merged_df['auc'] + np.random.uniform(-0.01, 0.01, size=len(merged_df))

    # Define the layout of your subplots
    n_rows, n_cols = 2, 3
    f, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), sharex=False, sharey=False)

    # Plot each group
    for i, ((features, data_name), group) in enumerate(merged_df.groupby(['features_used', 'data_name'])):
        row_index = 0 if features == 'raw_data' else 1
        col_index = ['train', 'validation', 'test'].index(data_name)
        
        ax = sns.scatterplot(data=group, x='accuracy', y='auc', hue='lbl_model', style='lbl_model', ax=axes[row_index, col_index])
        ax.set_title(f'{features} - {data_name}')
        ax.legend().set_visible(False)

    # Create an external legend from the last plot
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.05))

    plt.show()

    merged_df = merged_df.drop(columns=['lbl_model'])	
    return merged_df

def get_default_metrics() -> list:
    return [BinaryAccuracy(name='binary_accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')]

def get_default_monitor_metric() -> str:
    return 'val_auc'

def get_default_loss() -> str:
    return 'binary_crossentropy'

def hp_opt_cv(build_model_gp, search_space, X, Y, directory, trial_num=100, initial_random_trials=10, 
              early_stopping_patience=20, epochs=100, batch_size=64, n_splits=5):
    
    partial_objective = partial(objective_gp, X=X, Y=Y, build_model_gp=build_model_gp, epochs=epochs, 
                                batch_size=batch_size, n_splits=n_splits, monitor_metric=get_default_monitor_metric(),
                                early_stopping_patience=early_stopping_patience)

    np.int = np.int64
    result = gp_minimize(func=partial_objective, dimensions=search_space, 
                         n_calls=trial_num, n_initial_points=initial_random_trials, n_jobs=-1, verbose=True)

    # Process results for hyperparameter analysis
    hyperparam_analysis = process_hyperparams(result, search_space)
    with open(f'{directory}/hp_analysis.json', 'w') as file:
        json.dump(hyperparam_analysis, file, indent=4, default=convert_types)

    # Process results for best and worst trials
    overall_info = process_trials(result, search_space)
    with open(f'{directory}/overall_info.json', 'w') as file:
        json.dump(overall_info, file, indent=4, default=convert_types)

    # Best hyperparameters
    print("Best parameters: {}".format(result.x))


def objective_gp(params, X, Y, build_model_gp, epochs, batch_size, n_splits, monitor_metric, early_stopping_patience):
    early_stopping = EarlyStopping(
        monitor=monitor_metric, 
        patience=early_stopping_patience,
        mode='max'
    )
    model_fn = lambda: build_model_gp(params, X.shape[-2], X.shape[-1])

    score = custom_cross_val_score(model_fn, X, Y, n_splits, epochs, batch_size, early_stopping)
    return -score


def custom_cross_val_score(model_fn, X, Y, n_splits, epochs, batch_size, early_stopping):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        # Compute class weights
        classes = np.unique(Y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
        class_weight_dict = dict(zip(classes, class_weights))

        # Configure ModelCheckpoint
        checkpoint = ModelCheckpoint(
            filepath='tmp/tmp_model.keras',
            monitor=get_default_monitor_metric(),
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )

        model = model_fn()
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs, batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, checkpoint]
        )

        # Load the best weights
        model.load_weights('tmp/tmp_model.keras')
        os.remove('tmp/tmp_model.keras')

        val_scores = model.evaluate(X_val, Y_val)
        scores.append(val_scores[-1])

    return np.mean(val_scores)


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


def process_trials(result, search_space):
    # Mapping hyperparameter names to values for the best trial
    best_trial_params = {search_space[i].name: result.x[i] for i in range(len(search_space))}
    best_trial = {
        'hyperparameters': best_trial_params,
        'score': -result.fun
    }

    # Finding the worst trial
    worst_score = max(result.func_vals)
    worst_trial_index = np.argmax(result.func_vals)
    
    # Mapping hyperparameter names to values for the worst trial
    worst_trial_params = {search_space[i].name: result.x_iters[worst_trial_index][i] for i in range(len(search_space))}
    worst_trial = {
        'hyperparameters': worst_trial_params,
        'score': -worst_score
    }

    return {
        'best_trial': best_trial,
        'worst_trial': worst_trial
    }
