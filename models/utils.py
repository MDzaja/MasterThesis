import sys
sys.path.insert(0, '../')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import BayesianOptimization, Objective
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from functools import partial
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, VerboseCallback
from skopt import load
import gc
from memory_profiler import profile
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd
import pickle
import os
import psutil

from labels import oracle


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


def get_aligned_raw_feat_lbl(feat_csv_path, lbl_pkl_path):
    ticker_symbol = 'GC=F'
    start_date = '2000-01-01'
    end_date = '2023-11-01'
    raw_data = yf.download(ticker_symbol, start_date, end_date, interval='1d')
    raw_data.index = raw_data.index.tz_localize(None)

    features_df = pd.read_csv(feat_csv_path, index_col=0)

    with open(lbl_pkl_path, 'rb') as file:
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


def train_model(build_model_func, X_train, Y_train, X_val, Y_val, early_stopping_patience=20, epochs=100):
    # Compute class weights
    classes = np.unique(Y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    early_stopping = EarlyStopping(monitor=get_default_monitor_metric(), patience=early_stopping_patience)

    model = build_model_func(X_train.shape[-2], X_train.shape[-1])

    model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_val, Y_val), batch_size=64, class_weight=class_weight_dict, callbacks=[early_stopping])


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
    
    metric_history = []
    partial_objective = partial(objective_gp, X=X, Y=Y, build_model_gp=build_model_gp, epochs=epochs, 
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
                         n_calls=trial_num, n_initial_points=initial_random_trials,
                         x0=x0, y0=y0, callback=[tracker, checkpoint_saver], verbose=True)

    # Process results for hyperparameter analysis
    hyperparam_analysis = process_hyperparams(result, search_space)
    with open(f'{directory}/hp_analysis.json', 'w') as file:
        json.dump(hyperparam_analysis, file, indent=4, default=convert_types)

    # Process results for best and worst trials
    overall_info = process_trials(result, search_space, metric_history)
    with open(f'{directory}/overall_info.json', 'w') as file:
        json.dump(overall_info, file, indent=4, default=convert_types)


def objective_gp(params, X, Y, build_model_gp, epochs, batch_size, n_splits, early_stopping_patience, metric_history, directory):
    metrics = custom_cross_val_score(params, X, Y, build_model_gp, n_splits, epochs, batch_size, early_stopping_patience, directory)
    
    tf.keras.backend.clear_session()
    gc.collect()

    metric_history.append(metrics)
    return -metrics[get_default_monitor_metric()]


@profile
def custom_cross_val_score(params, X, Y, build_model_gp, n_splits, epochs, batch_size, early_stopping_patience, directory):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = {}

    memory_log_path = os.path.join(directory, "memory_usage.log")

    for train_index, val_index in tscv.split(X):

        mem_before = get_memory_usage()
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage before training split: {mem_before:.2f} MB\n")

        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                            .batch(batch_size)
                            .cache() # TODO probably remove this
                            .prefetch(tf.data.AUTOTUNE))
        val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, Y_val))
                            .batch(batch_size)
                            .cache() # TODO probably remove this
                            .prefetch(tf.data.AUTOTUNE))

        # Compute class weights
        classes = np.unique(Y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y_train)
        class_weight_dict = dict(zip(classes.astype(int), class_weights))

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor=get_default_monitor_metric(), 
            patience=early_stopping_patience,
            mode='max'
        )

        # Configure ModelCheckpoint
        checkpoint = ModelCheckpoint(
            filepath=f'{directory}/tmp_model.keras',
            monitor=get_default_monitor_metric(),
            save_best_only=True,
            save_weights_only=True,
            mode='max'
        )

        # Add profiling callback to the fit function
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../artifacts/models/memory_logs', profile_batch='500,520')

        model = build_model_gp(params, X_train.shape[-2], X_train.shape[-1])

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )

        # Load the best weights and evaluate the model
        model.load_weights(f'{directory}/tmp_model.keras')
        train_metrics = model.evaluate(X_train, Y_train, verbose=0)
        val_metrics = model.evaluate(X_val, Y_val, verbose=0)

        # Update the metrics dictionary
        for i, metric_name in enumerate(model.metrics_names):
            all_metrics.setdefault(f'train_{metric_name}', []).append(train_metrics[i])
            all_metrics.setdefault(f'val_{metric_name}', []).append(val_metrics[i])

        # Clear the model and Keras session to free memory
        del model, X_train, X_val, Y_train, Y_val, train_dataset, val_dataset
        tf.keras.backend.clear_session()
        gc.collect()

        # Remove the temporary model file
        os.remove(f'{directory}/tmp_model.keras')

        mem_after = get_memory_usage()
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage after training split: {mem_after:.2f} MB\n")
            mem_log.write(f"Memory usage difference: {mem_after - mem_before:.2f} MB\n\n")

    # Clear the memory again
    del tscv
    gc.collect()

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
    # best_trial_metrics = metric_history[best_trial_index]
    best_trial_params = {search_space[i].name: result.x[i] for i in range(len(search_space))}
    best_trial = {
        'hyperparameters': best_trial_params,
        'score': -result.fun,
        # 'metrics': best_trial_metrics
    }

    # Metrics for the worst trial
    worst_score = result.func_vals[worst_trial_index]
    # worst_trial_metrics = metric_history[worst_trial_index]
    worst_trial_params = {search_space[i].name: result.x_iters[worst_trial_index][i] for i in range(len(search_space))}
    worst_trial = {
        'hyperparameters': worst_trial_params,
        'score': -worst_score,
        # 'metrics': worst_trial_metrics
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


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def restart_script():
    print("Restarting script...")
    os.execl(sys.executable, sys.executable, *sys.argv)