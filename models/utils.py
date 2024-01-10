import sys
sys.path.insert(0, '../')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
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
import yaml
from memory_profiler import profile

from labels import oracle


# TODO out of date
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


# TODO out of date
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


def load_data(path) -> pd.DataFrame:
    data_df = pd.read_csv(path, index_col=0)
    data_df.index = pd.to_datetime(data_df.index)
    return data_df

def load_labels(path, label_name=None) -> pd.Series:
    with open(path, 'rb') as file:
        labels_dict = pickle.load(file)
    labels = labels_dict[label_name]
    labels.index = pd.to_datetime(labels.index)
    return labels

def load_weights(path, label_name, weight_name) -> pd.Series:
    with open(path, 'rb') as file:
        weights_dict = pickle.load(file)
    weights = weights_dict[label_name][weight_name]
    weights.index = pd.to_datetime(weights.index)
    return weights

def load_hyperparameters(path):
    with open(path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters

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


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB


def restart_script():
    print("Restarting script...")
    os.execl(sys.executable, sys.executable, *sys.argv)


def calculate_class_weight_dict(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict


def adjust_sample_weights(y_labels, class_weight_dict, sample_weights):
    # Initialize adjusted weights with zeros
    adjusted_weights = np.zeros_like(sample_weights)

    # Adjust sample weights by multiplying with class weights
    for i, label in enumerate(y_labels):
        adjusted_weights[i] = sample_weights[i] * class_weight_dict[label]

    return adjusted_weights


def train_model(model, X_train, Y_train, X_val, Y_val, train_weights, val_weights,
                batch_size, epochs, early_stopping_patience, directory) -> Sequential:
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor=get_default_monitor_metric(), 
        patience=early_stopping_patience,
        mode='max'
    )

    # Model checkpoint callback
    checkpoint_path = f'{directory}/tmp_model.keras'
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=get_default_monitor_metric(),
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )

    # Prepare the data
    train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train, train_weights)).batch(batch_size).cache()
    val_data = tf.data.Dataset.from_tensor_slices((X_val, Y_val, val_weights)).batch(batch_size).cache()

    # Train the model
    model.fit(train_data, 
                epochs=epochs, 
                validation_data=val_data, 
                callbacks=[early_stopping, checkpoint],
                verbose=0)
    
    del train_data, val_data
    gc.collect()
    
    # Load the best weights
    model.load_weights(checkpoint_path)
    os.remove(checkpoint_path)

    return model


@profile
def custom_cross_val(params, X, Y, W, build_model_gp, n_splits, epochs, batch_size, early_stopping_patience, directory, adjustedWeightsForEval):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_metrics = {}
    best_val_auc = float('-inf')
    best_model = None

    memory_log_path = os.path.join(directory, "memory_usage.log")#TODO

    class_weight_dict = calculate_class_weight_dict(Y)
    adjusted_W = adjust_sample_weights(Y, class_weight_dict, W)

    for train_index, val_index in tscv.split(X):

        mem_before = get_memory_usage()#TODO
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage before training split: {mem_before:.2f} MB\n")

        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        W_train, W_val = W[train_index], W[val_index]
        adjusted_W_train, adjusted_W_val = adjusted_W[train_index], adjusted_W[val_index]


        model = build_model_gp(params, X_train.shape[-2], X_train.shape[-1])
        try:
            model = train_model(model, X_train, Y_train, X_val, Y_val, 
                                        adjusted_W_train, adjusted_W_val,
                                        batch_size, epochs, early_stopping_patience, directory)
        except tf.errors.InternalError or tf.errors.ResourceExhaustedError as e:
            print(f"Error: {e}")
            restart_script()

        # Evaluate the model
        #TODO sta koristit ode, adjusted ili normalni weights; mozda za hp_opt normalni a za train_test adjusted
        train_metrics = model.evaluate(X_train, Y_train, 
                                       sample_weight=adjusted_W_train if adjustedWeightsForEval else W_train,
                                       verbose=0)
        val_metrics = model.evaluate(X_val, Y_val, 
                                     sample_weight=adjusted_W_val if adjustedWeightsForEval else W_val,
                                     verbose=0)

        # Update the metrics dictionary
        for i, metric_name in enumerate(model.metrics_names):
            all_metrics.setdefault(f'train_{metric_name}', []).append(train_metrics[i])
            all_metrics.setdefault(f'val_{metric_name}', []).append(val_metrics[i])

        # Check if the current model has the best validation AUC
        auc_index = model.metrics_names.index('auc')
        current_val_auc = val_metrics[auc_index]
        if current_val_auc > best_val_auc:
            best_val_auc = current_val_auc
            best_model = model

        # Clear the model and Keras session to free memory
        del model, X_train, X_val, Y_train, Y_val
        tf.keras.backend.clear_session()
        gc.collect()

        mem_after = get_memory_usage() #TODO
        with open(memory_log_path, "a") as mem_log:
            mem_log.write(f"Memory usage after training split: {mem_after:.2f} MB\n")
            mem_log.write(f"Memory usage difference: {mem_after - mem_before:.2f} MB\n\n")

    # Calculate and return mean metrics
    mean_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    return mean_metrics, best_model


def get_all_weight_names():
    return ['backward_looking', 'forward_looking', 'sequential_return', 'trend_interval_return']


def get_all_label_names():
    return ['ct_two_state', 'ct_three_state', 'fixed_time_horizon', 'oracle', 'triple_barrier']


def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
