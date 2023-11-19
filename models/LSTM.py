from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split

import json
import os
import time
import numpy as np

import utils as model_utils


# def model(window_size, n_features):
#    _model = Sequential()
#    _model.add(LSTM(100, return_sequences=True, input_shape=(window_size, n_features)))
#    _model.add(Activation('relu'))
#    _model.add(LSTM(100, return_sequences=False))
#    _model.add(Activation('relu'))
#    _model.add(Dense(1, activation='sigmoid'))
#
#    return _model


def build_model(hp, window_size, n_features):
    model = Sequential()

    # Add the first LSTM layer with input_shape
    model.add(Bidirectional(LSTM(
        units=hp.Int('units1', min_value=32, max_value=512, step=32),
        return_sequences=True,
        input_shape=(window_size, n_features))))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.0, max_value=0.5, step=0.1)))

    # Additional LSTM layers
    for i in range(hp.Int('num_add_lstm_layers', 1, 3)):
        model.add(Bidirectional(LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=512, step=32),
            return_sequences=(i < hp.get('num_add_lstm_layers') - 1))))
        model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', min_value=0, max_value=0.5, step=0.1)))

    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_dense', min_value=0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99))

    opt = Adam(
            learning_rate=lr_schedule,
            beta_1=hp.Float('beta_1', min_value=0.85, max_value=0.95),
            beta_2=hp.Float('beta_2', min_value=0.995, max_value=0.999),
            epsilon=hp.Float('epsilon', min_value=1e-8, max_value=1e-6))

    # Compile the model
    model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy()])#, Precision(), Recall()])

    return model


def hyperparameter_optimization():
    # Data retrieval
    X, Y = model_utils.get_dummy_X_n_Y()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    # EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Bayesian Optimization tuner
    tuner = BayesianOptimization(
        lambda hp: build_model(hp, X_train.shape[1], X_train.shape[2]),
        objective='val_loss',
        max_trials=100,  # Adjust based on your computational budget
        executions_per_trial=3,  # More executions for a more robust estimate
        directory='optimization_logs',
        project_name='lstm_1'
    )

    tuner.search(X_train, Y_train,
                 epochs=100,
                 validation_data=(X_val, Y_val),
                 callbacks=[early_stopping])

    # Save the trial results
    save_trial_results(tuner, 'optimization_logs/lstm_1')
    analyze_hyperparameters(tuner, 'optimization_logs/lstm_1')


def save_trial_results(tuner, directory):
    metric_name = 'val_binary_accuracy'
    os.makedirs(directory, exist_ok=True)

    # Get all completed trials
    all_trials = tuner.oracle.get_best_trials(num_trials=tuner.oracle.max_trials)

    # Collect and save metrics for each trial
    all_trial_metrics = []
    for trial in all_trials:
        #print(trial.metrics.metrics.keys())
        if metric_name not in trial.metrics.metrics.keys():
            continue
        # Access the history of the specified metric
        metric_history = trial.metrics.get_history(metric_name)
        # Extract the last metric value as a serializable format
        last_metric_value = {'step': metric_history[-1].step, 'value': metric_history[-1].value} if metric_history else None

        all_trial_metrics.append({
            'trial_id': trial.trial_id,
            'hyperparameters': trial.hyperparameters.values,
            'score': trial.score,
            'last_metric_value': last_metric_value
        })

    # Save the trial metrics to a file (JSON format)
    with open(os.path.join(directory, 'all_trial_metrics.json'), 'w') as f:
        json.dump(all_trial_metrics, f, indent=4)

    # Get and save the best trials
    best_trials = tuner.oracle.get_best_trials(num_trials=2)
    best_trial_metrics = []
    for trial in best_trials:
        # Access the history of the specified metric
        metric_history = trial.metrics.get_history(metric_name)
        # Extract the last metric value as a serializable format
        last_metric_value = {'step': metric_history[-1].step, 'value': metric_history[-1].value} if metric_history else None

        best_trial_metrics.append({
            'trial_id': trial.trial_id,
            'hyperparameters': trial.hyperparameters.values,
            'score': trial.score,
            'last_metric_value': last_metric_value
        })

    best_trial_metrics.append(find_min_max_binary_accuracy(tuner))

    # Save the best hyperparameters info to a file (JSON format)
    with open(os.path.join(directory, 'best_trials_info.json'), 'w') as f:
        json.dump(best_trial_metrics, f, indent=4)


def analyze_hyperparameters(tuner, directory):
    analysis = {'numeric_params': {}, 'categorical_params': {}}

    # Iterate through all trials
    for trial_id, trial in tuner.oracle.trials.items():
        hps = trial.hyperparameters.values
        for hp, value in hps.items():
            if isinstance(value, (int, float)):  # Numeric hyperparameters
                if hp not in analysis['numeric_params']:
                    analysis['numeric_params'][hp] = []
                analysis['numeric_params'][hp].append(value)
            else:  # Categorical hyperparameters
                if hp not in analysis['categorical_params']:
                    analysis['categorical_params'][hp] = {}
                if value not in analysis['categorical_params'][hp]:
                    analysis['categorical_params'][hp][value] = 0
                analysis['categorical_params'][hp][value] += 1

    # Process numeric hyperparameters
    for hp, values in analysis['numeric_params'].items():
        values_array = np.array(values)
        analysis['numeric_params'][hp] = {
            'average': float(np.mean(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            '90th_percentile': float(np.percentile(values_array, 90))
        }

    # Save the analysis to a JSON file
    file_path = os.path.join(directory, 'hp_analysis.json')
    with open(file_path, 'w') as f:
        json.dump(analysis, f, indent=4)


def find_min_max_binary_accuracy(tuner):
    binary_accuracy_values = []
    val_binary_accuracy_values = []

    # Iterate through all trials
    for trial_id, trial in tuner.oracle.trials.items():
        # Check if the trial has the metrics we are interested in
        if 'binary_accuracy' in trial.metrics.metrics:
            history = trial.metrics.get_history('binary_accuracy')
            binary_accuracy_values.extend([entry.value for entry in history])

        if 'val_binary_accuracy' in trial.metrics.metrics:
            history = trial.metrics.get_history('val_binary_accuracy')
            val_binary_accuracy_values.extend([entry.value for entry in history])

    # Compute min and max
    min_max_results = {
        'min_binary_accuracy': min(binary_accuracy_values) if binary_accuracy_values else None,
        'max_binary_accuracy': max(binary_accuracy_values) if binary_accuracy_values else None,
        'min_val_binary_accuracy': min(val_binary_accuracy_values) if val_binary_accuracy_values else None,
        'max_val_binary_accuracy': max(val_binary_accuracy_values) if val_binary_accuracy_values else None
    }

    return min_max_results


if __name__ == '__main__':
    hyperparameter_optimization()
