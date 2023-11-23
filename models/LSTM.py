import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

import json
import os
import numpy as np

import utils as model_utils


def build_model_raw(window_size, n_features):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(224,
             return_sequences=True,
             input_shape=(window_size, n_features)
             )
    ))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(416, return_sequences=True)))
    model.add(Bidirectional(LSTM(480, return_sequences=False)))
    model.add(Dense(units=192, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0021,
        decay_steps=3000,
        decay_rate=0.98)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])

    return model

def build_model_feat(window_size, n_features):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(288,
             return_sequences=True,
             input_shape=(window_size, n_features)
             )
    ))
    model.add(Bidirectional(LSTM(288, return_sequences=True)))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(288, return_sequences=False)))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0063,
        decay_steps=9000,
        decay_rate=0.91)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])

    return model


def build_model_hp(hp, window_size, n_features):
    model = Sequential()

    # Add the first LSTM layer with input_shape
    model.add(Bidirectional(LSTM(
        units=hp.Int('units1', min_value=32, max_value=512, step=32),
        return_sequences=True,
        input_shape=(window_size, n_features))))

    # Additional LSTM layers
    for i in range(hp.Int('num_add_lstm_layers', 1, 3)):
        model.add(Bidirectional(LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=512, step=32),
            return_sequences=(i < hp.get('num_add_lstm_layers') - 1))))

    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_dense', min_value=0, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=0.1, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99))

    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model

def cv_train_model():
    X, Y = model_utils.get_dummy_X_n_Y()

    tscv = TimeSeriesSplit(n_splits=10)

    # Iterate over each fold
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Training on fold {fold}...")
        
        # Split your data into training and testing sets for the current fold
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # Determine the window size and number of features from the input shape
        window_size, n_features = X_train.shape[1], X_train.shape[2]
        
        # Build and compile the model
        model = build_model(window_size, n_features)
        
        # Fit the model to the training data
        model.fit(X_train, Y_train, epochs=10)  # You can add more parameters such as batch_size if needed
        
        # Evaluate the model on the test data
        eval_result = model.evaluate(X_test, Y_test)
        print(f"Fold {fold} - Loss: {eval_result[0]}, Binary Accuracy: {eval_result[1]}")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    X, Y = model_utils.get_ft_n_Y(window_size=60)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model_utils.hyperparameter_optimization(build_model_hp, X_train, Y_train, X_val, Y_val, 
                                            'optimization_logs/lstm/test_w_features', 'trials', 
                                            max_trials=50, executions_per_trial=2, 
                                            early_stopping_patience=30, epochs=150)
    
    
    #model_utils.train_model(build_model, X_train, Y_train, X_val, Y_val, early_stopping_patience=30, epochs=150)
