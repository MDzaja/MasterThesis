import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

import json
import os
import numpy as np

import utils as model_utils


def build_model(window_size, n_features):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(170,
             return_sequences=True,
             input_shape=(window_size, n_features)
             )
    ))
    model.add(Bidirectional(LSTM(360, return_sequences=True)))
    model.add(Bidirectional(LSTM(360, return_sequences=False)))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dropout(rate=0.1))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=6500,
        decay_rate=0.96,
        staircase=True)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model

def train_model():
    X, Y = model_utils.get_dummy_X_n_Y()

    # Compute class weights
    classes = np.unique(Y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=Y.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)
    window_size, n_features = X_train.shape[1], X_train.shape[2]

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    model = build_model(window_size, n_features)

    model.fit(X_train, Y_train, epochs=100, validation_data=(X_val, Y_val), batch_size=64, class_weight=class_weight_dict, callbacks=[early_stopping])


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
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99))

    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model


if __name__ == '__main__':
    X, Y = model_utils.get_dummy_X_n_Y()
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model_utils.tune_hyperparameters(build_model_hp, X_train, Y_train, X_val, Y_val, 
                                            'optimization_logs/lstm/test1', 'trials', 
                                            max_trials=50, executions_per_trial=2, 
                                            early_stopping_patience=30, epochs=150)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_utils.train_model(build_model, X_train, Y_train, X_val, Y_val, early_stopping_patience=30, epochs=150)
