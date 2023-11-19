from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
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


def build_model(n_length, n_features):

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=6500,
        decay_rate=0.96,
        staircase=True)

    opt = Adam(learning_rate=0.001)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model

def build_model_hp(hp, n_length, n_features):
    model = Sequential()

    # Add the first Conv1D layer with input_shape
    model.add(TimeDistributed(Conv1D(
        filters=hp.Int('first_conv_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('first_conv_kernel_size', [3, 5]),
        activation='relu'),
        input_shape=(None, n_length, n_features)
    ))

    # Additional Conv1D layers defined in a loop
    for i in range(1, hp.Int('num_conv_layers', 1, 3)):
        model.add(TimeDistributed(Conv1D(
            filters=hp.Int(f'conv_filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f'conv_kernel_size_{i}', [3, 5]),
            activation='relu')
        ))

    # MaxPooling1D and Dropout after the last Conv1D layer
    model.add(TimeDistributed(MaxPooling1D(pool_size=hp.Choice('final_pool_size', [2, 3]))))
    model.add(TimeDistributed(Dropout(rate=hp.Float('final_conv_dropout', min_value=0, max_value=0.5, step=0.1))))

    # Flatten the output of the convolutional layers
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        model.add(Bidirectional(LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=512, step=32),
            return_sequences=(i < hp.get('num_lstm_layers') - 1))))
            
    # Dropout after the last LSTM layer
    model.add(Dropout(rate=hp.Float('lstm_dropout', min_value=0, max_value=0.5, step=0.1)))        

    # Dense layer
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=0.1, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99))

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=[BinaryAccuracy(), Precision(), Recall()])

    return model

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    X, Y = model_utils.get_dummy_X_n_Y(120)
    n_steps = 12
    n_length = 10
    n_fetures = X.shape[2]
    X = X.reshape((X.shape[0], n_steps, n_length, n_fetures))
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model_utils.hyperparameter_optimization(build_model_hp, X_train, Y_train, X_val, Y_val, 
                                            'optimization_logs/cnn_lstm/test2', 'trials', 
                                            max_trials=50, executions_per_trial=2, 
                                            early_stopping_patience=100, epochs=500)
    
    #model_utils.train_model(build_model, X_train, Y_train, X_val, Y_val, early_stopping_patience=100, epochs=500)
