import sys
sys.path.insert(0, '../')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real

from models import utils as model_utils


# Constants for parameter keys
LSTM_UNITS_0 = 'lstm_units_0'
NUM_LSTM_LAYERS = 'num_lstm_layers'
DENSE_UNITS = 'dense_units'
DROPOUT_RATE_DENSE = 'dropout_rate_dense'
LEARNING_RATE = 'learning_rate'
DECAY_STEPS = 'decay_steps'
DECAY_RATE = 'decay_rate'
LSTM_UNITS_1 = 'lstm_units_1'
LSTM_UNITS_2 = 'lstm_units_2'
LSTM_UNITS_3 = 'lstm_units_3'


def build_model(params, window_size, n_features):
    model = Sequential()

    # First LSTM layer
    model.add(Bidirectional(LSTM(units=params[LSTM_UNITS_0], 
                                 return_sequences=params[NUM_LSTM_LAYERS] > 1, 
                                 input_shape=(window_size, n_features))))

    # Additional LSTM layers
    for i in range(1, params[NUM_LSTM_LAYERS]):
        lstm_units_key = f'LSTM_UNITS_{i}'
        model.add(Bidirectional(LSTM(units=params.get(lstm_units_key), 
                                     return_sequences=(i < params[NUM_LSTM_LAYERS] - 1))))

    # Dense and Dropout layers
    model.add(Dense(units=params[DENSE_UNITS], activation='relu'))
    model.add(Dropout(rate=params[DROPOUT_RATE_DENSE]))
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate schedule
    lr_schedule = ExponentialDecay(initial_learning_rate=params[LEARNING_RATE], 
                                   decay_steps=params[DECAY_STEPS], 
                                   decay_rate=params[DECAY_RATE])
    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt, 
                  loss=model_utils.get_default_loss(), 
                  weighted_metrics=model_utils.get_default_metrics())

    return model


def build_model_gp(params, window_size, n_features):
    # Unpack parameters
    lstm_units_0, num_lstm_layers, dense_units, dropout_rate_dense, \
    learning_rate, decay_steps, decay_rate, \
    lstm_units_1, lstm_units_2, lstm_units_3 = params

    model = Sequential()

    # First LSTM layer
    return_sequences = True if num_lstm_layers > 1 else False
    model.add(Bidirectional(LSTM(units=lstm_units_0, return_sequences=return_sequences, input_shape=(window_size, n_features))))

    # Additional LSTM layers
    for i in range(1, num_lstm_layers):
        lstm_units = locals().get(f'lstm_units_{i}')
        return_sequences = True if i < num_lstm_layers - 1 else False
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=return_sequences)))

    # Dense and Dropout layers
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(rate=dropout_rate_dense))
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate schedule
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt, loss=model_utils.get_default_loss(), metrics=model_utils.get_default_metrics())

    return model


def define_search_space():
    return [
        Integer(32, 512, name=LSTM_UNITS_0),
        Integer(1, 3, name=NUM_LSTM_LAYERS),
        Integer(32, 256, name=DENSE_UNITS),
        Real(0.0, 0.5, name=DROPOUT_RATE_DENSE),
        Real(1e-4, 0.1, name=LEARNING_RATE, prior='log-uniform'),
        Integer(1000, 10000, name=DECAY_STEPS),
        Real(0.8, 0.99, name=DECAY_RATE),
        # Layer-specific hyperparameters
        Integer(32, 512, name=LSTM_UNITS_1),
        Integer(32, 512, name=LSTM_UNITS_2),
        Integer(32, 512, name=LSTM_UNITS_3)
    ]
