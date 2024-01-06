import sys
sys.path.insert(0, '../')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real, Categorical

from models import utils as model_utils


# Constants for parameter keys
CONV_FILTERS_0 = 'conv_filters_0'
CONV_KERNEL_SIZE_0 = 'conv_kernel_size_0'
NUM_CONV_LAYERS = 'num_conv_layers'
CONV_POOL_SIZE = 'conv_pool_size'
CONV_DROPOUT = 'conv_dropout'
NUM_LSTM_LAYERS = 'num_lstm_layers'
LSTM_DROPOUT = 'lstm_dropout'
DENSE_UNITS = 'dense_units'
LEARNING_RATE = 'learning_rate'
DECAY_STEPS = 'decay_steps'
DECAY_RATE = 'decay_rate'
CONV_FILTERS_1 = 'conv_filters_1'
CONV_KERNEL_SIZE_1 = 'conv_kernel_size_1'
CONV_FILTERS_2 = 'conv_filters_2'
CONV_KERNEL_SIZE_2 = 'conv_kernel_size_2'
LSTM_UNITS_0 = 'lstm_units_0'
LSTM_UNITS_1 = 'lstm_units_1'
LSTM_UNITS_2 = 'lstm_units_2'


def build_model(params, n_length, n_features):
    model = Sequential()

    # Conv1D and MaxPooling1D layers
    for i in range(params[NUM_CONV_LAYERS]):
        filters_key = f'CONV_FILTERS_{i}'
        conv_filters = globals()[filters_key]
        kernel_size_key = f'CONV_KERNEL_SIZE_{i}'
        kernel_size = globals()[kernel_size_key]
        if i == 0:
            model.add(TimeDistributed(Conv1D(filters=params.get(conv_filters), 
                                            kernel_size=params.get(kernel_size), 
                                            activation='relu'), 
                                    input_shape=(None, n_length, n_features)))
        else:
            model.add(TimeDistributed(Conv1D(filters=params.get(conv_filters), 
                                            kernel_size=params.get(kernel_size), 
                                            activation='relu', padding='same')))
            
    model.add(TimeDistributed(MaxPooling1D(pool_size=params[CONV_POOL_SIZE])))
    model.add(TimeDistributed(Dropout(params[CONV_DROPOUT])))
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    for i in range(params[NUM_LSTM_LAYERS]):
        lstm_units_key = f'LSTM_UNITS_{i}'
        lstm_units = globals()[lstm_units_key]
        model.add(Bidirectional(LSTM(units=params.get(lstm_units), 
                                     return_sequences=(i < params[NUM_LSTM_LAYERS] - 1))))

    # Dropout after LSTM layers
    model.add(Dropout(params[LSTM_DROPOUT]))

    # Dense layer
    model.add(Dense(params[DENSE_UNITS], activation='relu'))
    
    # Output layer
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


def build_model_gp(params, n_length, n_features):
    # Unpack parameters
    conv_filters_0, conv_kernel_size_0, num_conv_layers, conv_pool_size, conv_dropout, \
    num_lstm_layers, lstm_dropout, dense_units, learning_rate, decay_steps, decay_rate, \
    conv_filters_1, conv_kernel_size_1, conv_filters_2, conv_kernel_size_2, \
    lstm_units_0, lstm_units_1, lstm_units_2 = params

    model = Sequential()
    # First Conv1D layer (always included)
    model.add(TimeDistributed(Conv1D(filters=conv_filters_0, 
                                     kernel_size=(conv_kernel_size_0,), 
                                     activation='relu'), 
                              input_shape=(None, n_length, n_features)))

    # Additional Conv1D layers (if present)
    for i in range(1, num_conv_layers):
        model.add(TimeDistributed(Conv1D(filters=locals().get(f'conv_filters_{i}'), 
                                         kernel_size=(locals().get(f'conv_kernel_size_{i}'),), 
                                         activation='relu',
                                         padding='same')))

    # MaxPooling1D and Dropout after the Conv1D layers
    model.add(TimeDistributed(MaxPooling1D(pool_size=(conv_pool_size,))))
    model.add(TimeDistributed(Dropout(rate=conv_dropout)))

    # Flatten the output
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    for i in range(num_lstm_layers):
        lstm_units = locals().get(f'lstm_units_{i}')
        return_sequences = i < num_lstm_layers - 1
        return_sequences = True if return_sequences else False #TODO check why is this needed, it throws an error otherwise; Using a symbolic `tf.Tensor` as a Python `bool` is not allowed...
        model.add(Bidirectional(LSTM(units=lstm_units, 
                                     return_sequences=return_sequences)))

    # Dropout after LSTM layers
    model.add(Dropout(rate=lstm_dropout))

    # Dense layer
    model.add(Dense(units=dense_units, activation='relu'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate schedule
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, 
                                   decay_steps=decay_steps, 
                                   decay_rate=decay_rate)

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt, 
                  loss=model_utils.get_default_loss(), 
                  metrics=model_utils.get_default_metrics())

    return model

def define_search_space():
    return [
        Integer(32, 128, name=CONV_FILTERS_0),
        Categorical([3, 5], name=CONV_KERNEL_SIZE_0),
        Integer(1, 3, name=NUM_CONV_LAYERS),
        Categorical([2, 3], name=CONV_POOL_SIZE),
        Real(0.0, 0.5, name=CONV_DROPOUT),
        Integer(1, 3, name=NUM_LSTM_LAYERS),
        Real(0.0, 0.5, name=LSTM_DROPOUT),
        Integer(32, 256, name=DENSE_UNITS),
        Real(1e-4, 0.1, name=LEARNING_RATE, prior='log-uniform'),
        Integer(1000, 10000, name=DECAY_STEPS),
        Real(0.8, 0.99, name=DECAY_RATE),
        # Layer-specific hyperparameters
        Integer(32, 128, name=CONV_FILTERS_1),
        Categorical([3, 5], name=CONV_KERNEL_SIZE_1),
        Integer(32, 128, name=CONV_FILTERS_2),
        Categorical([3, 5], name=CONV_KERNEL_SIZE_2),
        Integer(32, 512, name=LSTM_UNITS_0),
        Integer(32, 512, name=LSTM_UNITS_1),
        Integer(32, 512, name=LSTM_UNITS_2),
    ]
