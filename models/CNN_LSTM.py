import sys
sys.path.insert(0, '../')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real, Categorical

from models import utils as model_utils


def build_model_raw(n_length, n_features):

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=47, kernel_size=5, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.0325)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(481, return_sequences=True)))
    model.add(Bidirectional(LSTM(490, return_sequences=False)))
    model.add(Bidirectional(LSTM(494, return_sequences=False)))
    model.add(Dropout(0.0053))
    model.add(Dense(218, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0008095,
        decay_steps=3287,
        decay_rate=0.97)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss=model_utils.get_default_loss(),
                  metrics=model_utils.get_default_metrics())

    return model


def build_model_feat(n_length, n_features):

    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=78, kernel_size=5, activation='relu'), input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=93, kernel_size=5, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
    model.add(TimeDistributed(Dropout(0.1429)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(420, return_sequences=False)))
    model.add(Dropout(0.3032))
    model.add(Dense(49, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.00357,
        decay_steps=4538,
        decay_rate=0.80068)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss=model_utils.get_default_loss(),
                  metrics=model_utils.get_default_metrics())

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
                  loss=model_utils.get_default_loss(),
                  metrics=model_utils.get_default_metrics())

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
        Integer(32, 128, name='conv_filters_0'),
        Categorical([3, 5], name='conv_kernel_size_0'),
        Integer(1, 3, name='num_conv_layers'),
        Categorical([2, 3], name='conv_pool_size'),
        Real(0.0, 0.5, name='conv_dropout'),
        Integer(1, 3, name='num_lstm_layers'),
        Real(0.0, 0.5, name='lstm_dropout'),
        Integer(32, 256, name='dense_units'),
        Real(1e-4, 0.1, name='learning_rate', prior='log-uniform'),
        Integer(1000, 10000, name='decay_steps'),
        Real(0.8, 0.99, name='decay_rate'),
        # Layer-specific hyperparameters
        Integer(32, 128, name='conv_filters_1'),
        Categorical([3, 5], name='conv_kernel_size_1'),
        Integer(32, 128, name='conv_filters_2'),
        Categorical([3, 5], name='conv_kernel_size_2'),
        Integer(32, 512, name='lstm_units_0'),
        Integer(32, 512, name='lstm_units_1'),
        Integer(32, 512, name='lstm_units_2'),
    ]
