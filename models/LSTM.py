import sys
sys.path.insert(0, '../')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real, Categorical

from models import utils as model_utils


def build_model_raw(window_size, n_features):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(279,
             return_sequences=True,
             input_shape=(window_size, n_features)
             )
    ))
    model.add(Bidirectional(LSTM(470, return_sequences=True)))
    model.add(Bidirectional(LSTM(81, return_sequences=False)))
    model.add(Dense(units=107, activation='relu'))
    model.add(Dropout(rate=0.001542))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0004979,
        decay_steps=7854,
        decay_rate=0.8249)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss=model_utils.get_default_loss(),
                  metrics=model_utils.get_default_metrics())

    return model


def build_model_feat(window_size, n_features):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(258,
             return_sequences=True,
             input_shape=(window_size, n_features)
             )
    ))
    model.add(Bidirectional(LSTM(34, return_sequences=False)))
    model.add(Dense(units=37, activation='relu'))
    model.add(Dropout(rate=0.49328))
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.000619,
        decay_steps=5161,
        decay_rate=0.85519)

    opt = Adam(learning_rate=lr_schedule)

    model.compile(optimizer=opt,
                  loss=model_utils.get_default_loss(),
                  metrics=model_utils.get_default_metrics())

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
        loss=model_utils.get_default_loss(),
        metrics=model_utils.get_default_metrics())

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
        Integer(32, 512, name='lstm_units_0'),
        Integer(1, 3, name='num_lstm_layers'),
        Integer(32, 256, name='dense_units'),
        Real(0.0, 0.5, name='dropout_rate_dense'),
        Real(1e-4, 0.1, name='learning_rate', prior='log-uniform'),
        Integer(1000, 10000, name='decay_steps'),
        Real(0.8, 0.99, name='decay_rate'),
        # Layer-specific hyperparameters
        Integer(32, 512, name='lstm_units_1'),
        Integer(32, 512, name='lstm_units_2'),
        Integer(32, 512, name='lstm_units_3')
    ]

