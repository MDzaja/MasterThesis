import sys
sys.path.insert(0, '../')

from tensorflow.keras import Model
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
# Layer-specific hyperparameters, depending on NUM_LSTM_LAYERS
LSTM_UNITS_1 = 'lstm_units_1'
LSTM_UNITS_2 = 'lstm_units_2'
LSTM_UNITS_3 = 'lstm_units_3'


class CustomLSTM(Model):
    def __init__(self, params, window_size, n_features):
        super(CustomLSTM, self).__init__()
        self.params = params
        self.window_size = window_size
        self.n_features = n_features

        # LSTM layers
        self.lstm_layers = []
        # First LSTM layer
        self.lstm_layers.append(Bidirectional(LSTM(units=params[LSTM_UNITS_0], 
                                                return_sequences=params[NUM_LSTM_LAYERS] > 1, 
                                                input_shape=(window_size, n_features))))
        # Additional LSTM layers
        for i in range(1, params[NUM_LSTM_LAYERS]):
            lstm_units_key = globals()[f'LSTM_UNITS_{i}']
            return_sequences = i < params[NUM_LSTM_LAYERS] - 1
            self.lstm_layers.append(Bidirectional(LSTM(params[lstm_units_key], 
                                                    return_sequences=return_sequences)))

        # Dense and Dropout layers
        self.dense_layer = Dense(params[DENSE_UNITS], activation='relu')
        self.dropout_layer = Dropout(params[DROPOUT_RATE_DENSE])
        self.output_layer = Dense(1, activation='sigmoid')


    def call(self, inputs):
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)

        x = self.dense_layer(x)
        x = self.dropout_layer(x)
        return self.output_layer(x)


    def compile(self):
        # Learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.params[LEARNING_RATE],
            decay_steps=self.params[DECAY_STEPS],
            decay_rate=self.params[DECAY_RATE]
        )

        # Optimizer
        opt = Adam(learning_rate=lr_schedule)
        super(CustomLSTM, self).compile(
            optimizer=opt,
            loss=model_utils.get_default_loss(),
            weighted_metrics=model_utils.get_default_metrics()
        )


def build_model(params, window_size, n_features):
    model = CustomLSTM(params, window_size, n_features)
    model.compile()
    return model


def build_model_gp(params_array, window_size, n_features):
    params_dict = {
        LSTM_UNITS_0: params_array[0],
        NUM_LSTM_LAYERS: params_array[1],
        DENSE_UNITS: params_array[2],
        DROPOUT_RATE_DENSE: params_array[3],
        LEARNING_RATE: params_array[4],
        DECAY_STEPS: params_array[5],
        DECAY_RATE: params_array[6],
        LSTM_UNITS_1: params_array[7] if len(params_array) > 7 else None,
        LSTM_UNITS_2: params_array[8] if len(params_array) > 8 else None,
        LSTM_UNITS_3: params_array[9] if len(params_array) > 9 else None
    }
    model = CustomLSTM(params_dict, window_size, n_features)
    model.compile()
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
        # Layer-specific hyperparameters, depending on NUM_LSTM_LAYERS
        Integer(32, 512, name=LSTM_UNITS_1),
        Integer(32, 512, name=LSTM_UNITS_2),
        Integer(32, 512, name=LSTM_UNITS_3)
    ]
