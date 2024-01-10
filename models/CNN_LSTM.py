import sys
sys.path.insert(0, '../')

from tensorflow.keras import Model
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
# Layer-specific hyperparameters, depending on NUM_CONV_LAYERS and NUM_LSTM_LAYERS
CONV_FILTERS_1 = 'conv_filters_1'
CONV_KERNEL_SIZE_1 = 'conv_kernel_size_1'
CONV_FILTERS_2 = 'conv_filters_2'
CONV_KERNEL_SIZE_2 = 'conv_kernel_size_2'
LSTM_UNITS_0 = 'lstm_units_0'
LSTM_UNITS_1 = 'lstm_units_1'
LSTM_UNITS_2 = 'lstm_units_2'


class CustomCNNLSTM(Model):
    def __init__(self, params, n_length, n_features):
        super(CustomCNNLSTM, self).__init__()
        self.params = params

        # CNN layers wrapped in TimeDistributed
        self.cnn_layers = []
        for i in range(params[NUM_CONV_LAYERS]):
            conv_filters_key = globals()[f'CONV_FILTERS_{i}']
            conv_kernel_size_key = globals()[f'CONV_KERNEL_SIZE_{i}']
            if i == 0:
                self.cnn_layers.append(TimeDistributed(Conv1D(filters=params[conv_filters_key], 
                                                         kernel_size=params[conv_kernel_size_key], 
                                                         activation='relu'), 
                                        input_shape=(None, n_length, n_features)))
            else:
                self.cnn_layers.append(TimeDistributed(Conv1D(filters=params[conv_filters_key], 
                                                         kernel_size=params[conv_kernel_size_key], 
                                                         activation='relu', padding='same')))

        self.pooling = TimeDistributed(MaxPooling1D(pool_size=params[CONV_POOL_SIZE]))
        self.cnn_dropout = TimeDistributed(Dropout(params[CONV_DROPOUT]))
        self.flatten = TimeDistributed(Flatten())

        # LSTM layers
        self.lstm_layers = []
        for i in range(params[NUM_LSTM_LAYERS]):
            lstm_units_key = globals()[f'LSTM_UNITS_{i}']
            return_sequences = i < params[NUM_LSTM_LAYERS] - 1
            self.lstm_layers.append(Bidirectional(LSTM(params[lstm_units_key], return_sequences=return_sequences)))

        self.lstm_dropout = Dropout(params[LSTM_DROPOUT])
        self.dense_layer = Dense(params[DENSE_UNITS], activation='relu')
        self.output_layer = Dense(1, activation='sigmoid')


    def call(self, inputs):
        x = inputs
        for layer in self.cnn_layers:
            x = layer(x)

        x = self.pooling(x)
        x = self.cnn_dropout(x)
        x = self.flatten(x)

        for layer in self.lstm_layers:
            x = layer(x)

        x = self.lstm_dropout(x)
        x = self.dense_layer(x)
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
        super(CustomCNNLSTM, self).compile(
            optimizer=opt,
            loss=model_utils.get_default_loss(),
            weighted_metrics=model_utils.get_default_metrics()
        )


def build_model(params, n_length, n_features):
    model = CustomCNNLSTM(params, n_length, n_features)
    model.compile()
    return model


def build_model_gp(params_array, n_length, n_features):
    params_dict = {
        CONV_FILTERS_0: params_array[0],
        CONV_KERNEL_SIZE_0: params_array[1],
        NUM_CONV_LAYERS: params_array[2],
        CONV_POOL_SIZE: params_array[3],
        CONV_DROPOUT: params_array[4],
        NUM_LSTM_LAYERS: params_array[5],
        LSTM_DROPOUT: params_array[6],
        DENSE_UNITS: params_array[7],
        LEARNING_RATE: params_array[8],
        DECAY_STEPS: params_array[9],
        DECAY_RATE: params_array[10],
        CONV_FILTERS_1: params_array[11] if len(params_array) > 11 else None,
        CONV_KERNEL_SIZE_1: params_array[12] if len(params_array) > 12 else None,
        CONV_FILTERS_2: params_array[13] if len(params_array) > 13 else None,
        CONV_KERNEL_SIZE_2: params_array[14] if len(params_array) > 14 else None,
        LSTM_UNITS_0: params_array[15] if len(params_array) > 15 else None,
        LSTM_UNITS_1: params_array[16] if len(params_array) > 16 else None,
        LSTM_UNITS_2: params_array[17] if len(params_array) > 17 else None
    }
    model = CustomCNNLSTM(params_dict, n_length, n_features)
    model.compile()
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
