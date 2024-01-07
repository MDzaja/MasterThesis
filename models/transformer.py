import sys
sys.path.insert(0, '../')

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real, Categorical

from models import utils as model_utils


# Constants for parameter keys
NUM_TRANSFORMER_BLOCKS = 'num_transformer_blocks'
HEAD_SIZE = 'head_size'
NUM_HEADS = 'num_heads'
FF_DIM = 'ff_dim'
TRANSFORMER_DROPOUT = 'transformer_dropout'
NUM_MLP_LAYERS = 'num_mlp_layers'
MLP_UNITS_0 = 'mlp_units_0'
MLP_DROPOUT = 'mlp_dropout'
LEARNING_RATE = 'learning_rate'
DECAY_STEPS = 'decay_steps'
DECAY_RATE = 'decay_rate'
# Layer-specific hyperparameters, depending on NUM_MLP_LAYERS
MLP_UNITS_1 = 'mlp_units_1'
MLP_UNITS_2 = 'mlp_units_2'
MLP_UNITS_3 = 'mlp_units_3'


class CustomTransformer(Model):
    def __init__(self, params, window_size, n_features):
        super(CustomTransformer, self).__init__()
        self.params = params
        self.window_size = window_size
        self.n_features = n_features

        # Define Transformer blocks
        self.transformer_blocks = [
            self._build_transformer_block()
            for _ in range(params[NUM_TRANSFORMER_BLOCKS])
        ]

        # Define GlobalAveragePooling1D layer
        self.global_avg_pool = GlobalAveragePooling1D(data_format="channels_first")

        # Define MLP layers
        self.mlp_layers = []
        for i in range(params[NUM_MLP_LAYERS]):
            mlp_units_key = f'MLP_UNITS_{i}'
            mlp_units = globals()[mlp_units_key]
            self.mlp_layers.append(Dense(params[mlp_units], activation="relu"))
            self.mlp_layers.append(Dropout(params[MLP_DROPOUT]))

        # Define Output layer
        self.output_layer = Dense(1, activation='sigmoid')


    def _build_transformer_block(self):
        inputs = Input(shape=(self.window_size, self.n_features))
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=self.params[HEAD_SIZE], num_heads=self.params[NUM_HEADS], dropout=self.params[TRANSFORMER_DROPOUT]
        )(x, x)
        x = Dropout(self.params[TRANSFORMER_DROPOUT])(x)
        res = x + inputs

        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(filters=self.params[FF_DIM], kernel_size=1, activation="relu")(x)
        x = Dropout(self.params[TRANSFORMER_DROPOUT])(x)
        x = Conv1D(filters=self.n_features, kernel_size=1)(x)
        return Model(inputs, x + res)


    def call(self, inputs):
        x = inputs
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = self.global_avg_pool(x)

        for layer in self.mlp_layers:
            x = layer(x)

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
        super(CustomTransformer, self).compile(
            optimizer=opt,
            loss=model_utils.get_default_loss(),
            weighted_metrics=model_utils.get_default_metrics()
        )


def build_model(params, window_size, n_features):
    model = CustomTransformer(params, window_size, n_features)
    model.compile()
    return model


def build_model_gp(params_array, window_size, n_features):
    params_dict = {
        NUM_TRANSFORMER_BLOCKS: params_array[0],
        HEAD_SIZE: params_array[1],
        NUM_HEADS: params_array[2],
        FF_DIM: params_array[3],
        TRANSFORMER_DROPOUT: params_array[4],
        NUM_MLP_LAYERS: params_array[5],
        MLP_UNITS_0: params_array[6],
        MLP_DROPOUT: params_array[7],
        LEARNING_RATE: params_array[8],
        DECAY_STEPS: params_array[9],
        DECAY_RATE: params_array[10],
        MLP_UNITS_1: params_array[11] if len(params_array) > 11 else None,
        MLP_UNITS_2: params_array[12] if len(params_array) > 12 else None,
        MLP_UNITS_3: params_array[13] if len(params_array) > 13 else None
    }
    model = CustomTransformer(params_dict, window_size, n_features)
    model.compile()
    return model


def define_search_space():
    return [
        Integer(1, 3, name=NUM_TRANSFORMER_BLOCKS),
        Integer(32, 256, name=HEAD_SIZE),
        Categorical([2, 4, 8], name=NUM_HEADS),
        Integer(2, 64, name=FF_DIM),
        Real(0.0, 0.5, name=TRANSFORMER_DROPOUT),
        Integer(1, 3, name=NUM_MLP_LAYERS),
        Integer(32, 512, name=MLP_UNITS_0),
        Real(0.0, 0.5, name=MLP_DROPOUT),
        Real(1e-4, 0.1, name=LEARNING_RATE, prior='log-uniform'),
        Integer(1000, 10000, name=DECAY_STEPS),
        Real(0.8, 0.99, name=DECAY_RATE),
        # Layer-specific hyperparameters, depending on NUM_MLP_LAYERS
        Integer(32, 512, name=MLP_UNITS_1),
        Integer(32, 512, name=MLP_UNITS_2),
        Integer(32, 512, name=MLP_UNITS_3)
    ]
