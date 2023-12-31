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
MLP_UNITS_1 = 'mlp_units_1'
MLP_UNITS_2 = 'mlp_units_2'
MLP_UNITS_3 = 'mlp_units_3'


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(params, window_size, n_features):
    inputs = Input(shape=(window_size, n_features))
    x = inputs

    # Transformer blocks
    for _ in range(params[NUM_TRANSFORMER_BLOCKS]):
        x = transformer_encoder(
            x,
            head_size=params[HEAD_SIZE],
            num_heads=params[NUM_HEADS],
            ff_dim=params[FF_DIM],
            dropout=params[TRANSFORMER_DROPOUT]
        )

    # GlobalAveragePooling1D layer
    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # MLP layers
    for i in range(params[NUM_MLP_LAYERS]):
        mlp_units_key = f'MLP_UNITS_{i}'
        x = Dense(params.get(mlp_units_key), activation="relu")(x)
        x = Dropout(params[MLP_DROPOUT])(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=params[LEARNING_RATE],
        decay_steps=params[DECAY_STEPS],
        decay_rate=params[DECAY_RATE]
    )

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=model_utils.get_default_loss(),
        weighted_metrics=model_utils.get_default_metrics()
    )

    return model


def build_model_gp(params, window_size, n_features):
    # Unpack parameters
    num_transformer_blocks, head_size, num_heads, ff_dim, transformer_dropout, \
    num_mlp_layers, mlp_units_0, mlp_dropout, \
    learning_rate, decay_steps, decay_rate, \
    mlp_units_1, mlp_units_2, mlp_units_3 = params

    inputs = Input(shape=(window_size, n_features))
    x = inputs

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(
            x,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=transformer_dropout
        )

    # GlobalAveragePooling1D layer
    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # MLP layers
    for i in range(num_mlp_layers):
        mlp_units = locals().get(f'mlp_units_{i}')
        x = Dense(mlp_units, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate)

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(optimizer=opt, loss=model_utils.get_default_loss(), metrics=model_utils.get_default_metrics())

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
        # Layer-specific hyperparameters
        Integer(32, 512, name=MLP_UNITS_1),
        Integer(32, 512, name=MLP_UNITS_2),
        Integer(32, 512, name=MLP_UNITS_3)
    ]
