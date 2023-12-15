from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from skopt.space import Integer, Real, Categorical

from sklearn.model_selection import train_test_split
import os
import numpy as np

import utils as model_utils

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


def build_model_raw(window_size, n_features):
    num_transformer_blocks = 2
    head_size = 128
    num_heads = 4
    ff_dim = 16
    dropout = 0.2
    mlp_units = [160]
    mlp_dropout = 0

    inputs = Input(shape=(window_size, n_features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0007,
        decay_steps=2000,
        decay_rate=0.97)

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=model_utils.get_default_loss(),
        metrics=model_utils.get_default_metrics()
    )

    return model


def build_model_feat(window_size, n_features):
    num_transformer_blocks = 3
    head_size = 64
    num_heads = 8
    ff_dim = 2
    dropout = 0.1
    mlp_units = [480, 256]
    mlp_dropout = 0.3

    inputs = Input(shape=(window_size, n_features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0028,
        decay_steps=5000,
        decay_rate=0.84)

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=model_utils.get_default_loss(),
        metrics=model_utils.get_default_metrics()
    )

    return model


def build_model_hp(hp, window_size, n_features):
    inputs = Input(shape=(window_size, n_features))
    x = inputs

    # Transformer blocks
    for i in range(hp.Int('num_transformer_blocks', 1, 3)):
        x = transformer_encoder(
            x,
            head_size=hp.Int('head_size', min_value=32, max_value=256, step=32),
            num_heads=hp.Choice('num_heads', [2, 4, 8]),
            ff_dim=hp.Int('ff_dim', min_value=2, max_value=64),
            dropout=hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)
        )

    # GlobalAveragePooling1D layer
    x = GlobalAveragePooling1D(data_format="channels_first")(x)

    # MLP layers with hyperparameters
    for i in range(hp.Int('num_mlp_layers', 1, 3)):  # Number of MLP layers as a hyperparameter
        x = Dense(
            hp.Int(f'mlp_units_{i}', min_value=32, max_value=512, step=32),
            activation="relu"
        )(x)
        x = Dropout(
            hp.Float('mlp_dropout', min_value=0.0, max_value=0.5, step=0.1)
        )(x)

    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=0.1, sampling='log'),
        decay_steps=hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000),
        decay_rate=hp.Float('decay_rate', min_value=0.8, max_value=0.99))

    # Optimizer
    opt = Adam(learning_rate=lr_schedule)

    # Compile the model
    model.compile(
        optimizer=opt,
        loss=model_utils.get_default_loss(),
        metrics=model_utils.get_default_metrics()
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
    for i in range(0, num_mlp_layers):
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
        Integer(1, 3, name='num_transformer_blocks'),
        Integer(32, 256, name='head_size'),
        Categorical([2, 4, 8], name='num_heads'),
        Integer(2, 64, name='ff_dim'),
        Real(0.0, 0.5, name='transformer_dropout'),
        Integer(1, 3, name='num_mlp_layers'),
        Integer(32, 512, name='mlp_units_0'),
        Real(0.0, 0.5, name='mlp_dropout'),
        Real(1e-4, 0.1, name='learning_rate', prior='log-uniform'),
        Integer(1000, 10000, name='decay_steps'),
        Real(0.8, 0.99, name='decay_rate'),
        # Layer-specific hyperparameters
        Integer(32, 512, name='mlp_units_1'),
        Integer(32, 512, name='mlp_units_2'),
        Integer(32, 512, name='mlp_units_3')
    ]


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    X, Y = model_utils.get_ft_n_Y(window_size=60)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

    model_utils.hyperparameter_optimization(build_model_hp, X_train, Y_train, X_val, Y_val,
                                            'optimization_logs/transformer/test_w_features', 'trials',
                                            max_trials=50, executions_per_trial=2,
                                            early_stopping_patience=100, epochs=500, batch_size=64)
