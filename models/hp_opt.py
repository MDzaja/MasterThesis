import yfinance as yf
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split

import CNN_LSTM as cnn_lstm
import LSTM as lstm
import transformer as tf
import utils as model_utils
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl()
    labels = labels_dict['oracle']

    window_size = 30
    #raw_X = model_utils.get_X(raw_data, window_size)
    feat_X = model_utils.get_X(features_df, window_size)
    Y = model_utils.get_Y(labels, window_size)
    
    X_train, X_val = train_test_split(feat_X, test_size=0.2, shuffle=False)
    Y_train, Y_val = train_test_split(Y, test_size=0.2, shuffle=False)

    # CNN-LSTM
    n_steps = 3
    n_length = 10
    n_features = X_train.shape[2]
    _X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_features))
    _X_val = X_val.reshape((X_val.shape[0], n_steps, n_length, n_features))
    model_utils.hyperparameter_optimization(cnn_lstm.build_model_hp, _X_train, Y_train, _X_val, Y_val, 
                                            'optimization_logs/cnn_lstm/test_feat', 'trials', 
                                            max_trials=40, executions_per_trial=2,
                                            early_stopping_patience=100, epochs=500, batch_size=64)
    # LSTM
    model_utils.hyperparameter_optimization(lstm.build_model_hp, X_train, Y_train, X_val, Y_val, 
                                            'optimization_logs/lstm/test_feat', 'trials', 
                                            max_trials=40, executions_per_trial=2,
                                            early_stopping_patience=70, epochs=300, batch_size=64)
    
    # Transformer
    model_utils.hyperparameter_optimization(tf.build_model_hp, X_train, Y_train, X_val, Y_val, 
                                            'optimization_logs/transformer/test_feat', 'trials', 
                                            max_trials=40, executions_per_trial=2, 
                                            early_stopping_patience=100, epochs=500, batch_size=64)
