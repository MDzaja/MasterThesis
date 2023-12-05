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
    Y = model_utils.get_Y(labels, window_size).values

    X = feat_X

    # CNN-LSTM
    n_steps = 3
    n_length = 10
    n_features = X.shape[2]
    X = X.reshape((X.shape[0], n_steps, n_length, n_features))
    model_utils.hyperparameter_optimization_cv(cnn_lstm.build_model_hp, X, Y, 
                                            'optimization_logs/cnn_lstm/cv_feat', 'trials', 
                                            max_trials=2, executions_per_trial=2,
                                            early_stopping_patience=10, epochs=20, 
                                            batch_size=64, n_splits=5)
    # LSTM
    #model_utils.hyperparameter_optimization(lstm.build_model_hp, X_train, Y_train, X_val, Y_val, 
    #                                        'optimization_logs/lstm/test_feat', 'trials', 
    #                                        max_trials=40, executions_per_trial=2,
    #                                        early_stopping_patience=70, epochs=300, batch_size=64)
    
    # Transformer
    #model_utils.hyperparameter_optimization(tf.build_model_hp, X_train, Y_train, X_val, Y_val, 
    #                                        'optimization_logs/transformer/test_feat', 'trials', 
    #                                        max_trials=40, executions_per_trial=2, 
    #                                        early_stopping_patience=100, epochs=500, batch_size=64)
