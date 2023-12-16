import sys
sys.path.insert(0, '../')

import yfinance as yf
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf

import CNN_LSTM as cnn_lstm
import LSTM as lstm
import transformer as trans
import utils as model_utils
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl(
        '../artifacts/features/features_2009-06-22_2023-10-30.csv',
        '../artifacts/labels/labels_dict_2009-06-22_2023-10-30.pkl'
    )
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
    model_utils.hp_opt_cv(cnn_lstm.build_model_gp, 
                        cnn_lstm.define_search_space(), X, Y, 
                        'optimization_logs/cnn_lstm/cv_feat_oracle', 
                        trial_num=0, initial_random_trials=0,
                        early_stopping_patience=50, epochs=500,
                        batch_size=64, n_splits=10)
    
    # LSTM
    # model_utils.hp_opt_cv(lstm.build_model_gp,
    #                    lstm.define_search_space(), X, Y,
    #                    'optimization_logs/lstm/cv_feat_oracle', 
    #                    trial_num=100, initial_random_trials=0,
    #                    early_stopping_patience=50, epochs=500,
    #                    batch_size=64, n_splits=10)
    
    # Transformer
    #model_utils.hp_opt_cv(tf.build_model_gp,
    #                    tf.define_search_space(), X, Y, 
    #                    'optimization_logs/transformer/cv_feat', 
    #                    trial_num=2, initial_random_trials=1,
    #                    early_stopping_patience=10, epochs=20,
    #                    batch_size=64, n_splits=5)
