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
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl()
    labels = labels_dict['oracle']

    window_size = 30
    #raw_X = model_utils.get_X(raw_data, window_size)
    feat_X = model_utils.get_X(features_df, window_size)
    Y = model_utils.get_Y(labels, window_size).values

    X = feat_X

    # CNN-LSTM
    # n_steps = 3
    # n_length = 10
    # n_features = X.shape[2]
    # X = X.reshape((X.shape[0], n_steps, n_length, n_features))
    # model_utils.hp_opt_cv(cnn_lstm.build_model_gp, 
    #                     cnn_lstm.define_search_space(), X, Y, 
    #                     'optimization_logs/cnn_lstm/cv_feat_oracle', 
    #                     trial_num=200, initial_random_trials=20,
    #                     early_stopping_patience=50, epochs=500, 
    #                     batch_size=64, n_splits=10)
    
    # LSTM
    model_utils.hp_opt_cv(lstm.build_model_gp,
                       lstm.define_search_space(), X, Y,
                       'optimization_logs/lstm/cv_feat_oracle', 
                       trial_num=200, initial_random_trials=20,
                       early_stopping_patience=50, epochs=500,
                       batch_size=64, n_splits=10)
    
    # Transformer
    #model_utils.hp_opt_cv(tf.build_model_gp,
    #                    tf.define_search_space(), X, Y, 
    #                    'optimization_logs/transformer/cv_feat', 
    #                    trial_num=2, initial_random_trials=1,
    #                    early_stopping_patience=10, epochs=20,
    #                    batch_size=64, n_splits=5)
