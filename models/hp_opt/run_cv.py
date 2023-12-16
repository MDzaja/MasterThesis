import sys
sys.path.insert(0, '../../')

import os

from models import CNN_LSTM as cnn_lstm
from models import LSTM as lstm
from models import transformer as tr
from models import utils as model_utils
from models.hp_opt import cv as cv_opt


if __name__ == '__main__':
    if (sys.argv[1] == '1'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl(
        '../../artifacts/features/features_2009-06-22_2023-10-30.csv',
        '../../artifacts/labels/labels_dict_2009-06-22_2023-10-30.pkl'
    )
    labels = labels_dict['oracle']

    if (sys.argv[2]):
        data_type = sys.argv[2]
        if (data_type == 'raw'):
            window_size = 60
            X = model_utils.get_X(raw_data, window_size)
        elif (data_type == 'feat'):
            window_size = 60
            X = model_utils.get_X(features_df, window_size)

    Y = model_utils.get_Y(labels, window_size).values

    if (sys.argv[3] and sys.argv[4]):
        model_name = sys.argv[3]

        if (model_name == 'cnn_lstm'):
            # CNN-LSTM
            n_steps = 3
            n_length = 10
            n_features = X.shape[2]
            X = X.reshape((X.shape[0], n_steps, n_length, n_features))
            cv_opt.hp_opt_cv(cnn_lstm.build_model_gp, 
                                cnn_lstm.define_search_space(), X, Y, 
                                sys.argv[4], 
                                trial_num=100, initial_random_trials=0,
                                early_stopping_patience=50, epochs=500,
                                batch_size=64, n_splits=10)
            
        elif (model_name == 'lstm'):
            # LSTM
            cv_opt.hp_opt_cv(lstm.build_model_gp,
                            lstm.define_search_space(), X, Y,
                            sys.argv[4], 
                            trial_num=100, initial_random_trials=0,
                            early_stopping_patience=50, epochs=500,
                            batch_size=64, n_splits=10)
            
        elif (model_name == 'transformer'):
            # Transformer
            cv_opt.hp_opt_cv(tr.build_model_gp,
                            tr.define_search_space(), X, Y, 
                            sys.argv[4], 
                            trial_num=100, initial_random_trials=10,
                            early_stopping_patience=50, epochs=500,
                            batch_size=64, n_splits=10)
