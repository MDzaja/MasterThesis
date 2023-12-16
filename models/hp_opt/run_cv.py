# Run example: python run_cv.py 0 raw cnn_lstm oracle /path/to/directory

import sys
sys.path.insert(0, '../../')

import os
import argparse

from models import CNN_LSTM as cnn_lstm
from models import LSTM as lstm
from models import transformer as tr
from models import utils as model_utils
from models.hp_opt import cv as cv_opt


def parse_args():
    parser = argparse.ArgumentParser(description="Model Optimization Script")
    parser.add_argument('gpu_id', type=str, choices=['0', '1'], help='GPU ID')
    parser.add_argument('data_type', type=str, choices=['raw', 'feat'], help='Data type: raw or feat')
    parser.add_argument('model_name', type=str, choices=['cnn_lstm', 'lstm', 'transformer'], help='Model name')
    parser.add_argument('labeling', type=str, choices=['ct_two_state', 'ct_three_state', 'fixed_time_horizon', 'oracle', 'triple_barrier'], help='Labeling for Ys')
    parser.add_argument('directory', type=str, help='Directory for output logs and results')
    return parser.parse_args()

def setup_logging(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    log_file = open(f'{directory}/output.log', 'a')
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file

def main(gpu_id, data_type, model_name, labeling, directory):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    raw_data, features_df, labels_dict = model_utils.get_aligned_raw_feat_lbl(
        '../../artifacts/features/features_2009-06-22_2023-10-30.csv',
        '../../artifacts/labels/labels_dict_2009-06-22_2023-10-30.pkl'
    )
    labels = labels_dict[labeling]

    window_size = 60 if data_type == 'raw' else 30
    X = model_utils.get_X(raw_data if data_type == 'raw' else features_df, window_size)
    Y = model_utils.get_Y(labels, window_size).values

    if model_name == 'cnn_lstm':
        n_length = 10
        n_steps = X.shape[1] // n_length
        n_features = X.shape[-1]
        X = X.reshape((X.shape[0], n_steps, n_length, n_features))
        model_gp, search_space = cnn_lstm.build_model_gp, cnn_lstm.define_search_space()
    elif model_name == 'lstm':
        model_gp, search_space = lstm.build_model_gp, lstm.define_search_space()
    elif model_name == 'transformer':
        model_gp, search_space = tr.build_model_gp, tr.define_search_space()

    cv_opt.hp_opt_cv(model_gp, search_space, X, Y, directory, 
                     trial_num=100, initial_random_trials=10,
                     early_stopping_patience=50, epochs=500,
                     batch_size=64, n_splits=10)

if __name__ == '__main__':
    args = parse_args()
    log_file = setup_logging(args.directory)
    main(args.gpu_id, args.data_type, args.model_name, args.labeling, args.directory)
    log_file.close()