import sys
sys.path.insert(0, '../')

from skopt.space import Real, Integer
from skopt import BayesSearchCV
import xgboost as xgb
import pandas as pd
import json
import numpy as np
import optuna
from functools import partial
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from models import utils as model_utils


def xgb_hp_opt_cv(X, Y, W, directory, use_class_balancing, trial_num=100, n_splits=5):
    X = X.reshape(X.shape[0], -1).tolist()

    if use_class_balancing:
            class_weight_dict = model_utils.calculate_class_weight_dict(Y)
            W = model_utils.adjust_sample_weights(Y, class_weight_dict, W)

    cv_indices_list = model_utils.custom_time_series_split(X, n_splits=n_splits)
    cv_indices_list = [(train_index, test_index) for train_index, test_index in cv_indices_list]

     # create a study
    study = optuna.create_study(direction='maximize')
    partial_objective = partial(objective, X=X, Y=Y, W=W, cv_indices_list=cv_indices_list)
    study.optimize(partial_objective, n_trials=trial_num)

    # get the best trial
    trial = study.best_trial

    with open(f'{directory}/best_hp.json', 'w') as fp:
        json.dump(trial.params, fp)
    
    with open(f'{directory}/best_score.txt', 'w') as fp:
        fp.write(f'Best AUC score: {trial.value*100}%\n')


def objective(trial, X, Y, W, cv_indices_list):
    n_estimators = trial.suggest_int('n_estimators', 100, 240)
    params = {
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': n_estimators,
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 10),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.1),
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor'
    }

    dtrain = xgb.DMatrix(data=X, label=Y, weight=W, silent=False)

    #train_folds = [[train_index for train_index, _ in cv_indices_list]]
    test_folds = [[test_index for _, test_index in cv_indices_list]]

    # Configure cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        folds=test_folds,
        metrics={'auc'},
        maximize=True,
        verbose_eval=False,
        shuffle=False,
        seed=0
    )

    print(cv_results)

    mean_auc = cv_results['test-auc-mean'].iloc[-1]
    return mean_auc