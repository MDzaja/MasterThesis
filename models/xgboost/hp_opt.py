import sys
sys.path.insert(0, '../')

from skopt.space import Real, Integer
from skopt import BayesSearchCV
import xgboost as xgb
import pandas as pd
import json
import numpy as np

from models import utils as model_utils


def xgb_hp_opt_cv(X, Y, W, directory, use_class_balancing, trial_num=100, n_splits=5):
    param_grid = {
        'n_estimators': Integer(low=50, high=2000, prior='uniform'),
        'max_depth': Integer(low=3, high=20, prior='uniform'),
        'gamma': Real(low=0.1, high=1, prior='uniform'),
        'learning_rate': Real(low=0.01, high=0.35, prior='uniform'),
        'reg_alpha': Real(low=0, high=1, prior='uniform'),
        'reg_lambda': Real(low=0, high=5, prior='uniform'),
        'min_child_weight': Integer(low=0, high=10, prior='uniform'),
        'subsample': Real(low=0.7, high=1.0, prior='uniform'),
    }

    X = X[:-135000]
    Y = Y.iloc[:-135000]
    W = W.iloc[:-135000]
        
    xgb_model = xgb.XGBClassifier()
    if use_class_balancing:
            class_weight_dict = model_utils.calculate_class_weight_dict(Y)
            W = model_utils.adjust_sample_weights(Y, class_weight_dict, W)

    cv_inidces_list = model_utils.custom_time_series_split(X, n_splits=n_splits)
    clf = BayesSearchCV(
            xgb_model,
            param_grid,
            n_iter=trial_num,
            cv=cv_inidces_list,
            scoring='roc_auc',
            random_state=42,
            n_jobs=1,
            verbose=3
        )
    fit_params ={
         'sample_weight': W
    }
    np.int = np.int64
    X = np.array(X)
    X = X.reshape(X.shape[0], -1).tolist()
    clf.fit(X, Y, **fit_params)

    with open(f'{directory}/best_hp.json', 'w') as fp:
        json.dump(clf.best_params_, fp)
    
    with open(f'{directory}/best_score.txt', 'w') as fp:
        fp.write(f'Best AUC score: {clf.best_score_*100}%\n')

    return clf.best_params_