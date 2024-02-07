import sys
sys.path.insert(0, '../')

import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from models import utils as model_utils
from backtest import utils as backtest_utils

def test_model(data_type, label_name, weight_name, model_name,
               data, Xs, Ys, Ws, hp_dict,
               use_class_balancing,
               n_splits,
               directory):
    
    hp_dict['n_jobs'] = -1
    hp_dict['objective'] = 'binary:logistic'
    hp_dict['eval_metric'] = 'auc'
    hp_dict['device'] = 'cuda'
    hp_dict['tree_method'] = 'gpu_hist'
    hp_dict['predictor'] = 'gpu_predictor'
    xgb_model = xgb.XGBClassifier(**hp_dict)

    for stage, _ in Xs.items():
        Xs[stage] = np.array(Xs[stage])
        Xs[stage] = Xs[stage].reshape(Xs[stage].shape[0], -1).tolist()

    if use_class_balancing:
        for stage, _ in Ws.items():
            class_weight_dict = model_utils.calculate_class_weight_dict(Ys[stage])
            Ws[stage] = model_utils.adjust_sample_weights(Ys[stage], class_weight_dict, Ws[stage])
    
    cv_inidces_list = model_utils.custom_time_series_split(Xs['train'], n_splits=n_splits)
    calibrator = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=cv_inidces_list, ensemble=True, n_jobs=1)
    calibrator.fit(Xs['train'], Ys['train'], sample_weight=Ws['train'])
    
    metrics = {}
    probs_dict = {}
    for stage in ['train', 'test']:
        save_backtest_plot_path = f"{directory}/backtests/{data_type}-{label_name}-{'CB_' if use_class_balancing else ''}{weight_name}-{model_name}-{stage}.html"

        probs_dict[stage] = {
            'label_name': label_name,
        }
        # Use the calibrated model for predictions
        probs_arr = calibrator.predict_proba(Xs[stage])[:, 1].flatten()
        probs_s = pd.Series(probs_arr, index=Ys[stage].index)
        combined_index = probs_s.index.union(data[stage].index)
        probs_s = probs_s.reindex(combined_index, fill_value=0)
        probs_dict[stage]['probs'] = probs_s

        bt_result = backtest_utils.do_backtest(data[stage], probs_dict[stage]['probs'], save_backtest_plot_path)

        metrics[stage] = {
            'accuracy': accuracy_score(Ys[stage], (probs_arr > 0.5).astype(int)),
            'f1': f1_score(Ys[stage], (probs_arr > 0.5).astype(int)),
            'mse': mean_squared_error(Ys[stage], probs_arr),
            'auc': roc_auc_score(Ys[stage], probs_arr),
            "cumulative_return": bt_result['Return [%]'] / 100,
            "exposure_time": bt_result['Exposure Time [%]'] / 100,
            "trades": bt_result['# Trades'],
            "duration": bt_result['Duration']
        }

    return metrics, probs_dict['train'], probs_dict['test']