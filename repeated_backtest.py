import json
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
from backtesting import Backtest
import os
import numpy as np

from backtest import strategies
from models import utils as model_utils


METRIC_PATH = '/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/intraday_feat/metrics.json'
HTML_DIR = '/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/intraday_feat/backtests_repeated'

test_data = pd.read_csv("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/feat/test_1m_2013-10-09_2013-10-22.csv", index_col=0)
test_data.index = pd.to_datetime(test_data.index)

with open("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/labels/all_labels_test_1m_2013-10-09_2013-10-22.pkl", 'rb') as file:
    labels_test_dict = pickle.load(file)
Y_test = model_utils.get_Y_or_W_day_separated(labels_test_dict['oracle'], 60)

with open(METRIC_PATH, 'r') as file:
    metrics = json.load(file)

with open("/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/intraday_feat/test_probs.pkl", 'rb') as file:
    test_probs = pickle.load(file)

for combination, dict in test_probs.items():
    probs = dict['probs']
    probs_s = pd.Series(probs, index=Y_test.index)
    combined_index = probs_s.index.union(labels_test_dict['oracle'].index)
    probs_s = probs_s.reindex(combined_index, fill_value=0)

    PredictionsStrategy = strategies.PredictionsIntradayStrategyFactory(probs_s)
    
    bt = Backtest(test_data, PredictionsStrategy, cash=10000, commission=0.0004,
              trade_on_close=True, exclusive_orders=True)
    
    # Run the optimization
    opt_results = bt.optimize(threshold=list(np.arange(0.4, 0.8, 0.001)),
                            maximize='Return [%]')
    
    results = bt.run(
        threshold=opt_results._strategy.threshold,
    )

    cumulative_return = results['Return [%]'] / 100

    metrics[combination]['best_model_test']['cumulative_return_repeated'] = cumulative_return
    metrics[combination]['best_model_test']['threshold_repeated'] = opt_results._strategy.threshold

    save_plot_path = f'{HTML_DIR}/{combination}.html'
    if not os.path.exists(os.path.dirname(save_plot_path)):
        os.makedirs(os.path.dirname(save_plot_path))
    bt.plot(filename=save_plot_path, open_browser=False)

with open(METRIC_PATH, 'w') as file:
    json.dump(metrics, file, indent=3, default=model_utils.convert_types)
