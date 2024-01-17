import sys
sys.path.insert(0, '../')
import warnings
warnings.filterwarnings("ignore")

import os
from backtesting import Backtest
import pandas as pd
import numpy as np

from backtest import strategies

def do_backtest(data: pd.DataFrame, probs: pd.Series, save_plot_path=None):
    PredictionsIntradayStrategy = strategies.PredictionsIntradayStrategyFactory(probs)
    bt = Backtest(data,
                  PredictionsIntradayStrategy,
                  cash=10000,
                  commission=0.0004,
                  trade_on_close=True,
                  exclusive_orders=True)
    results = bt.run()
    if save_plot_path:
        if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))
        bt.plot(filename=save_plot_path, open_browser=False)
    return results


def do_backtest_w_optimization(data: pd.DataFrame, probs: pd.Series, save_plot_path=None):
    PredictionsIntradayStrategy = strategies.PredictionsIntradayStrategyFactory(probs)
    bt = Backtest(data,
                  PredictionsIntradayStrategy,
                  cash=10000,
                  commission=0.0004,
                  trade_on_close=True,
                  exclusive_orders=True)
    opt_results = bt.optimize(threshold=list(np.arange(0.4, 0.8, 0.001)),
                              maximize='Return [%]')
    results = bt.run(
        threshold=opt_results._strategy.threshold,
    )
    if save_plot_path:
        if not os.path.exists(os.path.dirname(save_plot_path)):
            os.makedirs(os.path.dirname(save_plot_path))
        bt.plot(filename=save_plot_path, open_browser=False)
    return results, opt_results._strategy.threshold