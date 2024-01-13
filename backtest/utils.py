import sys
sys.path.insert(0, '../')
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest

from backtest import strategies

def do_backtest(data, model, window, save_plot_path=None):
    ModelBasedStrategy = strategies.ModelBasedStrategyFactory(model, window)
    bt = Backtest(data,
                  ModelBasedStrategy,
                  cash=10000,
                  commission=0.0004,
                  trade_on_close=True,
                  exclusive_orders=True)
    results = bt.run()
    if save_plot_path:
        bt.plot(filename=save_plot_path, open_browser=False)
    return results