import sys
sys.path.insert(0, '../')
import warnings
warnings.filterwarnings("ignore")

from backtesting import Backtest
from tensorflow.keras.models import load_model

from backtest import strategies
from models import utils as model_utils


# Parameters
WINDOW = 60
DATA_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/GC=F/data/raw/test_2021-12-06_2023-12-27.csv"
MODEL_PATH = "/home/mdzaja/MasterThesis/artifacts/models/test_logs/testing_tamplate/saved_models/raw-oracle-trend_interval_return-transformer.keras"
SAVE_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/GC=F/backtests/raw-oracle-trend_interval_return-transformer.html"


data = model_utils.load_data(DATA_PATH)
model = load_model(MODEL_PATH)
ModelBasedStrategy = strategies.ModelBasedStrategyFactory(model, WINDOW)

bt = Backtest(data,
              ModelBasedStrategy,
              cash=10000,
              commission=0.0004,
              trade_on_close=True,
              exclusive_orders=True)

results = bt.run()
print(results)
bt.plot(filename=SAVE_PATH, open_browser=False)

# Calculate and print the average annual return
total_return = results['Return [%]'] / 100
duration_years = (data.index[-1] - data.index[0]).days / 365.25
average_annual_return = (1 + total_return) ** (1 / duration_years) - 1
print("Average Annual Return:", average_annual_return * 100, "%")