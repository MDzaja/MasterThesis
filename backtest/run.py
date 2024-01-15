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
DATA_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/raw/test_1m_2021-12-20_2021-12-30.csv"
SCALE_REFERENCE_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/AAPL/data/raw/train_1m_2018-12-20_2021-12-17.csv"
MODEL_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/AAPL/models/test_logs/raw/saved_models/raw-ct_two_state-trend_interval_return-transformer.keras"
#SAVE_PATH = "/home/mdzaja/MasterThesis/artifacts/assets/GC=F/backtests/raw-oracle-trend_interval_return-transformer.html"


data = model_utils.load_data(DATA_PATH)
scale_reference_data = model_utils.load_data(SCALE_REFERENCE_PATH)
model = load_model(MODEL_PATH)
ModelBasedStrategy = strategies.ModelBasedIntradayStrategyFactory(model, WINDOW, scale_reference_data)

bt = Backtest(data,
              ModelBasedStrategy,
              cash=10000,
              commission=0.0004,
              trade_on_close=True,
              exclusive_orders=True)

results = bt.run()
print(results)
# bt.plot(filename=SAVE_PATH, open_browser=False)
