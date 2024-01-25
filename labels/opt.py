import sys
sys.path.insert(0, '../')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np

from labels import utils as lbl_utils
from labels import ct_two_state as ct2
from labels import ct_three_state as ct3
from labels import fixed_time_horizon as fth
from labels import triple_barrier as tb
from models import utils as model_utils

prices = model_utils.load_data('/home/mdzaja/MasterThesis/artifacts/assets/CL/data/feat/train_1m_2011-01-10_2012-11-06.csv')['Close']
prices.index = prices.index.tz_localize(None)

fee = 0.0004
num_threads = 16
file_store = '../artifacts/label_params/CL_new_minute_data.txt'

# CT2
print('Optimizing CT2')
param_grid = [
    np.arange(0, 0.003, 0.00002).tolist() # tau
]
best_params = lbl_utils.optimize_label_params(binary_trend_labels=ct2.binary_trend_labels, prices=prices
                                        , param_grid=param_grid, fee=fee, num_threads=num_threads)
with open(file_store, 'a') as f:
    f.write('CT2; fee={}; tau={}\n'.format(fee, best_params[0]))

# CT3
print('Optimizing CT3')
param_grid = [
    np.arange(0.002, 0.004, 0.00002).tolist(),  # tau
    np.arange(10, 20, 1).tolist()            # window
]
best_params = lbl_utils.optimize_label_params(binary_trend_labels=ct3.binary_trend_labels, prices=prices, 
                                              param_grid=param_grid, fee=fee, num_threads=num_threads)
with open(file_store, 'a') as f:
    f.write('CT3; fee={}; tau={}; window={}\n'.format(fee, best_params[0], best_params[1]))

# FTH
print('Optimizing FTH')
param_grid = [
    np.arange(0, 0.0003, 0.00002).tolist(),  # tau
    np.arange(1, 20, 1).tolist()            # H
]
best_params = lbl_utils.optimize_label_params(binary_trend_labels=fth.binary_trend_labels, prices=prices,
                                              param_grid=param_grid, fee=fee, num_threads=num_threads)
with open(file_store, 'a') as f:
    f.write('FTH; fee={}; tau={}; H={}\n'.format(fee, best_params[0], best_params[1]))

# Triple Barrier
# print('Optimizing Triple Barrier')
# tEvents = prices.index
# t1_list = []
# for window in range(30, 70, 10):
#     t1 = prices.index.searchsorted(tEvents + pd.Timedelta(days=window))
#     t1 = pd.Series((prices.index[i] if i < prices.shape[0] else pd.NaT for i in t1), index=tEvents)
#     t1_list.append(t1)
# volatilityList = []
# for span in range(2, 20, 2):
#     volatility = tb.getVolatility(prices, span=100)
#     volatility = volatility.reindex(tEvents).loc[tEvents].fillna(method='bfill')
#     volatilityList.append(volatility)
# param_grid = [
#     [tEvents],                          # tEvents
#     np.arange(0, 0.6, 0.1).tolist(),  #pt
#     np.arange(0, 0.6, 0.1).tolist(),  #sl
#     volatilityList,                        # volatility
#     [0],                                # minRet
#     t1_list,                               # t1
# ]
# best_params = lbl_utils.optimize_label_params(binary_trend_labels=tb.binary_trend_labels, prices=prices,
#                                               param_grid=param_grid, fee=fee, num_threads=num_threads)

# best_volatility = best_params[3]
# best_t1 = best_params[5]

# # Find the index in the respective lists
# best_volatility_index = next((i for i, x in enumerate(volatilityList) if x.equals(best_volatility)), None)
# best_t1_index = next((i for i, x in enumerate(t1_list) if x.equals(best_t1)), None)

# # Calculate the span and window values from the indexes
# best_span = 2 + (best_volatility_index * 2)
# best_window = 30 + (best_t1_index * 10)

# with open(file_store, 'a') as f:
#     f.write('TB; fee={}; pt={}; sl={}; vol_span={}; f1_window={}\n'.format(fee, best_params[1], best_params[2], best_span, best_window))
