import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np

import utils as lbl_utils
import ct_two_state as ct2
import ct_three_state as ct3
import fixed_time_horizon as fth
import oracle
import triple_barrier as tb

ticker_symbol = 'GC=F'
start_date = '2000-01-01'
end_date = '2023-11-01'

prices = yf.download(ticker_symbol, start_date, end_date, interval='1d')['Close']
prices.index = prices.index.tz_localize(None)

fee = 0.0004
num_threads = 16
file_store = '../artifacts/labels/params_detail_tb.txt'

# Triple Barrier
print('Optimizing Triple Barrier')
tEvents = prices.index
t1_list = []
for window in range(10, 150, 10):
    t1 = prices.index.searchsorted(tEvents + pd.Timedelta(days=window))
    t1 = pd.Series((prices.index[i] if i < prices.shape[0] else pd.NaT for i in t1), index=tEvents)
    t1_list.append(t1)
dayVolList = []
for span in range(40, 200, 20):
    dayVol = tb.getDayVol(prices, span=100)
    dayVol = dayVol.reindex(tEvents).loc[tEvents].fillna(method='bfill')
    dayVolList.append(dayVol)
param_grid = [
    [tEvents],                          # tEvents
    np.arange(0, 2, 0.1).tolist(),  #pt
    np.arange(0, 2, 0.1).tolist(),  #sl
    dayVolList,                        # volatility
    [0],                                # minRet
    t1_list,                               # t1
]
best_params = lbl_utils.optimize_label_params(binary_trend_labels=tb.binary_trend_labels, prices=prices,
                                              param_grid=param_grid, fee=fee, num_threads=num_threads)

best_dayVol = best_params[3]
best_t1 = best_params[5]

# Find the index in the respective lists
best_dayVol_index = dayVolList.index(best_dayVol)
best_t1_index = t1_list.index(best_t1)

# Calculate the span and window values from the indexes
best_span = 40 + (best_dayVol_index * 20)
best_window = 10 + (best_t1_index * 10)

print('TB; fee={}; pt={}; sl={}; vol_span={}; f1_window={}'.format(fee, best_params[1], best_params[2], best_span, best_window))
with open(file_store, 'a') as f:
    f.write('TB; fee={}; pt={}; sl={}; vol_span={}; f1_window={}\n'.format(fee, best_params[1], best_params[2], best_span, best_window))
