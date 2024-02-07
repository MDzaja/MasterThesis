import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np

def get_weights(prices: pd.Series, labels: pd.Series) -> pd.Series:
    # filter out prices that are not in labels
    prices = prices[prices.index.isin(labels.index)]
    weights = pd.Series(np.NaN, index=labels.index)
    trend_start_i = 0

    for i, label in enumerate(labels.iloc[1:], start=1):
        if label != labels.iloc[trend_start_i] or i == len(labels) - 1:
            trend_end_i = i - 1 if label != labels.iloc[trend_start_i] else i
            for j in range(trend_start_i, trend_end_i + 1):
                weights.iloc[j] = abs(prices.iloc[trend_end_i+1] / prices.iloc[j] - 1)
            trend_start_i = i

    # Scale the weights to sum of weights.shape[0]
    weights = weights / weights.sum() * weights.shape[0]

    return weights