import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np

def old_get_weights(prices: pd.Series, labels: pd.Series) -> pd.Series:
    # filter out prices that are not in labels
    prices = prices[prices.index.isin(labels.index)]

    weights = pd.Series(np.NaN, index=labels.index)

    # compute absolute change for first two prices, with abs
    abs_change = abs(prices.iloc[1] - prices.iloc[0])
    
    # Compute the weight for the first label
    weights.iloc[0] = abs(prices.iloc[1] / prices.iloc[0] - 1)
    
    trend_start_i = labels.index[0]
    prev_i = labels.index[0]
    # For each label, compute the weight
    for i, label in labels.iloc[1:].items():
        if label != labels[prev_i]:
            weights[i] = abs(prices[i] / prices[prev_i] - 1)
            trend_start_i = i
            prev_i = i
            continue
        weights[i] = abs(prices[i] / prices[trend_start_i] - 1)
        prev_i = i

    # Scale the weights to sum to weights.shape[0]
    weights = weights / weights.sum() * weights.shape[0]

    return weights

def get_weights(prices: pd.Series, labels: pd.Series) -> pd.Series:
    prices = prices[prices.index.isin(labels.index)]
    weights = pd.Series(np.NaN, index=labels.index)
    trend_start_i = labels.index[0]

    for i, label in enumerate(labels):
        if label != labels[trend_start_i]:
            trend_start_i = i
        weights[i] = abs(prices[i] / prices[trend_start_i] - 1)

    weights = weights / weights.sum() * weights.shape[0]

    return weights