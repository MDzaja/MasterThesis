import sys
sys.path.insert(0, '../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_weights(prices: pd.Series, labels: pd.Series) -> pd.Series:
    # Ensure prices are filtered to match labels' indices
    prices = prices[prices.index.isin(labels.index)]
    weights = pd.Series(np.NaN, index=labels.index)
    
    trend_start_i = 0  # Start index of the trend

    for i, label in enumerate(labels.iloc[1:], start=1):
        if label != labels.iloc[trend_start_i] or i == len(labels) - 1:
            # End of the trend interval
            trend_end_i = i - 1 if label != labels.iloc[trend_start_i] else i
            # Calculate the absolute return for the trend interval
            absolute_return = abs(prices.iloc[trend_end_i+1] / prices.iloc[trend_start_i] - 1)
            for j in range(trend_start_i, trend_end_i + 1):
                weights.iloc[j] = absolute_return
            trend_start_i = i  # Update the start of the new trend

    weights = weights / weights.sum() * weights.shape[0]

    return weights
