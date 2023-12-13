import pandas as pd
import numpy as np

def get_weights(prices: pd.Series, labels: pd.Series) -> pd.Series:
    # filter out prices that are not in labels
    prices = prices[prices.index.isin(labels.index)]

    # Calculate the relative changes in prices
    relative_changes = abs(prices.pct_change().dropna())

    # Assign these values to the weights Series
    weights = pd.Series(np.NaN, index=labels.index)
    weights[relative_changes.index] = relative_changes.values

    # Scale the weights to sum to weights.shape[0]
    weights = weights / weights.sum() * weights.shape[0]

    return weights