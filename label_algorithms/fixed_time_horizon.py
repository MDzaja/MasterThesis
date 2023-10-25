import pandas as pd

#fixed time horizon labeling algorithm
def get_labels(prices: pd.Series, tau: float=0.05, H: int=1) -> pd.Series:
    """
    Labels financial price series based on returns over a fixed horizon.

    Parameters:
    - price_series: pandas Series of prices.
    - tau: float, the threshold for the returns. Default is 0.05.
    - H: int, the number of periods over which returns are calculated. Default is 1.

    Returns:
    - labels: pandas Series of labels.
    """

    # Calculate returns over the fixed horizon
    returns = prices.pct_change(periods=H)
    returns = returns.dropna()

    # Labeling
    labels = returns.apply(lambda x: 1 if x >= tau else -1)

    return labels