import pandas as pd

#2-state labeling algorithm
def get_labels(prices: pd.Series, tau: float=0.15) -> pd.Series:
    first_price = prices[0]
    high_peak_p = prices[0]
    high_peak_t = 0
    low_peak_p = prices[0]
    low_peak_t = 0
    trend = 0
    first_peak_t = 0

    for i in range(1, len(prices)):
        if prices[i] > first_price + first_price*tau:
            high_peak_p = prices[i]
            high_peak_t = i
            first_peak_t = i
            trend = 1
            break
        if prices[i] > first_price - first_price*tau:
            low_peak_p = prices[i]
            low_peak_t = i
            first_peak_t = i
            trend = -1
            break
    
    labels = [0]*len(prices)
    for i in range(first_peak_t+1, len(prices)):
        if trend > 0:
            if prices[i] > high_peak_p:
                high_peak_p = prices[i]
                high_peak_t = i
            if prices[i] < high_peak_p-high_peak_p*tau and low_peak_t <= high_peak_t:
                for j in range(low_peak_t+1, high_peak_t+1):
                    labels[j] = 1
                low_peak_p = prices[i]
                low_peak_t = i
                trend = -1
        if trend < 0:
            if prices[i] < low_peak_p:
                low_peak_p = prices[i]
                low_peak_t = i
            if prices[i] > low_peak_p+low_peak_p*tau and high_peak_t <= low_peak_t:
                for j in range(high_peak_t+1, low_peak_t+1):
                    labels[j] = -1
                high_peak_p = prices[i]
                high_peak_t = i
                trend = 1

    last_peak = high_peak_t
    if trend > 0:
        last_peak = low_peak_t
    for i in range(last_peak+1, len(prices)):
        labels[i] = trend
    labels[0] = labels[1]
    
    labels_series = pd.Series(labels, index=prices.index)
    
    return labels_series
