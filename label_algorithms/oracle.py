import pandas as pd

def binary_trend_labels(prices: pd.Series, fee: float=0) -> pd.Series:
    T = len(prices)
    
    # Initialize state matrix S and label array y
    S = [[0 for _ in range(T)] for _ in range(2)]
    y = [None] * T
    
    # Compute transition cost matrix P
    P = [[[0 for _ in range(T)] for _ in range(2)] for _ in range(2)]
    for t in range(1, T):
        tmp = -prices.iloc[t] * fee
        P[0][1][t] = tmp
        P[1][0][t] = tmp
        P[1][1][t] = prices.iloc[t] - prices.iloc[t-1]
    
    # Forward pass
    for t in range(2, T):  # loop from 2 to T
        S[0][t] = max(S[0][t-1] + P[0][0][t-1], S[1][t-1] + P[1][0][t-1])
        S[1][t] = max(S[0][t-1] + P[0][1][t-1], S[1][t-1] + P[1][1][t-1])
    
    # Initialize last label based on which state has maximum return at T
    kappa = 1 if S[1][T-1] > S[0][T-1] else 0
    
    # Backward pass to determine labels
    for t in range(T-1, 0, -1):  # loop from T-1 to 1
        idx = max(0, 1, key=lambda i:S[i][t] + P[i][kappa][t])
        y[t] = idx
        kappa = idx
    
    # Ensure every last sample of trend is labeled with the opposite label
    for t in range(1, T-1):
        if y[t] != y[t+1] and y[t] == y[t-1]:
            y[t] = 1 - y[t]

    y = pd.Series(y, index=prices.index).dropna()
    return y
