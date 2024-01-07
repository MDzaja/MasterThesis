import pandas as pd
import statsmodels.api as sm


def compute(stock_data: pd.DataFrame, market_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling asset-to-market index features for given stock data and market data.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data with 'Close' prices.
    market_data (pd.DataFrame): DataFrame containing market index data with 'Close' prices.
    window_size (int): Look-back window size (L) for calculating features.

    Returns:
    pd.DataFrame: DataFrame with rolling asset-to-market index features.
    """

    # Calculate returns for both stock and market index
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()

    # Initialize DataFrame to hold the features
    asset_to_market_features = pd.DataFrame(index=stock_data.index)

    # Asset-To-Index Alpha and Beta (Capital Asset Pricing Model parameters)
    # Rolling window regression to calculate alpha and beta
    for i in range(window_size, len(stock_data)):
        # Get the windowed data
        stock_window = stock_returns[i-window_size+1:i+1]
        market_window = market_returns[i-window_size+1:i+1]

        if len(stock_window) < window_size:  # Skip if there isn't enough data
            continue

        # Add a constant for OLS regression
        market_window = sm.add_constant(market_window)

        # Perform the regression
        model = sm.OLS(stock_window, market_window).fit()

        # Store the alpha and beta
        asset_to_market_features.at[stock_data.index[i], 'Alpha'] = model.params['const']
        asset_to_market_features.at[stock_data.index[i], 'Beta'] = model.params['Close']

    # Index Returns
    asset_to_market_features['Index_Returns'] = market_returns

    # Index Volatility: Standard deviation of market returns
    asset_to_market_features['Index_Volatility'] = market_returns.rolling(window=window_size).std()

    # Pearson Correlation Coefficient between stock and market returns
    asset_to_market_features['Correlation'] = stock_returns.rolling(window=window_size).corr(market_returns)

    # Return Distribution Dependencies: Rolling Covariance and Variance
    asset_to_market_features['Covariance'] = stock_returns.rolling(window=window_size).cov(market_returns)
    market_variance = market_returns.rolling(window=window_size).var()

    # Adjusted Close for both stock and market to compute Beta without using rolling regression
    adjusted_stock_close = stock_data['Close'] / stock_data['Close'].iloc[0]
    adjusted_market_close = market_data['Close'] / market_data['Close'].iloc[0]
    asset_to_market_features['Beta_Direct'] = adjusted_stock_close.rolling(window=window_size).cov(adjusted_market_close) / market_variance

    return asset_to_market_features