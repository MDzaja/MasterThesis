import pandas as pd
import numpy as np

def binary_trend_labels(prices: pd.Series, tEvents: pd.Series, pt: float, sl:float, volatility: float, minRet: float=0, t1: pd.Series=None, side: pd.Series=None) -> pd.Series:
    labels = get_labels(prices, tEvents, [pt, sl], volatility, minRet, t1, side)
    labels[labels == -1] = 0
    return labels

def get_labels(prices: pd.Series, tEvents: pd.Series, ptSl: list, volatility: float, minRet: float=0, t1: pd.Series=None, side: pd.Series=None) -> pd.Series:
    events = getEvents(prices, tEvents, ptSl, volatility, minRet, t1, side)
    labels = getBins(events, prices)['bin']
    return labels

def applyPtSlOnT1(prices: pd.Series, events: pd.DataFrame, ptSl: list) -> pd.Series:
    """
    Apply stop loss/profit taking, if it takes place before t1 (end of event)

    Parameters:
    - prices (pd.Series): A pandas series of prices.
    - events (pd.DataFrame): A dataframe with:
        * t1: The timestamp of the vertical barrier. NaN means no vertical barrier.
        * trgt: The unit width of the horizontal barriers.
        * side: Indicates the side of the bet (long/short as +1/-1 respectively).
    - pt_sl (list): A list of two non-negative float values:
        * pt_sl[0]: The factor that multiplies trgt to set the width of the upper barrier. If 0, there won't be an upper barrier.
        * pt_sl[1]: The factor that multiplies trgt to set the width of the lower barrier. If 0, there won't be a lower barrier.
    - molecule (list): A list with the subset of event indices that will be processed by a single thread.

    Returns:
    - out (pd.DataFrame): A dataframe with the timestamps at which each barrier was touched.
    """
    out = events[['t1']].copy(deep=True)
    
    if ptSl[0] > 0:
        pt = ptSl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs
    
    if ptSl[1] > 0:
        sl = -ptSl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs

    for loc, t1 in events['t1'].fillna(prices.index[-1]).items():
        df0 = prices[loc:t1]  # path prices
        df0 = (df0 / prices[loc] - 1) * events.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.

    return out

def getEvents(prices, tEvents, ptSl, trgt, minRet, t1=None, side=None):
    # Get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]  # minRet
    
    # Get time boundary t1
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=tEvents)
    t1 = t1.fillna(prices.index[-1])

    
    # Form events object, apply stop loss on t1
    if side is None: 
        side_ = pd.Series(1., index=trgt.index)
    else:
        side_ = side.loc[trgt.index]
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    df0 = applyPtSlOnT1(prices, events, ptSl=ptSl[:2])
    
    events['t1_old'] = events['t1']
    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan
    if side is None:
        events = events.drop('side', axis=1)
    
    return events

def getBins(events, prices):
    # Prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = prices.reindex(px, method='bfill')
    
    # Create output object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    if 'side' in events_: # meta-labeling
        out['ret']*=events_['side']
    out['bin'] = np.where(events_['t1_old'] == events_['t1'], 0, np.sign(out['ret']))
    if 'side' in events_: # meta-labeling
        out.loc[out['ret']<=0,'bin']=0
    
    return out

def getDayVol(prices, span=100):
    # Calculate the minute returns
    df0 = prices.index.searchsorted(prices.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    
    # Differences in the indices
    df0 = pd.Series(prices.index[df0 - 1], index=prices.index[prices.shape[0] - df0.shape[0]:])
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1 # minute returns

    # Calculate the volatility
    df0 = df0.ewm(span=span).std()

    return df0