import pandas as pd
import talib


def compute(stock_data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling asset technical indicators for given stock data using TA-Lib.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data.
    window_size (int): Look-back window size for calculating technical indicators.

    Returns:
    pd.DataFrame: DataFrame with rolling asset technical indicators.
    """

    # Technical Indicators
    # Using TA-Lib to calculate each technical indicator.
    technical_indicators = pd.DataFrame(index=stock_data.index)

    # Average Directional Index (ADX)
    technical_indicators['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            timeperiod=window_size)

    # Absolute Price Oscillator (APO)
    technical_indicators['APO'] = talib.APO(stock_data['Close'], fastperiod=window_size // 2, slowperiod=window_size,
                                            matype=0)

    # Commodity Channel Index (CCI)
    technical_indicators['CCI'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            timeperiod=window_size)

    # Directional Movement Index (DX)
    technical_indicators['DX'] = talib.DX(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                          timeperiod=window_size)

    # Money Flow Index (MFI)
    technical_indicators['MFI'] = talib.MFI(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                            stock_data['Volume'], timeperiod=window_size)

    # Relative Strength Index (RSI)
    technical_indicators['RSI'] = talib.RSI(stock_data['Close'], timeperiod=window_size)

    # Ultimate Oscillator (ULTOSC)
    technical_indicators['ULTOSC'] = talib.ULTOSC(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                                  timeperiod1=window_size // 2, timeperiod2=window_size,
                                                  timeperiod3=window_size * 2)

    # Williams' %R (WILLR)
    technical_indicators['WILLR'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                                timeperiod=window_size)

    # Normalized Average True Range (NATR)
    technical_indicators['NATR'] = talib.NATR(stock_data['High'], stock_data['Low'], stock_data['Close'],
                                              timeperiod=window_size)

    return technical_indicators