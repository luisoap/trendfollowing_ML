import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt
import talib as ta
import numpy as np


def read_crypto(crypto, start, end='today'):
    '''
    Function to load the data of the crypto currencies of your choice in the period of your choice.
    :param crypto: List of crypto-currencies. Example: ['BTC-USD','LTC-USD','BCH-USD','XRP-USD','ETH-USD']
    :param start: Date of start of your series. Example : '2010-01-01'.
    :param end: Date of end of your series. Default 'today'.
    :return: Returns a Dictionary with the keys being the name of the crypto currency and the values being a Dataframe.
    '''
    if end == 'today':
        end = dt.datetime.now().strftime("%Y-%m-%d")
    df = {}

    for i in crypto:
        df_crypto = pdr.DataReader(i, data_source='yahoo', start=start, end=end)

        df[i] = df_crypto
    return df

