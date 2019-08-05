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




def gen_data(df,crypto_string):
        
    data = df[crypto_string]
#    data = data.resample('W').last()
    data['High'] = data['High']
    data['Low'] = data['Low']
    data['Adj Close'] = data['Adj Close']
    data['RSI'] = ta.RSI(np.array(data['Adj Close']))
    data['MACD']= macd(data)
    data['ADX'] = ta.ADX(np.array(data['High']), np.array(data['Low']),np.array(data['Adj Close']))
    data['Return1']= data['Adj Close'].pct_change(1)
    data['Return4']= data['Adj Close'].pct_change(4)
    data['Return8']= data['Adj Close'].pct_change(8)
    
    
    data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],axis=1)
    
    
    
    signals = pd.DataFrame(index=data.index[1:], columns=['Signals'])


    for d in signals.index:
#        if np.isnan(data['Return1'].loc[d]): 
#            continue
        if data['Return1'].loc[d] >= 0:
            signals['Signals'].loc[d]= 1
        elif data['Return1'].loc[d] < 0:
            signals['Signals'].loc[d]= -1
  

    signals=signals[1:]
    X = data.shift(1).dropna()
    y = signals[data.index[0]:]
    
    return X, y



def macd(df):
       
    EMA_slow = df['Adj Close'].ewm(span=26).mean()
    EMA_fast = df['Adj Close'].ewm(span=12).mean()
    macd = EMA_fast - EMA_slow
    
    return macd