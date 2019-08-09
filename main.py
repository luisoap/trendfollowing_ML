from sklearn.neighbors import KNeighborsClassifier
import pandas_datareader.data as pdr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.svm import SVC
import datetime as dt
from sklearn.metrics import accuracy_score
# import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
import data_crypto as dc
from port_construction import HRP, IVP, MinVar

# Executa os arquivos .py com os comandos necessários para rodar os diferentes modelos
# Dentro de cada um desses arquivos que seram rodados com a função exec(),
# puxamos os dados das cryptocurrencies utilizando
# a biblioteca 'data_crypto' criada por nós, todos os valores sendo em USD.


# Abaixo estamos fazendo o download dos dados de 13 weeks t-bill, baixando-os do site Yahoo Finance.
# Taxa livre de risco semanal.

end = dt.datetime.now().strftime("%Y-%m-%d")
df_tbill = pdr.DataReader('^IRX', data_source='yahoo', start='2010-01-01', end=end)
df_livre_risco = pd.DataFrame(index=df_tbill.index, columns=["IRX Anual", "IRX Semanal"])

for i in df_tbill.index:
    df_livre_risco["IRX Anual"].loc[i] = df_tbill["Close"].loc[i] / 100

df_livre_risco = df_livre_risco.resample("W").last()

for i in df_livre_risco.index:
    df_livre_risco["IRX Semanal"].loc[i] = (1 + df_livre_risco["IRX Anual"].loc[i]) ** (1 / 52) - 1

# Custo de transação 0.25 bps, sendo que 1 bps é 0.1%

CT = 0.0001 * 0.25

# índice de mercado

df_sp500 = pdr.DataReader('^GSPC', data_source='yahoo', start='2010-01-01', end=end)

df_sp500 = df_sp500.resample('W').last()

df_mercado = pd.DataFrame(index=df_sp500.index, columns=["Return"])

df_mercado['Return'] = df_sp500['Adj Close'].pct_change(1)

df_mercado['Index'] = np.nan

df_mercado = df_mercado.loc['2015-05-03':]

df_mercado['Index'].iloc[0] = 100

for d, dm1 in zip(df_mercado.index[1:], df_mercado.index[:-1]):
    df_mercado['Index'].loc[d] = df_mercado['Index'].loc[dm1] * (1 + df_mercado['Return'].loc[d])


########################################################################################################################

# KNN Model

########################################################################################################################

# Habilita rodar os diferentes métodos de construção de portfólios

# Lista de crypto currencies

crypto_list = ['BTC-USD', 'LTC-USD', 'XRP-USD', 'ETH-USD', 'XMR-USD', 'DASH-USD', 'DCR-USD']
start = '2010-01-01'
# Usando a função da biblioteca "data_crypto".
df = dc.read_crypto(crypto_list, start)

# Como vamos gerar estratégias semanais, passamos os dados para frequência semanal
for crypto_string in crypto_list:
    df[crypto_string] = df[crypto_string].resample('W').last()

# Cria os data Frames que serão usados posteriormente para armazenar os dados finais
returns_knn = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
summary_final_knn = pd.DataFrame(index=crypto_list, columns=['Accuracy', 'Sharp', 'Mean Return'])
scores_final_knn = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_knn = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_final_knn = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
return_final_knn = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
df_returns = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)

# Cria data frame de retornos semanais realizados
for crypto in crypto_list:
    df_returns[crypto] = df[crypto]['Adj Close'].pct_change(1)


# cria função MACD
def macd(_df):
    ema_slow = _df['Adj Close'].ewm(span=26).mean()
    ema_fast = _df['Adj Close'].ewm(span=12).mean()
    _macd = ema_fast - ema_slow

    return _macd


# Para cada cryptomoeda, vai gerar a matriz "X" de dados usados para gerar as previsões fora da amostra
# para os sinais de compra ou venda

for crypto_string in crypto_list:
    data = df[crypto_string]
    data.index = pd.to_datetime(data.index)

    data['High'] = data['High']
    data['Low'] = data['Low']
    data['Adj Close'] = data['Adj Close']
    data['MACD'] = macd(data)
    # a biblioteca Talib pode dar erro se você não a instalou corretamente
    # Para instalar em Windows,
    # siga o tutorial: https://medium.com/@keng16302/how-to-install-ta-lib-in-python-on-window-9303eb003fbb
    # Caso não consiga de jeito nenhum instalar a biblioteca Talib, uma última opção é "deletar" as duas linhas abaixo,
    # o que deve reduzir bastante a acurácia do modelo
    #     data['RSI'] = ta.RSI(np.array(data['Adj Close']))
    #     data['ADX'] = ta.ADX(np.array(data['High']), np.array(data['Low']),np.array(data['Adj Close']))

    data['Return1'] = data['Adj Close'].pct_change(1)
    data['Return2'] = data['Adj Close'].pct_change(2)
    data['Return4'] = data['Adj Close'].pct_change(3)

    data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], axis=1)
    data = data.loc[~data.index.duplicated(keep='first')]

    signals = pd.DataFrame(index=data.index[1:], columns=['Signals'])

    # Gera uma coluna com os sinais realizados para serem usados como variável Y do modelo
    for d in signals.index:

        if data['Return1'].loc[d] >= 0:
            signals['Signals'].loc[d] = 1
        elif data['Return1'].loc[d] < 0:
            signals['Signals'].loc[d] = -1

    X = data.shift(1).dropna()
    y = signals[X.index[0]:]

    # Vamos permitir iterações polinomiais de segunda ordem entre as variáveis explicativas dos sinais,
    # para tornar o modelo menos linear
    polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = polynomial_interaction.fit_transform(X)
    X = pd.DataFrame(data=X_poly, index=X.index)

    # Gera Componentes Principais
    # Note que quando n_components =0.97, equivale a cuspir o número de componentes principais de tal forma que
    # 97% da variância seja explicada. Portanto, o número de componentes será diferente para cada cryptomoeda,
    # para cada modelo e para cada janela temporal...é variante no tempo!

    pca = PCA(n_components=0.97, whiten=True)

    # Quebro a amostra em 50% para teste (que será o backtest final), 40% para treino e 10% para validação
    split_test = 0.5
    split_train_val = 0.5
    split_train = 0.4

    test = int(split_test * len(X))
    train = int(split_train * len(X))
    train_val = int(split_train_val * len(X))
    val = train_val - train

    # Defino intervalos temporais nos quais irei "Tunar" os hyperparâmetros do modelo
    # Idealmente, o correto é fazer isto para todos os períodos de tempo para tornar o modelo mais flexível e
    # aumentar a taxa de aprendizado do modelo, mas como isto gera custo computacional, coloco como padrão
    # uma atualização ótima dos hyperparâmetros a cada 4 semanas ( um mês)
    window_lengths = list(range(train_val + 1, len(X), 4))

    scores = []
    param = []
    pred = []
    true_values = []

    # Aqui faço um "expanding sample": A cada novo período, expando um ponto a amostra de treino,
    # desloco um ponto para frente a amostra de validação
    # e um ponto a observação de teste (mantendo o tamnho da amostra de validação e teste constantes,
    # apenas a amostra de treino que cresce com o tempo)
    # Este método visa replicar a estratégia real que teria sido feita caso replicássemos no passado o procedimento,
    # semana após semana, até os dias atuais
    # Este tipo de abordagem é usual na literatura de forecasting e de validação de modelos out-of-sample
    # O procedimento segue o paper "Empirical Asset Pricing via Machine Learning (2019) "
    # Equivale a fazer previsão fora da amostra 1 passo à frente, com validação mensal dos hyperparâmetros e treino
    # semanal dos parâmetros

    for i in range(train_val + 1, len(X)):
        Xs = X.loc[:X.index[i]]
        # Padroniza as variáveis para terem média zero e variância 1, de tal forma que o PCA não seja guiado por
        # variáveis cuja variância se sobrepõe às outras variáveis
        Xs = StandardScaler().fit_transform(Xs)
        # Aplica PCA para amostra disponível no período, ou seja, treino + validação
        Xs = pd.DataFrame(data=pca.fit_transform(Xs), index=X.index[:i + 1])

        # Define as amostras de treino, validação e teste
        x_train_val = Xs.loc[:Xs.index[-2]]
        x_val = Xs.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        x_train = Xs.loc[:x_val.index[0] + datetime.timedelta(-1)]
        x_test = Xs.loc[Xs.index[i]:Xs.index[i]]

        y_train_val = y.loc[:Xs.index[i - 1]]
        y_val = y.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        y_train = y.loc[:x_val.index[0] + datetime.timedelta(-1)]
        y_test = y.loc[Xs.index[i]:Xs.index[i]]

        best_parameter = 1

        if i in window_lengths:

            best_score = 0

            # A cada 4 semanas, faz o procedimento de "Tuning Hyperparameters "
            # Seleciona na amostra de validação o modelo que gera melhor acurácia fora da amostra para diferentes
            # hyperparametros
            for k in range(1, 13, 2):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(x_train, np.ravel(y_train, order='C'))
                score = knn.score(x_val, np.ravel(y_val, order='C'))

                if score > best_score:
                    best_score = score
                    best_parameter = k

            # Escolhido o melhor modelo, faça fit dos dados in-sample com a amostra de treino + validação
            # Gere out-of-sample forecasting um passo à frente para gerar o sinal prevista para o portfólio da
            # semana seguinte
            scores_final_knn.set_value(x_test.index, crypto_string, best_score)
            model = KNeighborsClassifier(n_neighbors=best_parameter, n_jobs=2)
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            signal_knn.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            pred.append(model.predict(x_test)[0])
            true_values.append(y_test['Signals'][0])
            param.append(best_parameter)

        else:
            # Se não cair na semana de fazer o Tuning hyperparamenter, fite os dados normalmente e gere previsões
            # Treinando o mesmo modelo usado anteriormente (melhor modelo escolhido algumas semanas atrás)  na amostra
            # treino + validação
            # Gera forecasting um passo à frente fora da amostra
            model = KNeighborsClassifier(n_neighbors=best_parameter)
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            signal_knn.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            pred.append(model.predict(x_test)[0])
            model.fit(x_train, np.ravel(y_train, order='C'))
            scores_final_knn.set_value(x_test.index, crypto_string, model.score(x_val, np.ravel(y_val, order='C')))

            true_values.append(y_test['Signals'][0])
            param.append(best_parameter)

    accuracy = accuracy_score(true_values, pred)

    # gera um data frame com os retornos advindos dos sinais gerados
    # Se o sinal era de compra (1) retornos são usuais
    # Se sinal era de venda (-1), retornos são invertidos

    for d in signal_knn.index:
        if signal_knn[crypto_string].loc[d] == 1:
            returns_knn.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[d] / df[crypto_string]['Adj Close'].loc[:d].iloc[-2]) - 1 -
                                  df_livre_risco['IRX Semanal'].loc[d])
        elif signal_knn[crypto_string].loc[d] == -1:
            returns_knn.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[:d].iloc[-2] / df[crypto_string]['Adj Close'].loc[d]) - 1 -
                                  df_livre_risco['IRX Semanal'].loc[d])

    strat_index = pd.DataFrame(index=X.index[train_val:], columns=['Returns', 'Level'])

    strat_index['Level'].iloc[0] = 100
    strat_index['Returns'] = returns_knn[crypto_string]

    for d, dm1 in zip(strat_index.index[1:], strat_index.index[:-1]):
        strat_index['Level'].loc[d] = strat_index['Level'].loc[dm1] * (1 + strat_index['Returns'].loc[d])

    ret_y = strat_index['Level'].pct_change(52)
    # ret_y.plot()
    vol_y = strat_index['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

    SR = ret_y / vol_y
    # SR.plot()

    SR_mean = (strat_index['Level'].pct_change(1).mean() * 52) / (
                strat_index['Level'].pct_change(1).std() * np.sqrt(52))

    summary_final_knn['Accuracy'].loc[crypto_string] = accuracy
    summary_final_knn['Sharp'].loc[crypto_string] = SR_mean
    summary_final_knn['Mean Return'].loc[crypto_string] = strat_index['Returns'].mean() * 52

    for d in strat_index.index:
        return_final_knn.set_value(d, crypto_string, strat_index['Level'].loc[d])
#        signal_final_knn.set_value(d,crypto_string, signal_knn[crypto_string].loc[d])


print(summary_final_knn)
return_final_knn.dropna(axis=0, how='all').plot()

# A seguir, são gerados portfolios por diferentes métodos : EW, Otimização por mínima variância, IVP e CLusterização
# Cada método puxa as classes geradas no arquivo "port_construction.py" dentro do projeto

# Note que como fazemos balanceamente semanal, a cada semana geramos novas otimizações e tudo irá depender do
# número de cryptomoedas disponíveis até o momento inicial da amostra de sinais gerados.


######################

# Roda Equally Weighted Portfolios

signals_ew = signal_knn.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')

ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)

soma = abs(signal_knn.dropna(axis=0, how='all')).sum(axis=1)

for d in ew_weights.index:
    for tracker in ew_weights.columns:
        if signals_ew[tracker].loc[d] == 0:
            ew_weights.set_value(d, tracker, 0)
        else:
            ew_weights.set_value(d, tracker, 1 / soma.loc[d])

    #    weights_ew = ew_weights
returns_ew = returns_knn.dropna(axis=0, how='all').fillna(0)  # [1:]

strat_index_knn_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 'Level': np.nan})

strat_index_knn_ew['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_knn_ew.index[1:], strat_index_knn_ew.index[:-1]):
    strat_index_knn_ew['Level'].loc[d] = strat_index_knn_ew['Level'].loc[dm1] *\
                                         (1 + strat_index_knn_ew['Return'].loc[d])

strat_index_knn_ew['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_knn_ew = strat_index_knn_ew['Level'].pct_change(52)
# ret_y.plot()
vol_y_knn_ew = strat_index_knn_ew['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_knn_ew = ret_y_knn_ew / vol_y_knn_ew
# SR.plot()

SR_mean_knn_ew = (strat_index_knn_ew['Level'].pct_change(1).mean() * 52) /\
                 (strat_index_knn_ew['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_knn_ew)

######################################

# Roda IVP

signals_ivp = signal_knn.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')
ivp_weights = pd.DataFrame(data=0, index=signals_ivp.index, columns=signals_ivp.columns)
soma = abs(signal_knn.dropna(axis=0, how='all')).sum(axis=1)

for d in ivp_weights.index:
    if soma.loc[d] == 1:
        ivp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = ivp_weights['BTC-USD'].sum()

for d in tqdm(ivp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ivp = IVP(ret)
    w = ivp.weights
    for tracker in tqdm(ret.columns):
        ivp_weights[tracker].loc[d] = w[tracker]
#            ivp_weights.set_value(d, tracker, w[tracker])

returns_ivp = returns_knn.dropna(axis=0, how='all').fillna(0)

strat_index_knn_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_knn_ivp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_knn_ivp.index[1:], strat_index_knn_ivp.index[:-1]):
    strat_index_knn_ivp['Level'].loc[d] = strat_index_knn_ivp['Level'].loc[dm1] *\
                                          (1 + strat_index_knn_ivp['Return'].loc[d])

strat_index_knn_ivp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_knn_ivp = strat_index_knn_ivp['Level'].pct_change(52)
vol_y_knn_ivp = strat_index_knn_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_knn_ivp = ret_y_knn_ivp / vol_y_knn_ivp

SR_mean_knn_ivp = (strat_index_knn_ivp['Level'].pct_change(1).mean() * 52) /\
                  (strat_index_knn_ivp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_knn_ivp)

######################################

# RODA MVO

signals_mvp = signal_knn.dropna(axis=0, how='all').fillna(0)
return_mvp = returns_knn.dropna(axis=0, how='all').fillna(0)
mvp_weights = pd.DataFrame(data=0, index=signals_mvp.index, columns=signals_mvp.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        mvp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = mvp_weights['BTC-USD'].sum()

for d in tqdm(mvp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    mvp = MinVar(ret)
    w = mvp.weights
    for tracker in tqdm(mvp_weights.columns):
        mvp_weights[tracker].loc[d] = w[tracker]

strat_index_knn_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_knn_mvp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_knn_mvp.index[1:], strat_index_knn_mvp.index[:-1]):
    strat_index_knn_mvp['Level'].loc[d] = strat_index_knn_mvp['Level'].loc[dm1] *\
                                          (1 + strat_index_knn_mvp['Return'].loc[d])

strat_index_knn_mvp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_knn_mvp = strat_index_knn_mvp['Level'].pct_change(52)
vol_y_knn_mvp = strat_index_knn_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_knn_mvp = ret_y_knn_mvp / vol_y_knn_mvp

SR_mean_knn_mvp = (strat_index_knn_mvp['Level'].pct_change(1).mean() * 52) /\
                  (strat_index_knn_mvp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_knn_mvp)

#########################################

######################
# PESOS por HYERARCHICAL CLUSTERING #######

signals_cluster = signal_knn.dropna(axis=0, how='all').fillna(0)
return_cluster = returns_knn.dropna(axis=0, how='all').fillna(0)
hrp_weights = pd.DataFrame(data=0, index=signals_cluster.index, columns=signals_cluster.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        hrp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = hrp_weights['BTC-USD'].sum()

for d in tqdm(hrp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    hrp = HRP(ret)
    w = hrp.weights
    for tracker in tqdm(hrp_weights.columns):
        hrp_weights[tracker].loc[d] = w[tracker]

strat_index_knn_cluster = pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1),
                                             'Level': np.nan})

strat_index_knn_cluster['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_knn_cluster.index[1:], strat_index_knn_cluster.index[:-1]):
    strat_index_knn_cluster['Level'].loc[d] = strat_index_knn_cluster['Level'].loc[dm1] *\
                                              (1 + strat_index_knn_cluster['Return'].loc[d])

strat_index_knn_cluster['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_knn_cluster = strat_index_knn_cluster['Level'].pct_change(52)
vol_y_knn_cluster = strat_index_knn_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_knn_cluster = ret_y_knn_cluster / vol_y_knn_cluster

SR_mean_knn_cluster = (strat_index_knn_cluster['Level'].pct_change(1).mean() * 52) /\
                      (strat_index_knn_cluster['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_knn_cluster)


########################################################################################################################

# Logistic Model

########################################################################################################################


# Lista de crypto currencies

crypto_list = ['BTC-USD', 'LTC-USD', 'XRP-USD', 'ETH-USD', 'XMR-USD', 'DASH-USD', 'DCR-USD']  # 'BCH-USD',
start = '2010-01-01'
# Usando a função da biblioteca "data_crypto".
df = dc.read_crypto(crypto_list, start)
# Como vamos gerar estratégias semanais, passamos os dados para frequência semanal
for crypto_string in crypto_list:
    df[crypto_string] = df[crypto_string].resample('W').last()

# Cria os data Frames que serão usados posteriormente para armazenar os dados finais

returns_logit = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
summary_final_logit = pd.DataFrame(index=crypto_list, columns=['Accuracy', 'Sharp', 'Mean Return'])
scores_final_logit = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_logit = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_final_logit = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
return_final_logit = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
df_returns = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)

# Cria data frame de retornos semanais realizados
for crypto in crypto_list:
    df_returns[crypto] = df[crypto]['Adj Close'].pct_change(1)


# cria função MACD
def macd(df1):
    ema_slow = df1['Adj Close'].ewm(span=26).mean()
    ema_fast = df1['Adj Close'].ewm(span=12).mean()
    macd1 = ema_fast - ema_slow

    return macd1


# Para cada cryptomoeda, vai gerar a matriz "X" de dados usados para gerar as previsões fora da amostra
# para os sinais de compra ou venda
for crypto_string in crypto_list:

    data = df[crypto_string]
    data.index = pd.to_datetime(data.index)

    data['High'] = data['High']
    data['Low'] = data['Low']
    data['Adj Close'] = data['Adj Close']

    data['MACD'] = macd(data)
    # a biblioteca Talib pode dar erro se você não a instalou corretamente
    # Para instalar em Windows, siga o tutorial:
    # https://medium.com/@keng16302/how-to-install-ta-lib-in-python-on-window-9303eb003fbb

    # Caso não consiga de jeito nenhum instalar a biblioteca Talib, uma última opção é "deletar" as duas linhas abaixo,
    # o que deve reduzir bastante a acurácia do modelo

    # data['RSI'] = ta.RSI(np.array(data['Adj Close']))
    # data['ADX'] = ta.ADX(np.array(data['High']), np.array(data['Low']),np.array(data['Adj Close']))

    data['Return1'] = data['Adj Close'].pct_change(1)
    data['Return2'] = data['Adj Close'].pct_change(2)
    data['Return3'] = data['Adj Close'].pct_change(3)

    data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], axis=1)
    data = data.loc[~data.index.duplicated(keep='first')]

    signals = pd.DataFrame(index=data.index[1:], columns=['Signals'])

    # Gera uma coluna com os sinais realizados para serem usados como variável Y do modelo
    for d in signals.index:

        if data['Return1'].loc[d] >= 0:
            signals['Signals'].loc[d] = 1
        elif data['Return1'].loc[d] < 0:
            signals['Signals'].loc[d] = -1

    X = data.shift(1).dropna()
    y = signals[X.index[0]:]

    # Vamos permitir iterações polinomiais de segunda ordem entre as variáveis explicativas dos sinais,
    # para tornar o modelo menos linear
    polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = polynomial_interaction.fit_transform(X)
    X = pd.DataFrame(data=X_poly, index=X.index)
    # Gera Componentes Principais
    # Note que quando n_components =0.97, equivale a cuspir o número de componentes principais de tal forma que
    # 97% da variância seja explicada. Portanto, o número de componentes será diferente para cada cryptomoeda,
    # para cada modelo e para cada janela temporal...é variante no tempo!

    pca = PCA(n_components=0.97, whiten=True)

    # Quebro a amostra em 50% para teste (que será o backtest final), 40% para treino e 10% para validação

    split_test = 0.5
    split_train_val = 0.5
    split_train = 0.4

    test = int(split_test * len(X))
    train = int(split_train * len(X))
    train_val = int(split_train_val * len(X))
    val = train_val - train

    # Defino intervalos temporais nos quais irei "Tunar" os hyperparâmetros do modelo
    # Idealmente, o correto é fazer isto para todos os períodos de tempo para tornar o modelo mais flexível e
    # aumentar a taxa de aprendizado do modelo, mas como isto gera custo computacional, coloco como padrão
    # uma atualização ótima dos hyperparâmetros a cada 4 semanas ( um mês)

    window_lengths = list(range(train_val + 1, len(X), 4))

    scores = []
    param = []
    pred = []
    true_values = []

    # Aqui faço um "expanding sample": A cada novo período, expando um ponto a amostra de treino,
    # desloco um ponto para frente a amostra de validação
    # e um ponto a observação de teste (mantendo o tamnho da amostra de validação e teste constantes,
    # apenas a amostra de treino que cresce com o tempo)
    # Este método visa replicar a estratégia real que teria sido feita caso replicássemos no passado o procedimento,
    # semana após semana, até os dias atuais
    # Este tipo de abordagem é usual na literatura de forecasting e de validação de modelos out-of-sample
    # O procedimento segue o paper "Empirical Asset Pricing via Machine Learning (2019) "
    # Equivale a fazer previsão fora da amostra 1 passo à frente, com validação mensal dos hyperparâmetros e treino
    # semanal dos parâmetros

    for i in range(train_val + 1, len(X)):
        Xs = X.loc[:X.index[i]]
        # Padroniza as variáveis para terem média zero e variância 1, de tal forma que o PCA não seja guiado por
        # variáveis cuja variância se sobrepõe às outras variáveis
        Xs = StandardScaler().fit_transform(Xs)
        # Aplica PCA para amostra disponível no período, ou seja, treino + validação
        Xs = pd.DataFrame(data=pca.fit_transform(Xs), index=X.index[:i + 1])
        # Define as amostras de treino, validação e teste
        x_train_val = Xs.loc[:Xs.index[-2]]

        x_val = Xs.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        x_train = Xs.loc[:x_val.index[0] + datetime.timedelta(-1)]
        x_test = Xs.loc[Xs.index[i]:Xs.index[i]]

        y_train_val = y.loc[:Xs.index[i - 1]]
        y_val = y.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        y_train = y.loc[:x_val.index[0] + datetime.timedelta(-1)]
        y_test = y.loc[Xs.index[i]:Xs.index[i]]

        if i in window_lengths:
            # parameter tuning
            best_score = 0
            # A cada 4 semanas, faz o procedimento de "Tuning Hyperparameters "
            # Seleciona na amostra de validação o modelo que gera melhor acurácia fora da amostra para diferentes
            # hyperparametros

            for c in [0.01, 0.1, 1, 10, 100]:
                for l in ['l1', 'l2', 'none']:
                    logit = LogisticRegression(penalty=l, C=c, solver='saga')
                    logit.fit(x_train, np.ravel(y_train, order='C'))
                    score = logit.score(x_val, np.ravel(y_val, order='C'))

                    if score > best_score:
                        best_score = score
                        best_parameters = {'C': c, 'Penalty': l}

            # Escolhido o melhor modelo, faça fit dos dados in-sample com a amostra de treino + validação
            # Gere out-of-sample forecasting um passo à frente para gerar o sinal prevista para o portfólio da semana
            # seguinte

            scores_final_logit.set_value(x_test.index, crypto_string, best_score)
            model = LogisticRegression(penalty=best_parameters['Penalty'], C=best_parameters['C'], solver='saga')
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            signal_logit.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            pred.append(model.predict(x_test)[0])
            true_values.append(y_test['Signals'][0])
            param.append(best_parameters)

        else:
            # Se não cair na semana de fazer o Tuning hyperparamenter, fite os dados normalmente e gere previsões
            # Treinando o mesmo modelo usado anteriormente (melhor modelo escolhido algumas semanas atrás)
            # na amostra treino + validação
            # Gera forecasting um passo à frente fora da amostra

            model = LogisticRegression(C=best_parameters['C'], penalty=best_parameters['Penalty'], solver='saga')
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            pred.append(model.predict(x_test)[0])
            signal_logit.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            model.fit(x_train, np.ravel(y_train, order='C'))
            scores_final_logit.set_value(x_test.index, crypto_string, model.score(x_val, np.ravel(y_val, order='C')))
            true_values.append(y_test['Signals'][0])

    accuracy = accuracy_score(true_values, pred)

    # gera um data frame com os retornos advindos dos sinais gerados
    # Se o sinal era de compra (1) retornos são usuais
    # Se sinal era de venda (-1), retornos são invertidos
    for d in signal_logit.index:
        if signal_logit[crypto_string].loc[d] == 1:
            returns_logit.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[d] / df[crypto_string]['Adj Close'].loc[:d].iloc[-2]) - 1 -
                                    df_livre_risco['IRX Semanal'].loc[d])
        elif signal_logit[crypto_string].loc[d] == -1:
            returns_logit.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[:d].iloc[-2] / df[crypto_string]['Adj Close'].loc[d]) - 1 -
                                    df_livre_risco['IRX Semanal'].loc[d])

    strat_index = pd.DataFrame(index=X.index[train_val:], columns=['Returns', 'Level'])

    strat_index['Level'].iloc[0] = 100
    strat_index['Returns'] = returns_logit[crypto_string]

    for d, dm1 in zip(strat_index.index[1:], strat_index.index[:-1]):
        strat_index['Level'].loc[d] = strat_index['Level'].loc[dm1] * (1 + strat_index['Returns'].loc[d])

    ret_y = strat_index['Level'].pct_change(52)
    # ret_y.plot()
    vol_y = strat_index['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

    SR = ret_y / vol_y
    # SR.plot()

    SR_mean = (strat_index['Level'].pct_change(1).mean() * 52) / (
                strat_index['Level'].pct_change(1).std() * np.sqrt(52))

    summary_final_logit['Accuracy'].loc[crypto_string] = accuracy
    summary_final_logit['Sharp'].loc[crypto_string] = SR_mean
    summary_final_logit['Mean Return'].loc[crypto_string] = strat_index['Returns'].mean() * 52

    for d in strat_index.index:
        return_final_logit.set_value(d, crypto_string, strat_index['Level'].loc[d])
#        signal_final_logit.set_value(d,crypto_string, signal_logit[crypto_string].loc[d])


print(summary_final_logit)
return_final_logit.dropna(axis=0, how='all').plot()

# A seguir, são gerados portfolios por diferentes métodos : EW, Otimização por mínima variância, IVP e CLusterização
# Cada método puxa as classes geradas no arquivo "port_construction.py" dentro do projeto

# Note que como fazemos balanceamente semanal, a cada semana geramos novas otimizações e tudo irá depender do
# número de cryptomoedas disponíveis até o momento inicial da amostra de sinais gerados.


######################

# Roda Equally Weighted ####


signals_ew = signal_logit.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')

ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)

soma = abs(signal_logit.dropna(axis=0, how='all')).sum(axis=1)

for d in ew_weights.index:
    for tracker in ew_weights.columns:
        if signals_ew[tracker].loc[d] == 0:
            ew_weights.set_value(d, tracker, 0)
        else:
            ew_weights.set_value(d, tracker, 1 / soma.loc[d])

#    weights_ew = ew_weights
returns_ew = returns_logit.dropna(axis=0, how='all').fillna(0)  # [1:]

strat_index_logit_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 'Level': np.nan})

strat_index_logit_ew['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_logit_ew.index[1:], strat_index_logit_ew.index[:-1]):
    strat_index_logit_ew['Level'].loc[d] = strat_index_logit_ew['Level'].loc[dm1] *\
                                           (1 + strat_index_logit_ew['Return'].loc[d])

strat_index_logit_ew['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_logit_ew = strat_index_logit_ew['Level'].pct_change(52)
# ret_y.plot()
vol_y_logit_ew = strat_index_logit_ew['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_logit_ew = ret_y_logit_ew / vol_y_logit_ew
# SR.plot()

SR_mean_logit_ew = (strat_index_logit_ew['Level'].pct_change(1).mean() * 52) /\
                   (strat_index_logit_ew['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_logit_ew)

######################################

# Roda IVP ###

signals_ivp = signal_logit.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')
ivp_weights = pd.DataFrame(data=0, index=signals_ivp.index, columns=signals_ivp.columns)
soma = abs(signal_logit.dropna(axis=0, how='all')).sum(axis=1)

for d in ivp_weights.index:
    if soma.loc[d] == 1:
        ivp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = ivp_weights['BTC-USD'].sum()

for d in tqdm(ivp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ivp = IVP(ret)
    w = ivp.weights
    for tracker in tqdm(ret.columns):
        ivp_weights[tracker].loc[d] = w[tracker]
#            ivp_weights.set_value(d, tracker, w[tracker])

returns_ivp = returns_logit.dropna(axis=0, how='all').fillna(0)

strat_index_logit_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_logit_ivp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_logit_ivp.index[1:], strat_index_logit_ivp.index[:-1]):
    strat_index_logit_ivp['Level'].loc[d] = strat_index_logit_ivp['Level'].loc[dm1] *\
                                            (1 + strat_index_logit_ivp['Return'].loc[d])

strat_index_logit_ivp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_logit_ivp = strat_index_logit_ivp['Level'].pct_change(52)
vol_y_logit_ivp = strat_index_logit_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_logit_ivp = ret_y_logit_ivp / vol_y_logit_ivp

SR_mean_logit_ivp = (strat_index_logit_ivp['Level'].pct_change(1).mean() * 52) /\
                    (strat_index_logit_ivp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_logit_ivp)

######################################

# RODA MVO ###

signals_mvp = signal_logit.dropna(axis=0, how='all').fillna(0)
return_mvp = returns_logit.dropna(axis=0, how='all').fillna(0)
mvp_weights = pd.DataFrame(data=0, index=signals_mvp.index, columns=signals_mvp.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        mvp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = mvp_weights['BTC-USD'].sum()

for d in tqdm(mvp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    mvp = MinVar(ret)
    w = mvp.weights
    for tracker in tqdm(mvp_weights.columns):
        mvp_weights[tracker].loc[d] = w[tracker]

strat_index_logit_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_logit_mvp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_logit_mvp.index[1:], strat_index_logit_mvp.index[:-1]):
    strat_index_logit_mvp['Level'].loc[d] = strat_index_logit_mvp['Level'].loc[dm1] *\
                                            (1 + strat_index_logit_mvp['Return'].loc[d])

strat_index_logit_mvp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_logit_mvp = strat_index_logit_mvp['Level'].pct_change(52)
vol_y_logit_mvp = strat_index_logit_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_logit_mvp = ret_y_logit_mvp / vol_y_logit_mvp

SR_mean_logit_mvp = (strat_index_logit_mvp['Level'].pct_change(1).mean() * 52) /\
                    (strat_index_logit_mvp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_logit_mvp)

#########################################

######################
# PESOS por HYERARCHICAL CLUSTERING #######

signals_cluster = signal_logit.dropna(axis=0, how='all').fillna(0)
return_cluster = returns_logit.dropna(axis=0, how='all').fillna(0)
hrp_weights = pd.DataFrame(data=0, index=signals_cluster.index, columns=signals_cluster.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        hrp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = hrp_weights['BTC-USD'].sum()

for d in tqdm(hrp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    hrp = HRP(ret)
    w = hrp.weights
    for tracker in tqdm(hrp_weights.columns):
        hrp_weights[tracker].loc[d] = w[tracker]

strat_index_logit_cluster = \
    pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1), 'Level': np.nan})

strat_index_logit_cluster['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_logit_cluster.index[1:], strat_index_logit_cluster.index[:-1]):
    strat_index_logit_cluster['Level'].loc[d] = strat_index_logit_cluster['Level'].loc[dm1] *\
                                                (1 + strat_index_logit_cluster['Return'].loc[d])

strat_index_logit_cluster['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_logit_cluster = strat_index_logit_cluster['Level'].pct_change(52)
vol_y_logit_cluster = strat_index_logit_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_logit_cluster = ret_y_logit_cluster / vol_y_logit_cluster

SR_mean_logit_cluster = (strat_index_logit_cluster['Level'].pct_change(1).mean() * 52) /\
                        (strat_index_logit_cluster['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_logit_cluster)

########################################################################################################################

# SVC model

########################################################################################################################

# Lista de crypto currencies

crypto_list = ['BTC-USD', 'LTC-USD', 'XRP-USD', 'ETH-USD', 'XMR-USD', 'DASH-USD', 'DCR-USD']  # 'BCH-USD',
start = '2010-01-01'
# Usando a função da biblioteca "data_crypto".
df = dc.read_crypto(crypto_list, start)
# Como vamos gerar estratégias semanais, passamos os dados para frequência semanal
for crypto_string in crypto_list:
    df[crypto_string] = df[crypto_string].resample('W').last()

# Cria os data Frames que serão usados posteriormente para armazenar os dados finais

returns_svc = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
summary_final_svc = pd.DataFrame(index=crypto_list, columns=['Accuracy', 'Sharp', 'Mean Return'])
scores_final_svc = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_svc = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
signal_final_svc = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
return_final_svc = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)
df_returns = pd.DataFrame(index=df['BTC-USD'].index, columns=crypto_list)

# Cria data frame de retornos semanais realizados
for crypto in crypto_list:
    df_returns[crypto] = df[crypto]['Adj Close'].pct_change(1)


# cria função MACD
def macd(df1):
    ema_slow = df1['Adj Close'].ewm(span=26).mean()
    ema_fast = df1['Adj Close'].ewm(span=12).mean()
    macd1 = ema_fast - ema_slow

    return macd1


# Para cada cryptomoeda, vai gerar a matriz "X" de dados usados para gerar as previsões fora da amostra
# para os sinais de compra ou venda
for crypto_string in crypto_list:

    data = df[crypto_string]
    data.index = pd.to_datetime(data.index)

    data['High'] = data['High']
    data['Low'] = data['Low']
    data['Adj Close'] = data['Adj Close']

    data['MACD'] = macd(data)
    # a biblioteca Talib pode dar erro se você não a instalou corretamente
    # Para instalar em Windows, siga o tutorial:
    # https://medium.com/@keng16302/how-to-install-ta-lib-in-python-on-window-9303eb003fbb

    # Caso não consiga de jeito nenhum instalar a biblioteca Talib,
    # uma última opção é "deletar" as duas linhas abaixo, o que deve reduzir bastante a acurácia do modelo
    #     data['ADX'] = ta.ADX(np.array(data['High']), np.array(data['Low']),np.array(data['Adj Close']))
    #     data['RSI'] = ta.RSI(np.array(data['Adj Close']))

    data['Return1'] = data['Adj Close'].pct_change(1)
    data['Return2'] = data['Adj Close'].pct_change(2)
    data['Return3'] = data['Adj Close'].pct_change(3)

    data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'], axis=1)
    data = data.loc[~data.index.duplicated(keep='first')]

    signals = pd.DataFrame(index=data.index[1:], columns=['Signals'])

    # Gera uma coluna com os sinais realizados para serem usados como variável Y do modelo

    for d in signals.index:

        if data['Return1'].loc[d] >= 0:
            signals['Signals'].loc[d] = 1
        elif data['Return1'].loc[d] < 0:
            signals['Signals'].loc[d] = -1

    X = data.shift(1).dropna()
    y = signals[X.index[0]:]
    # Vamos permitir iterações polinomiais de segunda ordem entre as variáveis explicativas dos sinais,
    # para tornar o modelo menos linear
    polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = polynomial_interaction.fit_transform(X)
    X = pd.DataFrame(data=X_poly, index=X.index)
    # Gera Componentes Principais
    # Note que quando n_components =0.97, equivale a cuspir o número de componentes principais de tal forma que
    # 97% da variância seja explicada. Portanto, o número de componentes será diferente para cada cryptomoeda,
    # para cada modelo e para cada janela temporal...é variante no tempo!

    pca = PCA(n_components=0.97, whiten=True)

    # Quebro a amostra em 50% para teste (que será o backtest final), 40% para treino e 10% para validação
    split_test = 0.5
    split_train_val = 0.5
    split_train = 0.4

    test = int(split_test * len(X))
    train = int(split_train * len(X))
    train_val = int(split_train_val * len(X))
    val = train_val - train

    # Defino intervalos temporais nos quais irei "Tunar" os hyperparâmetros do modelo
    # Idealmente, o correto é fazer isto para todos os períodos de tempo para tornar o modelo mais flexível e
    # aumentar a taxa de aprendizado do modelo, mas como isto gera custo computacional, coloco como padrão
    # uma atualização ótima dos hyperparâmetros a cada 4 semanas ( um mês)

    window_lengths = list(range(train_val + 1, len(X), 4))

    scores = []
    param = []
    pred = []
    true_values = []

    # Aqui faço um "expanding sample": A cada novo período, expando um ponto a amostra de treino,
    # desloco um ponto para frente a amostra de validação
    # e um ponto a observação de teste (mantendo o tamnho da amostra de validação e teste constantes,
    # apenas a amostra de treino que cresce com o tempo)
    # Este método visa replicar a estratégia real que teria sido feita caso replicássemos no passado o
    # procedimento, semana após semana, até os dias atuais
    # Este tipo de abordagem é usual na literatura de forecasting e de validação de modelos out-of-sample
    # O procedimento segue o paper "Empirical Asset Pricing via Machine Learning (2019) "
    # Equivale a fazer previsão fora da amostra 1 passo à frente, com validação mensal dos hyperparâmetros e
    # treino semanal dos parâmetros

    for i in range(train_val + 1, len(X)):
        Xs = X.loc[:X.index[i]]
        # Padroniza as variáveis para terem média zero e variância 1, de tal forma que o PCA não seja guiado por
        # variáveis cuja variância se sobrepõe às outras variáveis
        Xs = StandardScaler().fit_transform(Xs)
        # Aplica PCA para amostra disponível no período, ou seja, treino + validação

        Xs = pd.DataFrame(data=pca.fit_transform(Xs), index=X.index[:i + 1])
        # Define as amostras de treino, validação e teste
        x_train_val = Xs.loc[:Xs.index[-2]]

        x_val = Xs.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        x_train = Xs.loc[:x_val.index[0] + datetime.timedelta(-1)]
        x_test = Xs.loc[Xs.index[i]:Xs.index[i]]

        y_train_val = y.loc[:Xs.index[i - 1]]
        y_val = y.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        y_train = y.loc[:x_val.index[0] + datetime.timedelta(-1)]
        y_test = y.loc[Xs.index[i]:Xs.index[i]]

        if i in window_lengths:
            # parameter tuning
            best_score = 0
            # A cada 4 semanas, faz o procedimento de "Tuning Hyperparameters "
            # Seleciona na amostra de validação o modelo que gera melhor acurácia fora da amostra para diferentes
            # hyperparametros

            for models in ['rbf', 'linear']:
                for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
                    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
                        # for each combination of parameters, train an SVC
                        svm = SVC(gamma=gamma, C=c)
                        svm.fit(x_train, np.ravel(y_train, order='C'))
                        # evaluate the SVC on the test set
                        score = svm.score(x_val, np.ravel(y_val, order='C'))
                        # if we got a better score, store the score and parameters
                        if score > best_score:
                            best_score = score
                            best_parameters = {'C': c, 'gamma': gamma, 'Kernel': models}

            # Escolhido o melhor modelo, faça fit dos dados in-sample com a amostra de treino + validação
            # Gere out-of-sample forecasting um passo à frente para gerar o sinal prevista para o
            # portfólio da semana seguinte

            scores_final_svc.set_value(x_test.index, crypto_string, best_score)
            model = SVC(C=best_parameters['C'], class_weight=None, gamma=best_parameters['gamma'],
                        kernel=best_parameters['Kernel'])
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            signal_svc.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            pred.append(model.predict(x_test)[0])
            true_values.append(y_test['Signals'][0])
            param.append(best_parameters)

        else:
            # Se não cair na semana de fazer o Tuning hyperparamenter, fite os dados normalmente e gere previsões
            # Treinando o mesmo modelo usado anteriormente (melhor modelo escolhido algumas semanas atrás)
            # na amostra treino + validação
            # Gera forecasting um passo à frente fora da amostra

            model = SVC(C=best_parameters['C'], class_weight=None, gamma=best_parameters['gamma'],
                        kernel=best_parameters['Kernel'])
            model.fit(x_train_val, np.ravel(y_train_val, order='C'))
            pred.append(model.predict(x_test)[0])
            signal_svc.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            model.fit(x_train, np.ravel(y_train, order='C'))
            scores_final_svc.set_value(x_test.index, crypto_string, model.score(x_val, np.ravel(y_val, order='C')))

            true_values.append(y_test['Signals'][0])
            scores.append(score)

        # gera um data frame com os retornos advindos dos sinais gerados
    # Se o sinal era de compra (1) retornos são usuais
    # Se sinal era de venda (-1), retornos são invertidos
    accuracy = accuracy_score(true_values, pred)

    for d in signal_svc.index:
        if signal_svc[crypto_string].loc[d] == 1:
            returns_svc.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[d] / df[crypto_string]['Adj Close'].loc[:d].iloc[-2]) - 1 -
                                  df_livre_risco['IRX Semanal'].loc[d])
        elif signal_svc[crypto_string].loc[d] == -1:
            returns_svc.set_value(d, crypto_string, (1 - CT) * (
                        df[crypto_string]['Adj Close'].loc[:d].iloc[-2] / df[crypto_string]['Adj Close'].loc[d]) - 1 -
                                  df_livre_risco['IRX Semanal'].loc[d])

    strat_index = pd.DataFrame(index=X.index[train_val:], columns=['Returns', 'Level'])

    strat_index['Level'].iloc[0] = 100
    strat_index['Returns'] = returns_svc[crypto_string]

    for d, dm1 in zip(strat_index.index[1:], strat_index.index[:-1]):
        strat_index['Level'].loc[d] = strat_index['Level'].loc[dm1] * (1 + strat_index['Returns'].loc[d])

    ret_y = strat_index['Level'].pct_change(52)
    # ret_y.plot()
    vol_y = strat_index['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

    SR = ret_y / vol_y
    # SR.plot()

    SR_mean = (strat_index['Level'].pct_change(1).mean() * 52) / (
                strat_index['Level'].pct_change(1).std() * np.sqrt(52))

    summary_final_svc['Accuracy'].loc[crypto_string] = accuracy
    summary_final_svc['Sharp'].loc[crypto_string] = SR_mean
    summary_final_svc['Mean Return'].loc[crypto_string] = strat_index['Returns'].mean() * 52

    for d in strat_index.index:
        return_final_svc.set_value(d, crypto_string, strat_index['Level'].loc[d])
#        signal_final_svc.set_value(d,crypto_string, signal_svc[crypto_string].loc[d])


print(summary_final_svc)
return_final_svc.dropna(axis=0, how='all').plot()

# A seguir, são gerados portfolios por diferentes métodos : EW, Otimização por mínima variância, IVP e CLusterização
# Cada método puxa as classes geradas no arquivo "port_construction.py" dentro do projeto

# Note que como fazemos balanceamente semanal, a cada semana geramos novas otimizações e tudo irá depender do
# número de cryptomoedas disponíveis até o momento inicial da amostra de sinais gerados.

######################

# Roda Equally Weighted ####

signals_ew = signal_svc.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')

ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)

soma = abs(signal_svc.dropna(axis=0, how='all')).sum(axis=1)

for d in ew_weights.index:
    for tracker in ew_weights.columns:
        if signals_ew[tracker].loc[d] == 0:
            ew_weights.set_value(d, tracker, 0)
        else:
            ew_weights.set_value(d, tracker, 1 / soma.loc[d])

    #    weights_ew = ew_weights
returns_ew = returns_svc.dropna(axis=0, how='all').fillna(0)  # [1:]

strat_index_svc_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 'Level': np.nan})

strat_index_svc_ew['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_svc_ew.index[1:], strat_index_svc_ew.index[:-1]):
    strat_index_svc_ew['Level'].loc[d] = strat_index_svc_ew['Level'].loc[dm1] *\
                                         (1 + strat_index_svc_ew['Return'].loc[d])

strat_index_svc_ew['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_svc_ew = strat_index_svc_ew['Level'].pct_change(52)
# ret_y.plot()
vol_y_svc_ew = strat_index_svc_ew['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_svc_ew = ret_y_svc_ew / vol_y_svc_ew
# SR.plot()

SR_mean_svc_ew = (strat_index_svc_ew['Level'].pct_change(1).mean() * 52) /\
                 (strat_index_svc_ew['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_svc_ew)

######################################

# Roda IVP ###

signals_ivp = signal_svc.dropna(axis=0, how='all').fillna(0)  # .dropna(axis=1,how='any')
ivp_weights = pd.DataFrame(data=0, index=signals_ivp.index, columns=signals_ivp.columns)
soma = abs(signal_svc.dropna(axis=0, how='all')).sum(axis=1)

for d in ivp_weights.index:
    if soma.loc[d] == 1:
        ivp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = ivp_weights['BTC-USD'].sum()

for d in tqdm(ivp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ivp = IVP(ret)
    w = ivp.weights
    for tracker in tqdm(ret.columns):
        ivp_weights[tracker].loc[d] = w[tracker]
#            ivp_weights.set_value(d, tracker, w[tracker])

returns_ivp = returns_svc.dropna(axis=0, how='all').fillna(0)

strat_index_svc_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_svc_ivp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_svc_ivp.index[1:], strat_index_svc_ivp.index[:-1]):
    strat_index_svc_ivp['Level'].loc[d] = strat_index_svc_ivp['Level'].loc[dm1] *\
                                          (1 + strat_index_svc_ivp['Return'].loc[d])

strat_index_svc_ivp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_svc_ivp = strat_index_svc_ivp['Level'].pct_change(52)
vol_y_svc_ivp = strat_index_svc_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_svc_ivp = ret_y_svc_ivp / vol_y_svc_ivp

SR_mean_svc_ivp = (strat_index_svc_ivp['Level'].pct_change(1).mean() * 52) /\
                  (strat_index_svc_ivp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_svc_ivp)

######################################

# RODA MVO ###

signals_mvp = signal_svc.dropna(axis=0, how='all').fillna(0)
return_mvp = returns_svc.dropna(axis=0, how='all').fillna(0)
mvp_weights = pd.DataFrame(data=0, index=signals_mvp.index, columns=signals_mvp.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        mvp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = mvp_weights['BTC-USD'].sum()

for d in tqdm(mvp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    mvp = MinVar(ret)
    w = mvp.weights
    for tracker in tqdm(mvp_weights.columns):
        mvp_weights[tracker].loc[d] = w[tracker]

strat_index_svc_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_svc_mvp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_svc_mvp.index[1:], strat_index_svc_mvp.index[:-1]):
    strat_index_svc_mvp['Level'].loc[d] = strat_index_svc_mvp['Level'].loc[dm1] *\
                                          (1 + strat_index_svc_mvp['Return'].loc[d])

strat_index_svc_mvp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_svc_mvp = strat_index_svc_mvp['Level'].pct_change(52)
vol_y_svc_mvp = strat_index_svc_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_svc_mvp = ret_y_svc_mvp / vol_y_svc_mvp

SR_mean_svc_mvp = (strat_index_svc_mvp['Level'].pct_change(1).mean() * 52) /\
                  (strat_index_svc_mvp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_svc_mvp)

#########################################

######################
# PESOS por HYERARCHICAL CLUSTERING #######

signals_cluster = signal_svc.dropna(axis=0, how='all').fillna(0)
return_cluster = returns_svc.dropna(axis=0, how='all').fillna(0)
hrp_weights = pd.DataFrame(data=0, index=signals_cluster.index, columns=signals_cluster.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        hrp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = hrp_weights['BTC-USD'].sum()

for d in tqdm(hrp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    hrp = HRP(ret)
    w = hrp.weights
    for tracker in tqdm(hrp_weights.columns):
        hrp_weights[tracker].loc[d] = w[tracker]

strat_index_svc_cluster =\
    pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1), 'Level': np.nan})

strat_index_svc_cluster['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_svc_cluster.index[1:], strat_index_svc_cluster.index[:-1]):
    strat_index_svc_cluster['Level'].loc[d] = strat_index_svc_cluster['Level'].loc[dm1] *\
                                              (1 + strat_index_svc_cluster['Return'].loc[d])

strat_index_svc_cluster['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_svc_cluster = strat_index_svc_cluster['Level'].pct_change(52)
vol_y_svc_cluster = strat_index_svc_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_svc_cluster = ret_y_svc_cluster / vol_y_svc_cluster

SR_mean_svc_cluster = (strat_index_svc_cluster['Level'].pct_change(1).mean() * 52) /\
                      (strat_index_svc_cluster['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_svc_cluster)


########################################################################################################################

# Model Combination

########################################################################################################################

scores_final_knn = scores_final_knn.dropna(axis=0, how='all').fillna(0)
scores_final_logit = scores_final_logit.dropna(axis=0, how='all').fillna(0)
scores_final_svc = scores_final_svc.dropna(axis=0, how='all').fillna(0)

signal_knn = signal_knn.dropna(axis=0, how='all').fillna(0)
signal_logit = signal_logit.dropna(axis=0, how='all').fillna(0)
signal_svc = signal_svc.dropna(axis=0, how='all').fillna(0)


# Função que gera sinais, dados os valores de entrada
def make_signal(x1):
    if x1 > 0:
        return 1
    else:
        return -1


signal_combination = pd.DataFrame(index=signal_svc.index, columns=signal_svc.columns)
returns_combination = pd.DataFrame(index=signal_svc.index, columns=signal_svc.columns)

# gera sinal através de model combination, ou seja, pega a média dos sinais gerados pelos modelos Logistic, KNN e SVC.
for d in signal_combination.index:
    for crypto in signal_combination.columns:
        if signal_svc[crypto].loc[d] == 0:
            signal_combination[crypto].loc[d] = 0
        else:
            comb = signal_knn[crypto].loc[d] * (scores_final_knn[crypto].loc[d] / scores_final_knn[crypto].loc[d] +
                                                scores_final_logit[crypto].loc[d] + scores_final_svc[crypto].loc[d]) + \
                   signal_logit[crypto].loc[d] * (signal_logit[crypto].loc[d] / scores_final_knn[crypto].loc[d] +
                                                  scores_final_logit[crypto].loc[d] +
                                                  scores_final_svc[crypto].loc[d]) + \
                   signal_svc[crypto].loc[d] * (signal_svc[crypto].loc[d] / scores_final_knn[crypto].loc[d] +
                                                scores_final_logit[crypto].loc[d] + scores_final_svc[crypto].loc[d])

            signal_combination[crypto].loc[d] = make_signal(comb)

# Dados os sinais, constrói série de retornos gerados pelos sinais construídos

for d in signal_combination.index:
    for crypto in signal_combination.columns:
        if signal_combination[crypto].loc[d] == 1:
            returns_combination.set_value(d, crypto, (1 - CT) * (
                        df[crypto]['Adj Close'].loc[d] / df[crypto]['Adj Close'].loc[:d].iloc[-2]) - 1 -
                                          df_livre_risco['IRX Semanal'].loc[d])
        elif signal_combination[crypto].loc[d] == -1:
            returns_combination.set_value(d, crypto, (1 - CT) * (
                        df[crypto]['Adj Close'].loc[:d].iloc[-2] / df[crypto]['Adj Close'].loc[d]) - 1 -
                                          df_livre_risco['IRX Semanal'].loc[d])

######################

# Roda Equally Weighted Portfolio do modelo Combination ####

signals_ew = signal_combination

ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)

soma = abs(signal_combination.dropna(axis=0, how='all')).sum(axis=1)

for d in ew_weights.index:
    for tracker in ew_weights.columns:
        if signals_ew[tracker].loc[d] == 0:
            ew_weights.set_value(d, tracker, 0)
        else:
            ew_weights.set_value(d, tracker, 1 / soma.loc[d])

#    weights_ew = ew_weights
returns_ew = returns_combination.dropna(axis=0, how='all').fillna(0)  # [1:]

strat_index_comb_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 'Level': np.nan})

strat_index_comb_ew['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_comb_ew.index[1:], strat_index_comb_ew.index[:-1]):
    strat_index_comb_ew['Level'].loc[d] = strat_index_comb_ew['Level'].loc[dm1] *\
                                          (1 + strat_index_comb_ew['Return'].loc[d])

strat_index_comb_ew['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_comb_ew = strat_index_comb_ew['Level'].pct_change(52)
# ret_y.plot()
vol_y_comb_ew = strat_index_comb_ew['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_comb_ew = ret_y_comb_ew / vol_y_comb_ew
# SR.plot()

SR_mean_comb_ew = (strat_index_comb_ew['Level'].pct_change(1).mean() * 52) /\
                  (strat_index_comb_ew['Level'].pct_change(1).std() * np.sqrt(52))

# Sharp Ratio médio
print(SR_mean_comb_ew)

######################################

# Roda IVP Portfólio para o modelo Combination ###

signals_ivp = signal_combination
ivp_weights = pd.DataFrame(data=0, index=signals_ivp.index, columns=signals_ivp.columns)
soma = abs(signal_combination.dropna(axis=0, how='all')).sum(axis=1)

for d in ivp_weights.index:
    if soma.loc[d] == 1:
        ivp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = ivp_weights['BTC-USD'].sum()

for d in tqdm(ivp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ivp = IVP(ret)
    w = ivp.weights
    for tracker in tqdm(ret.columns):
        ivp_weights[tracker].loc[d] = w[tracker]
#            ivp_weights.set_value(d, tracker, w[tracker])

returns_ivp = returns_combination.dropna(axis=0, how='all').fillna(0)

strat_index_comb_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_comb_ivp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_comb_ivp.index[1:], strat_index_comb_ivp.index[:-1]):
    strat_index_comb_ivp['Level'].loc[d] = strat_index_comb_ivp['Level'].loc[dm1] *\
                                           (1 + strat_index_comb_ivp['Return'].loc[d])

strat_index_comb_ivp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_comb_ivp = strat_index_comb_ivp['Level'].pct_change(52)
vol_y_comb_ivp = strat_index_comb_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_comb_ivp = ret_y_comb_ivp / vol_y_comb_ivp

SR_mean_comb_ivp = (strat_index_comb_ivp['Level'].pct_change(1).mean() * 52) /\
                   (strat_index_comb_ivp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_comb_ivp)

######################################

# RODA MVO Portfólio para o modelo Combination ###

signals_mvp = signal_combination
return_mvp = returns_combination.dropna(axis=0, how='all').fillna(0)
mvp_weights = pd.DataFrame(data=0, index=signals_mvp.index, columns=signals_mvp.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        mvp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = mvp_weights['BTC-USD'].sum()

for d in tqdm(
        mvp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    mvp = MinVar(ret)
    w = mvp.weights
    for tracker in tqdm(mvp_weights.columns):
        mvp_weights[tracker].loc[d] = w[tracker]

strat_index_comb_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 'Level': np.nan})

strat_index_comb_mvp['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_comb_mvp.index[1:], strat_index_comb_mvp.index[:-1]):
    strat_index_comb_mvp['Level'].loc[d] = \
        strat_index_comb_mvp['Level'].loc[dm1] * (1 + strat_index_comb_mvp['Return'].loc[d])

strat_index_comb_mvp['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_comb_mvp = strat_index_comb_mvp['Level'].pct_change(52)
vol_y_comb_mvp = strat_index_comb_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_comb_mvp = ret_y_comb_mvp / vol_y_comb_mvp

SR_mean_comb_mvp = (strat_index_comb_mvp['Level'].pct_change(1).mean() * 52) /\
                   (strat_index_comb_mvp['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_comb_mvp)

#########################################

######################
# PESOS por HYERARCHICAL CLUSTERING Portfólio para o modelo Combination#######

signals_cluster = signal_combination
return_cluster = returns_combination.dropna(axis=0, how='all').fillna(0)
hrp_weights = pd.DataFrame(data=0, index=signals_cluster.index, columns=signals_cluster.columns)

for d in mvp_weights.index:
    if soma.loc[d] == 1:
        hrp_weights['BTC-USD'].loc[d] = 1

    else:
        break

i = hrp_weights['BTC-USD'].sum()

for d in tqdm(hrp_weights.index[i:]):
    ret = df_returns.loc[:d + datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
    hrp = HRP(ret)
    w = hrp.weights
    for tracker in tqdm(hrp_weights.columns):
        hrp_weights[tracker].loc[d] = w[tracker]

strat_index_comb_cluster = \
    pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1), 'Level': np.nan})

strat_index_comb_cluster['Level'].iloc[0] = 100

for d, dm1 in zip(strat_index_comb_cluster.index[1:], strat_index_comb_cluster.index[:-1]):
    strat_index_comb_cluster['Level'].loc[d] = strat_index_comb_cluster['Level'].loc[dm1] *\
                                               (1 + strat_index_comb_cluster['Return'].loc[d])

strat_index_comb_cluster['Level'].plot(figsize=(8, 5))

#     SHARP RATIO
ret_y_comb_cluster = strat_index_comb_cluster['Level'].pct_change(52)
vol_y_comb_cluster = strat_index_comb_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std) * np.sqrt(52)

SR_comb_cluster = ret_y_comb_cluster / vol_y_comb_cluster

SR_mean_comb_cluster = (strat_index_comb_cluster['Level'].pct_change(1).mean() * 52) /\
                       (strat_index_comb_cluster['Level'].pct_change(1).std() * np.sqrt(52))

print(SR_mean_comb_cluster)

#########################################


# BOOTSTRAP ###


frames = [strat_index_comb_ew['Level'], strat_index_comb_ivp['Level'], strat_index_comb_mvp['Level'],
          strat_index_comb_cluster['Level'], strat_index_logit_ew['Level'], strat_index_logit_ivp['Level'],
          strat_index_logit_mvp['Level'], strat_index_logit_cluster['Level'], strat_index_svc_ew['Level'],
          strat_index_svc_ivp['Level'], strat_index_svc_mvp['Level'], strat_index_svc_cluster['Level'],
          strat_index_knn_ew['Level'], strat_index_knn_ivp['Level'], strat_index_knn_mvp['Level'],
          strat_index_knn_cluster['Level']]

backtests = pd.concat(frames, axis=1)

backtests.columns = ['Comb_EW', 'Comb_IVP', 'Comb_MVP', 'Comb_Cluster', 'Logit_EW', 'Logit_IVP', 'Logit_MVP',
                     'Logit_Cluster', 'SVC_EW', 'SVC_IVP', 'SVC_MVP', 'SVC_Cluster', 'KNN_EW', 'KNN_IVP', 'KNN_MVP',
                     'KNN_Cluster', ]

# 1) Para cada janela de tempo (com sobreposição), amostra com reposição 52 observações aleatórias.
# 2) Feito isso, calcula o Sharp Ratio dessa amostra aleatória
# 3) Repete o passo 2 (dois) 500 vezes e guarda esses 500 Sharp Ratios
# 4) Tira a média dos 500 Sharp Ratios gerados dentro da primeira janela de tempo
# 5) Repete os passos anteriores para todas as janelas de tempo
# 6) Gera uma distribuição de Sharp Ratios ao longo do tempo
# 7) Gera histograma de SR
SR = {}
for model in backtests.columns:
    simulated_SR = []
    for d1, d2 in zip(backtests.index[53:], backtests.index[:-52]):
        fold = backtests[model].loc[d2:d1].pct_change(1).dropna()
        SRs = []
        for _ in range(500):
            sample = np.random.choice(fold, size=52)
            SRs.append((sample.mean() * 52 / (sample.std() * np.sqrt(52))))

        simulated_SR.append(np.mean(SRs))

        SR[model] = simulated_SR

for i, model in enumerate(backtests.columns):
    fig, ax = plt.subplots()
    mu = np.mean(SR[model])
    median = np.median(SR[model])
    sigma = np.std(SR[model])
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\mathrm{median}=%.2f$' % (median,),
        r'$\sigma=%.2f$' % (sigma,)))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.hist(SR[model])
    plt.title(model)
    plt.show()


###################################################################################
# Function to give some analysis

def getperformancetable(indexseries, freq='Daily'):
    adju_factor = 252.0
    if freq == 'Monthly':
        adju_factor = 12.0
    elif freq == 'Weekly':
        adju_factor = 52.0

    table = pd.Series(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                             'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                             '10th percentile in Vol Terms'])

    cleanindexseries = indexseries.dropna().sort_index()

    er_index = pd.Series(index=cleanindexseries.index)
    er_index[cleanindexseries.index[0]] = 100.0
    for d3, d_minus_1 in zip(er_index.index[1:], er_index.index[:-1]):
        er = cleanindexseries[d3] / cleanindexseries[d_minus_1] - 1.0
        er_index[d3] = er_index[d_minus_1] * (1 + er)

    table['Excess Return'] = (cleanindexseries[-1] / cleanindexseries[0]) ** (
                adju_factor / (len(cleanindexseries) - 1.0)) - 1
    table['Volatility'] = (np.log(er_index).diff(1).dropna()).std() * np.sqrt(adju_factor)
    table['Sharpe'] = table['Excess Return'] / table['Volatility']
    table['Sortino'] = table['Excess Return'] / (np.sqrt(adju_factor) * (
        np.log(er_index).diff(1).dropna()[np.log(er_index).diff(1).dropna() < 0.0]).std())
    table['Max Drawdown'] = max_dd(er_index)
    table['Max Drawdown in Vol Terms'] = max_dd(er_index) / table['Volatility']
    table['5th percentile in Vol Terms'] = (er_index.pct_change(1).dropna()).quantile(q=0.05) / table['Volatility']
    table['10th percentile in Vol Terms'] = (er_index.pct_change(1).dropna()).quantile(q=0.1) / table['Volatility']
    return table


def max_dd(ser):
    max2here = ser.expanding(min_periods=1).max()
    dd2here = ser / max2here - 1.0
    return dd2here.min()


###########################################################################################

descriptions = pd.DataFrame(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                                   'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                                   '10th percentile in Vol Terms'])

# Gera gráfico de backtest dos 16 modelos diferentes
for i, model in enumerate(backtests.columns):
    plt.figure(i + 16)
    backtests[model].plot()
    df_mercado['Index'].plot()
    plt.title(model)
    plt.show()

    descriptions[model] = getperformancetable(backtests[model], 'Weekly')

descriptions.to_excel('Stats.xlsx')

# Calculando o beta das estratégias com S&P

betas = pd.DataFrame(index=backtests.columns, columns=["Beta", 'PValue'])

for model in backtests.columns:
    x = df_mercado['Index'].pct_change(1)
    x = sm.add_constant(x)
    x = x.dropna()

    y = backtests[model].pct_change()
    y = y.dropna()
    y = y.loc[x.index[0]:]

    regression = sm.OLS(y, x)

    results = regression.fit()

    betas['Beta'].loc[model] = results.params['Index']
    betas['PValue'].loc[model] = results.pvalues['Index']

betas.to_excel('betas.xlsx')
