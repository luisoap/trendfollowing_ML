import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.svm import SVC
import datetime as dt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm 
#from ML.port_construction import HRP, IVP, MinVar
import data_crypto as dc
from data_crypto.crypto import gen_data
from port_construction import HRP, IVP, MinVar

# Habilita rodar os diferentes métodos de construção de portfólios
roda_HRP = 1
roda_MVO = 1
roda_IVP = 1
roda_EW = 1
###############

# Lista de crypto currencies

crypto_list = ['BTC-USD','LTC-USD','XRP-USD','ETH-USD','XMR-USD','DASH-USD','DCR-USD'] #'BCH-USD',
start = '2010-01-01'
# Usando a função da biblioteca "data_crypto".
df = dc.read_crypto(crypto_list, start)
# Como vamos gerar estratégias semanais, passamos os dados para frequência semanal
for crypto_string in crypto_list:
    df[crypto_string] = df[crypto_string].resample('W').last()

 
# Cria os data Frames que serão usados posteriormente para armazenar os dados finais

returns_svc = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)    
summary_final_svc = pd.DataFrame(index=crypto_list, columns=['Accuracy', 'Sharp', 'Mean Return'])  
scores_final_svc = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)  
signal_svc = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)  
signal_final_svc = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)
return_final_svc = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)
df_returns = pd.DataFrame(index= df['BTC-USD'].index, columns=crypto_list)

# Cria data frame de retornos semanais realizados
for crypto in crypto_list:
    df_returns[crypto] = df[crypto]['Adj Close'].pct_change(1)
    
# cria função MACD
def macd(df):
       
    EMA_slow = df['Adj Close'].ewm(span=26).mean()
    EMA_fast = df['Adj Close'].ewm(span=12).mean()
    macd = EMA_fast - EMA_slow
    
    return macd

# Para cada cryptomoeda, vai gerar a matriz "X" de dados usados para gerar as previsões fora da amostra
# para os sinais de compra ou venda
for crypto_string in crypto_list:
   
    data = df[crypto_string]
    data.index = pd.to_datetime(data.index)

    data['High'] = data['High']
    data['Low'] = data['Low']
    data['Adj Close'] = data['Adj Close']

    data['MACD']= macd(data)
# a biblioteca Talib pode dar erro se você não a instalou corretamente
# Para instalar em Windows, siga o tutorial: https://medium.com/@keng16302/how-to-install-ta-lib-in-python-on-window-9303eb003fbb

# Caso não consiga de jeito nenhum instalar a biblioteca Talib, uma última opção é "deletar" as duas linhas abaixo, o que deve reduzir bastante a acurácia do modelo
#     data['ADX'] = ta.ADX(np.array(data['High']), np.array(data['Low']),np.array(data['Adj Close']))
#     data['RSI'] = ta.RSI(np.array(data['Adj Close']))
    
    data['Return1']= data['Adj Close'].pct_change(1)
    data['Return2']= data['Adj Close'].pct_change(2)
    data['Return3']= data['Adj Close'].pct_change(3)    
    
    data = data.drop(['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'],axis=1)
    data = data.loc[~data.index.duplicated(keep='first')]
        
    signals = pd.DataFrame(index=data.index[1:], columns=['Signals'])

# Gera uma coluna com os sinais realizados para serem usados como variável Y do modelo
    
    for d in signals.index:

        if data['Return1'].loc[d]>= 0:
            signals['Signals'].loc[d]= 1
        elif data['Return1'].loc[d] < 0:
            signals['Signals'].loc[d]= -1

    X = data.shift(1).dropna()
    y = signals[X.index[0]:]    
# Vamos permitir iterações polinomiais de segunda ordem entre as variáveis explicativas dos sinais, para tornar o modelo menos linear        
    polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = polynomial_interaction.fit_transform(X)    
    X = pd.DataFrame(data=X_poly, index=X.index)
# Gera Componentes Principais
# Note que quando n_components =0.97, equivale a cuspir o número de componentes principais de tal forma que 
# 97% da variância seja explicada. Portanto, o número de componentes será diferente para cada cryptomoeda, para cada modelo e para cada janela temporal...é variante no tempo!
 
    pca = PCA(n_components=0.97, whiten=True)

# Quebro a amostra em 50% para teste (que será o backtest final), 40% para treino e 10% para validação
    split_test = 0.5
    split_train_val = 0.5
    split_train=0.4
           
    test=int(split_test*len(X))    
    train=int(split_train*len(X))    
    train_val = int(split_train_val*len(X))
    val = train_val - train

# Defino intervalos temporais nos quais irei "Tunar" os hyperparâmetros do modelo    
# Idealmente, o correto é fazer isto para todos os períodos de tempo para tornar o modelo mais flexível e 
# aumentar a taxa de aprendizado do modelo, mas como isto gera custo computacional, coloco como padrão 
# uma atualização ótima dos hyperparâmetros a cada 4 semanas ( um mês)
     
    window_lengths = list(range(train_val+1,len(X), 4))  
           
    scores = []       
    param = []            
    pred = []
    true_values = []


# Aqui faço um "expanding sample": A cada novo período, expando um ponto a amostra de treino, desloco um ponto para frente a amostra de validação
# e um ponto a observação de teste (mantendo o tamnho da amostra de validação e teste constantes, apenas a amostra de treino que cresce com o tempo)
# Este método visa replicar a estratégia real que teria sido feita caso replicássemos no passado o procedimento, semana após semana, até os dias atuais
# Este tipo de abordagem é usual na literatura de forecasting e de validação de modelos out-of-sample
# O procedimento segue o paper "Empirical Asset Pricing via Machine Learning (2019) "
# Equivale a fazer previsão fora da amostra 1 passo à frente, com validação mensal dos hyperparâmetros e treino semanal dos parâmetros
   
    for i in range(train_val+1,len(X)): 
        Xs = X.loc[:X.index[i]]
# Padroniza as variáveis para terem média zero e variância 1, de tal forma que o PCA não seja guiado por variáveis cuja variância se sobrepõe às outras variáveis
        Xs = StandardScaler().fit_transform(Xs) 
# Aplica PCA para amostra disponível no período, ou seja, treino + validação

        Xs = pd.DataFrame(data=pca.fit_transform(Xs), index=X.index[:i+1])
# Define as amostras de treino, validação e teste
        x_train_val = Xs.loc[:Xs.index[-2]]

        x_val = Xs.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        x_train = Xs.loc[:x_val.index[0]+datetime.timedelta(-1)]
        x_test = Xs.loc[Xs.index[i]:Xs.index[i]]        
        
        y_train_val = y.loc[:Xs.index[i-1]]
        y_val = y.loc[x_train_val.index[-val]:x_train_val.index[-1]]
        y_train = y.loc[:x_val.index[0]+datetime.timedelta(-1)]
        y_test = y.loc[Xs.index[i]:Xs.index[i]]      
        

        if i in window_lengths:
        # parameter tuning
            best_score = 0
# A cada 4 semanas, faz o procedimento de "Tuning Hyperparameters "
# Seleciona na amostra de validação o modelo que gera melhor acurácia fora da amostra para diferentes hyperparametros
            
            for models in ['rbf','linear']:
                for gamma  in [0.001, 0.01, 0.1, 1, 10, 100]:
                    for c in [0.001, 0.01, 0.1, 1, 10, 100]:
            # for each combination of parameters, train an SVC
                        svm = SVC(gamma=gamma, C=c)
                        svm.fit(x_train, y_train)
# evaluate the SVC on the test set
                        score = svm.score(x_val, y_val)
# if we got a better score, store the score and parameters
                        if score > best_score:
                            best_score = score
                            best_parameters = {'C': c, 'gamma': gamma, 'Kernel' : models}

# Escolhido o melhor modelo, faça fit dos dados in-sample com a amostra de treino + validação
# Gere out-of-sample forecasting um passo à frente para gerar o sinal prevista para o portfólio da semana seguinte
             
            scores_final_svc.set_value(x_test.index, crypto_string, best_score)
            model = SVC(C=best_parameters['C'], class_weight=None, gamma=best_parameters['gamma'], kernel=best_parameters['Kernel'])
            model.fit(x_train_val, y_train_val)
            signal_svc.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            pred.append(model.predict(x_test)[0])
            true_values.append(y_test['Signals'][0])
            param.append(best_parameters)    
           
            
        else:
# Se não cair na semana de fazer o Tuning hyperparamenter, fite os dados normalmente e gere previsões
# Treinando o mesmo modelo usado anteriormente (melhor modelo escolhido algumas semanas atrás)  na amostra treino + validação
# Gera forecasting um passo à frente fora da amostra            

            model = SVC(C=best_parameters['C'], class_weight=None, gamma=best_parameters['gamma'], kernel=best_parameters['Kernel'])
            model.fit(x_train_val, y_train_val)
            pred.append(model.predict(x_test)[0])
            signal_svc.set_value(x_test.index, crypto_string, model.predict(x_test)[0])
            model.fit(x_train,y_train)
            scores_final_svc.set_value(x_test.index, crypto_string, model.score(x_val, y_val))
           
            true_values.append(y_test['Signals'][0]) 
            scores.append(score) 
            
# gera um data frame com os retornos advindos dos sinais gerados
# Se o sinal era de compra (1) retornos são usuais
# Se sinal era de venda (-1), retornos são invertidos            
    accuracy = accuracy_score(true_values, pred)
    

    for d in signal_svc.index:
        if signal_svc[crypto_string].loc[d]==1:
            returns_svc.set_value(d, crypto_string, (1-CT)*(df[crypto_string]['Adj Close'].loc[d]/df[crypto_string]['Adj Close'].loc[:d].iloc[-2]) - 1 - df_livre_risco['IRX Semanal'].loc[d])
        elif signal_svc[crypto_string].loc[d]==-1:
            returns_svc.set_value(d, crypto_string, (1-CT)*(df[crypto_string]['Adj Close'].loc[:d].iloc[-2]/df[crypto_string]['Adj Close'].loc[d]) -1 - df_livre_risco['IRX Semanal'].loc[d])
    
    
    
    strat_index = pd.DataFrame(index=X.index[train_val:], columns=['Returns', 'Level'])

    strat_index['Level'].iloc[0] = 100
    strat_index['Returns'] = returns_svc[crypto_string]

    for d, dm1 in zip(strat_index.index[1:], strat_index.index[:-1]):
        strat_index['Level'].loc[d] = strat_index['Level'].loc[dm1] * (1 + strat_index['Returns'].loc[d])

            
    ret_y = strat_index['Level'].pct_change(52)
    #ret_y.plot()
    vol_y = strat_index['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR = ret_y/vol_y
    #SR.plot()

    SR_mean = (strat_index['Level'].pct_change(1).mean()*52)/(strat_index['Level'].pct_change(1).std()*np.sqrt(52))

    summary_final_svc['Accuracy'].loc[crypto_string] = accuracy
    summary_final_svc['Sharp'].loc[crypto_string] = SR_mean    
    summary_final_svc['Mean Return'].loc[crypto_string] = strat_index['Returns'].mean()*52

    
    
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

### Roda Equally Weighted ####
    
if roda_EW == 1:

    signals_ew = signal_svc.dropna(axis=0, how='all').fillna(0)#.dropna(axis=1,how='any')

    ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)
         
    
    soma = abs(signal_svc.dropna(axis=0, how='all')).sum(axis=1)    
    
    for d in ew_weights.index:
        for tracker in ew_weights.columns:
            if signals_ew[tracker].loc[d]==0:
                ew_weights.set_value(d,tracker, 0)
            else:
                ew_weights.set_value(d,tracker, 1/soma.loc[d])
                
                
            
#    weights_ew = ew_weights       
    returns_ew  = returns_svc.dropna(axis=0, how='all').fillna(0)#[1:] 
   
    
    strat_index_svc_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_svc_ew['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_svc_ew.index[1:], strat_index_svc_ew.index[:-1]):
        strat_index_svc_ew['Level'].loc[d] = strat_index_svc_ew['Level'].loc[dm1] * (1+strat_index_svc_ew['Return'].loc[d])

    strat_index_svc_ew['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_svc_ew = strat_index_svc_ew['Level'].pct_change(52)
#ret_y.plot()
    vol_y_svc_ew = strat_index_svc_ew['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_svc_ew = ret_y_svc_ew/vol_y_svc_ew
#SR.plot()

    SR_mean_svc_ew = (strat_index_svc_ew['Level'].pct_change(1).mean()*52)/(strat_index_svc_ew['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_svc_ew)   




######################################
    
    ## Roda IVP ###
    
if roda_IVP ==1:
    
    signals_ivp = signal_svc.dropna(axis=0, how='all').fillna(0) #.dropna(axis=1,how='any')
    ivp_weights = pd.DataFrame(data = 0, index=signals_ivp.index, columns=signals_ivp.columns)
    soma = abs(signal_svc.dropna(axis=0, how='all')).sum(axis=1)
    
    for d in ivp_weights.index:        
        if soma.loc[d]==1:
            ivp_weights['BTC-USD'].loc[d] = 1
            
        else:
            break
                
                            
    i = ivp_weights['BTC-USD'].sum()
    
    for d in tqdm(ivp_weights.index[i:]):
        ret= df_returns.loc[:d+datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
        ivp= IVP(ret)
        w = ivp.weights
        for tracker in tqdm(ret.columns):  
            ivp_weights[tracker].loc[d] = w[tracker]
#            ivp_weights.set_value(d, tracker, w[tracker]) 
    
    
    returns_ivp  = returns_svc.dropna(axis=0, how='all').fillna(0)
    
    strat_index_svc_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_svc_ivp['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_svc_ivp.index[1:], strat_index_svc_ivp.index[:-1]):
        strat_index_svc_ivp['Level'].loc[d] = strat_index_svc_ivp['Level'].loc[dm1] * (1+strat_index_svc_ivp['Return'].loc[d])

    strat_index_svc_ivp['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_svc_ivp = strat_index_svc_ivp['Level'].pct_change(52)
    vol_y_svc_ivp = strat_index_svc_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_svc_ivp = ret_y_svc_ivp/vol_y_svc_ivp

    SR_mean_svc_ivp = (strat_index_svc_ivp['Level'].pct_change(1).mean()*52)/(strat_index_svc_ivp['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_svc_ivp)     
    

######################################
    
#### RODA MVO ###
        
if roda_MVO==1:
    
    signals_mvp = signal_svc.dropna(axis=0, how='all').fillna(0)
    return_mvp = returns_svc.dropna(axis=0, how='all').fillna(0)
    mvp_weights = pd.DataFrame(data = 0, index=signals_mvp.index, columns=signals_mvp.columns)
    
    for d in mvp_weights.index:        
        if soma.loc[d]==1:
            mvp_weights['BTC-USD'].loc[d] = 1
            
        else:
            break
                
                            
    i = mvp_weights['BTC-USD'].sum()
    
    for d in tqdm(mvp_weights.index[i:]):
        ret= df_returns.loc[:d+datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
        mvp= MinVar(ret)
        w = mvp.weights
        for tracker in tqdm(mvp_weights.columns):         
            mvp_weights[tracker].loc[d] = w[tracker]
    

    strat_index_svc_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_svc_mvp['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_svc_mvp.index[1:], strat_index_svc_mvp.index[:-1]):
        strat_index_svc_mvp['Level'].loc[d] = strat_index_svc_mvp['Level'].loc[dm1] * (1+strat_index_svc_mvp['Return'].loc[d])

    strat_index_svc_mvp['Level'].plot(figsize=(8, 5))   
   
 
#     SHARP RATIO 
    ret_y_svc_mvp = strat_index_svc_mvp['Level'].pct_change(52)
    vol_y_svc_mvp = strat_index_svc_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_svc_mvp = ret_y_svc_mvp/vol_y_svc_mvp

    SR_mean_svc_mvp = (strat_index_svc_mvp['Level'].pct_change(1).mean()*52)/(strat_index_svc_mvp['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_svc_mvp)     


#########################################
    
######################
#### PESOS por HYERARCHICAL CLUSTERING #######
    
if roda_HRP==1:
    signals_cluster = signal_svc.dropna(axis=0, how='all').fillna(0)
    return_cluster = returns_svc.dropna(axis=0, how='all').fillna(0)
    hrp_weights = pd.DataFrame(data = 0, index=signals_cluster.index, columns=signals_cluster.columns)
 
    for d in mvp_weights.index:        
        if soma.loc[d]==1:
            hrp_weights['BTC-USD'].loc[d] = 1
            
        else:
            break
                
                            
    i = hrp_weights['BTC-USD'].sum()
    
    for d in tqdm(hrp_weights.index[i:]):
        ret= df_returns.loc[:d+datetime.timedelta(-1)].dropna(axis=0, how='all').dropna(axis=1, how='all')
        hrp= HRP(ret)
        w = hrp.weights
        for tracker in tqdm(hrp_weights.columns):         
            hrp_weights[tracker].loc[d] = w[tracker]
            
    
    strat_index_svc_cluster = pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_svc_cluster['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_svc_cluster.index[1:], strat_index_svc_cluster.index[:-1]):
        strat_index_svc_cluster['Level'].loc[d] = strat_index_svc_cluster['Level'].loc[dm1] * (1+strat_index_svc_cluster['Return'].loc[d])

    strat_index_svc_cluster['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_svc_cluster = strat_index_svc_cluster['Level'].pct_change(52)
    vol_y_svc_cluster = strat_index_svc_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_svc_cluster = ret_y_svc_cluster/vol_y_svc_cluster

    SR_mean_svc_cluster = (strat_index_svc_cluster['Level'].pct_change(1).mean()*52)/(strat_index_svc_cluster['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_svc_cluster)    
    

    