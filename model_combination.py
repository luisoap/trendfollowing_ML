### MODEL COMBINATION 

import pandas_datareader.data as pdr
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Taxa livre de risco semanal, retorno semanal

end = dt.datetime.now().strftime("%Y-%m-%d")

df_tbill = pdr.DataReader('^IRX', data_source='yahoo', start='2010-01-01', end=end)

df_livre_risco = pd.DataFrame(index=df_tbill.index, columns=["IRX Anual", "IRX Semanal"])

for i in df_tbill.index:
    df_livre_risco["IRX Anual"].loc[i] = df_tbill["Close"].loc[i]/100

df_livre_risco = df_livre_risco.resample("W").last()

for i in df_livre_risco.index:
    df_livre_risco["IRX Semanal"].loc[i] = (1 + df_livre_risco["IRX Anual"].loc[i])**(1/52)-1

# Custo de transação 0.25 bps, sendo que 1 bps é 0.1%

CT = 0.0001*0.25

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


# Executa os arquivos .py com os comandos necessários para rodar os diferentes modelos
exec(open("knn.py").read())
exec(open("logistic.py").read())
exec(open("SVC.py").read())


####
# Habilita rodar os diferentes tipos de construção de portfólios
roda_HRP = 1
roda_MVO = 1
roda_IVP = 1
roda_EW = 1


scores_final_knn = scores_final_knn.dropna(axis=0, how='all').fillna(0)
scores_final_logit = scores_final_logit.dropna(axis=0, how='all').fillna(0)
scores_final_svc = scores_final_svc.dropna(axis=0, how='all').fillna(0)


signal_knn = signal_knn.dropna(axis=0, how='all').fillna(0)
signal_logit = signal_logit.dropna(axis=0, how='all').fillna(0)
signal_svc = signal_svc.dropna(axis=0, how='all').fillna(0)

# Função que gera sinais, dados os valores de entrada
def make_signal(x):
    if x>0:
        return 1
    else: 
        return -1
    
signal_combination = pd.DataFrame(index = signal_svc.index, columns = signal_svc.columns)
returns_combination = pd.DataFrame(index = signal_svc.index, columns = signal_svc.columns)

# gera sinal através de model combination, ou seja, pega a média dos sinais gerados pelos modelos Logistic, KNN e SVC.
for d in signal_combination.index:
    for crypto in signal_combination.columns:
        if signal_svc[crypto].loc[d]==0:
            signal_combination[crypto].loc[d]=0
        else:
            comb = signal_knn[crypto].loc[d]*(scores_final_knn[crypto].loc[d] / scores_final_knn[crypto].loc[d] + scores_final_logit[crypto].loc[d] + scores_final_svc[crypto].loc[d]) 
            + signal_logit[crypto].loc[d]*(signal_logit[crypto].loc[d] / scores_final_knn[crypto].loc[d] + scores_final_logit[crypto].loc[d] + scores_final_svc[crypto].loc[d])
            + signal_svc[crypto].loc[d]*(signal_svc[crypto].loc[d] / scores_final_knn[crypto].loc[d] + scores_final_logit[crypto].loc[d] + scores_final_svc[crypto].loc[d])
                                    
            signal_combination[crypto].loc[d]= make_signal(comb)
 
# Dados os sinais, constrói série de retornos gerados pelos sinais construídos
    
for d in signal_combination.index:
    for crypto in signal_combination.columns:
        if signal_combination[crypto].loc[d]==1:
            returns_combination.set_value(d, crypto, (1-CT)*(df[crypto]['Adj Close'].loc[d]/df[crypto]['Adj Close'].loc[:d].iloc[-2]) - 1 - df_livre_risco['IRX Semanal'].loc[d])
        elif signal_combination[crypto].loc[d]==-1:
            returns_combination.set_value(d, crypto, (1-CT)*(df[crypto]['Adj Close'].loc[:d].iloc[-2]/df[crypto]['Adj Close'].loc[d]) -1 - df_livre_risco['IRX Semanal'].loc[d])
            
            

######################

### Roda Equally Weighted Portfolio do modelo Combination ####
    
if roda_EW == 1:

    signals_ew = signal_combination

    ew_weights = pd.DataFrame(index=signals_ew.index, columns=signals_ew.columns)
         
    
    soma = abs(signal_combination.dropna(axis=0, how='all')).sum(axis=1)    
    
    for d in ew_weights.index:
        for tracker in ew_weights.columns:
            if signals_ew[tracker].loc[d]==0:
                ew_weights.set_value(d,tracker, 0)
            else:
                ew_weights.set_value(d,tracker, 1/soma.loc[d])
                
                
            
#    weights_ew = ew_weights       
    returns_ew  = returns_combination.dropna(axis=0, how='all').fillna(0)#[1:] 
   
    
    strat_index_comb_ew = pd.DataFrame(data={'Return': (ew_weights * returns_ew).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_comb_ew['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_comb_ew.index[1:], strat_index_comb_ew.index[:-1]):
        strat_index_comb_ew['Level'].loc[d] = strat_index_comb_ew['Level'].loc[dm1] * (1+strat_index_comb_ew['Return'].loc[d])

    strat_index_comb_ew['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_comb_ew = strat_index_comb_ew['Level'].pct_change(52)
#ret_y.plot()
    vol_y_comb_ew = strat_index_comb_ew['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_comb_ew = ret_y_comb_ew/vol_y_comb_ew
#SR.plot()

    SR_mean_comb_ew = (strat_index_comb_ew['Level'].pct_change(1).mean()*52)/(strat_index_comb_ew['Level'].pct_change(1).std()*np.sqrt(52))

# Sharp Ratio médio
    print(SR_mean_comb_ew)   




######################################
    
    ## Roda IVP Portfólio para o modelo Combination ###
    
if roda_IVP ==1:
    
    signals_ivp = signal_combination
    ivp_weights = pd.DataFrame(data = 0, index=signals_ivp.index, columns=signals_ivp.columns)
    soma = abs(signal_combination.dropna(axis=0, how='all')).sum(axis=1)
    
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
    
    
    returns_ivp  = returns_combination.dropna(axis=0, how='all').fillna(0)
    
    strat_index_comb_ivp = pd.DataFrame(data={'Return': (ivp_weights * returns_ivp).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_comb_ivp['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_comb_ivp.index[1:], strat_index_comb_ivp.index[:-1]):
        strat_index_comb_ivp['Level'].loc[d] = strat_index_comb_ivp['Level'].loc[dm1] * (1+strat_index_comb_ivp['Return'].loc[d])

    strat_index_comb_ivp['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_comb_ivp = strat_index_comb_ivp['Level'].pct_change(52)
    vol_y_comb_ivp = strat_index_comb_ivp['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_comb_ivp = ret_y_comb_ivp/vol_y_comb_ivp

    SR_mean_comb_ivp = (strat_index_comb_ivp['Level'].pct_change(1).mean()*52)/(strat_index_comb_ivp['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_comb_ivp)     
    

######################################
    
#### RODA MVO Portfólio para o modelo Combination ###
        
if roda_MVO==1:
    
    signals_mvp = signal_combination
    return_mvp = returns_combination.dropna(axis=0, how='all').fillna(0)
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
    

    strat_index_comb_mvp = pd.DataFrame(data={'Return': (mvp_weights * return_mvp).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_comb_mvp['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_comb_mvp.index[1:], strat_index_comb_mvp.index[:-1]):
        strat_index_comb_mvp['Level'].loc[d] = strat_index_comb_mvp['Level'].loc[dm1] * (1+strat_index_comb_mvp['Return'].loc[d])

    strat_index_comb_mvp['Level'].plot(figsize=(8, 5))   
   
 
#     SHARP RATIO 
    ret_y_comb_mvp = strat_index_comb_mvp['Level'].pct_change(52)
    vol_y_comb_mvp = strat_index_comb_mvp['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_comb_mvp = ret_y_comb_mvp/vol_y_comb_mvp

    SR_mean_comb_mvp = (strat_index_comb_mvp['Level'].pct_change(1).mean()*52)/(strat_index_comb_mvp['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_comb_mvp)     


#########################################
    
######################
#### PESOS por HYERARCHICAL CLUSTERING Portfólio para o modelo Combination#######
    
if roda_HRP==1:
    signals_cluster = signal_combination
    return_cluster = returns_combination.dropna(axis=0, how='all').fillna(0)
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
            
    
    strat_index_comb_cluster = pd.DataFrame(data={'Return': (hrp_weights * return_cluster).dropna().sum(axis=1), 
                           'Level': np.nan})

    strat_index_comb_cluster['Level'].iloc[0] = 100

    for d, dm1 in zip(strat_index_comb_cluster.index[1:], strat_index_comb_cluster.index[:-1]):
        strat_index_comb_cluster['Level'].loc[d] = strat_index_comb_cluster['Level'].loc[dm1] * (1+strat_index_comb_cluster['Return'].loc[d])

    strat_index_comb_cluster['Level'].plot(figsize=(8, 5))   
   
    
    
#     SHARP RATIO 
    ret_y_comb_cluster = strat_index_comb_cluster['Level'].pct_change(52)
    vol_y_comb_cluster = strat_index_comb_cluster['Level'].pct_change(1).rolling(52).aggregate(np.std)*np.sqrt(52)


    SR_comb_cluster = ret_y_comb_cluster/vol_y_comb_cluster

    SR_mean_comb_cluster = (strat_index_comb_cluster['Level'].pct_change(1).mean()*52)/(strat_index_comb_cluster['Level'].pct_change(1).std()*np.sqrt(52))

    print(SR_mean_comb_cluster)    
    
#########################################
    
    
    
    
    
  #### BOOTSTRAP ###


frames = [strat_index_comb_ew['Level'], strat_index_comb_ivp['Level'], strat_index_comb_mvp['Level'], strat_index_comb_cluster['Level'],  strat_index_logit_ew['Level'], strat_index_logit_ivp['Level'], strat_index_logit_mvp['Level'], strat_index_logit_cluster['Level'], strat_index_svc_ew['Level'], strat_index_svc_ivp['Level'], strat_index_svc_mvp['Level'], strat_index_svc_cluster['Level'], strat_index_knn_ew['Level'], strat_index_knn_ivp['Level'], strat_index_knn_mvp['Level'], strat_index_knn_cluster['Level']]


backtests = pd.concat(frames, axis=1)
  
backtests.columns = ['Comb_EW', 'Comb_IVP', 'Comb_MVP', 'Comb_Cluster', 'Logit_EW', 'Logit_IVP', 'Logit_MVP', 'Logit_Cluster', 'SVC_EW', 'SVC_IVP', 'SVC_MVP', 'SVC_Cluster', 'KNN_EW', 'KNN_IVP', 'KNN_MVP', 'KNN_Cluster',]    
 


# 1) Para cada janela de tempo (com sobreposição), amostra com reposição 52 observações aleatórias.
# 2) Feito isso, calcula o Sharp Ratio dessa amostra aleatória 
# 3) Repete o passo 2 (dois) 500 vezes e guarda esses 500 Sharp Ratios
# 4) Tira a média dos 500 Sharp Ratios gerados dentro da primeira janela de tempo
#5) Repete os passos anteriores para todas as janelas de tempo
# 6) Gera uma distribuição de Sharp Ratios ao longo do tempo
# 7) Gera histograma de SR
SR={}
for model in backtests.columns:
    simulated_SR=[]
    for d1, d2 in zip(backtests.index[53:], backtests.index[:-52]) :
        fold = backtests[model].loc[d2:d1].pct_change(1).dropna()
        SRs=[]
        for _ in range(500):    
            sample = np.random.choice(fold, size=52) 
            SRs.append((sample.mean()*52/(sample.std()*np.sqrt(52))))
        
        simulated_SR.append(np.mean(SRs))

        SR[model] = simulated_SR
  
    
for i,model in enumerate(backtests.columns):
    fig, ax = plt.subplots()
    mu = np.mean(SR[model])
    median = np.median(SR[model])
    sigma = np.std(SR[model])
    textstr = '\n'.join((
    r'$\mu=%.2f$' % (mu, ),
    r'$\mathrm{median}=%.2f$' % (median, ),
    r'$\sigma=%.2f$' % (sigma, )))
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    plt.hist(SR[model])
    plt.title(model)
    plt.show()

###################################################################################
# Function to give some analysis

def GetPerformanceTable(IndexSeries, freq='Daily'):
    adju_factor = 252.0
    if freq == 'Monthly':
        adju_factor = 12.0
    elif freq == 'Weekly':
        adju_factor = 52.0

    Table = pd.Series(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                             'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                             '10th percentile in Vol Terms'])

    CleanIndexSeries = IndexSeries.dropna().sort_index()

    ER_index = pd.Series(index=CleanIndexSeries.index)
    ER_index[CleanIndexSeries.index[0]] = 100.0
    for d, d_minus_1 in zip(ER_index.index[1:], ER_index.index[:-1]):
        ER = CleanIndexSeries[d] / CleanIndexSeries[d_minus_1] - 1.0
        ER_index[d] = ER_index[d_minus_1] * (1 + ER)

    Table['Excess Return'] = (CleanIndexSeries[-1] / CleanIndexSeries[0]) ** (adju_factor / (len(CleanIndexSeries) - 1.0)) - 1
    Table['Volatility'] = (np.log(ER_index).diff(1).dropna()).std() * np.sqrt(adju_factor)
    Table['Sharpe'] = Table['Excess Return'] / Table['Volatility']
    Table['Sortino'] = Table['Excess Return'] / (np.sqrt(adju_factor) * (
    np.log(ER_index).diff(1).dropna()[np.log(ER_index).diff(1).dropna() < 0.0]).std())
    Table['Max Drawdown'] = max_dd(ER_index)
    Table['Max Drawdown in Vol Terms'] = max_dd(ER_index) / Table['Volatility']
    Table['5th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.05) / Table['Volatility']
    Table['10th percentile in Vol Terms'] = (ER_index.pct_change(1).dropna()).quantile(q=0.1) / Table['Volatility']
    return Table


def max_dd(ser):
    max2here = ser.expanding(min_periods=1).max()
    dd2here = ser / max2here - 1.0
    return dd2here.min()

###########################################################################################

descriptions = pd.DataFrame(index=['Excess Return', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown',
                                   'Max Drawdown in Vol Terms', '5th percentile in Vol Terms',
                                   '10th percentile in Vol Terms'])

# Gera gráfico de backtest dos 16 modelos diferentes
for i,model in enumerate(backtests.columns):
    plt.figure(i+16)
    backtests[model].plot()
    df_mercado['Index'].plot()
    plt.title(model)
    plt.show()

    descriptions[model] = GetPerformanceTable(backtests[model],'Weekly')

descriptions.to_excel('Stats.xlsx')

# Calculando o beta das estratégias com S&P

betas = pd.DataFrame(index=backtests.columns, columns=["Beta",'PValue'])

for model in (backtests.columns):
    x = df_mercado['Index'].pct_change(1)
    x = sm.add_constant(x)
    x = x.dropna()

    y = backtests[model].pct_change()
    y = y.dropna()

    regression = sm.OLS(y,x)

    results = regression.fit()

    betas['Beta'].loc[model] = results.params['Index']
    betas['PValue'].loc[model] = results.pvalues['Index']

betas.to_excel('betas.xlsx')



