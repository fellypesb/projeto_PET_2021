#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  10 14:36:17 2021

@author: Fellype Siqueira Barroso
"""

#%%

'''
-------------------------------------Estudo de caso E2---------------------------------------

Neste estudo de caso utilizou-se 4 variáveis preditoras.

Variáveis preditoras: Ângulo Zenital, Irradiância de céu claro, Umidade relativa, Temperatura
Variável alvo: Irradiância Solar.


'''

# Importação para o estudo de caso E2
import pandas as pd

train = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/train_local_BR.csv',
                    usecols=['ghi', 'solar_zenith_angle', 'clearsky_ghi', 'relative_humidity', 'temperature'])
valid = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/valid_local_BR.csv',
                    usecols=['ghi', 'solar_zenith_angle', 'clearsky_ghi', 'relative_humidity', 'temperature'])

#%%

'''
-------------------------------------Estudo de caso E3---------------------------------------

Neste estudo de caso utilizou-se 5 variáveis preditoras.


Variáveis preditoras: Ângulo Zenital, Irradiância de céu claro, Umidade relativa, Temperatura, Horário e mês dos registros
Variável alvo: Irradiância Solar

'''

# Importação para o estudo de caso E3
import pandas as pd

train = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/train_local_BR.csv',
                    usecols=['hour', 'month', 'ghi', 'solar_zenith_angle', 'clearsky_ghi', 'relative_humidity', 'temperature'])
valid = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/valid_local_BR.csv',
                    usecols=['hour', 'month', 'ghi', 'solar_zenith_angle', 'clearsky_ghi', 'relative_humidity', 'temperature'])

#%%
'''
-------------------------------------Estudo de caso E4---------------------------------------

Neste estudo de caso utilizou-se 2 variáveis preditoras.


Variáveis preditoras: Umidade relativa, Temperatura
Variável alvo: Irradiância Solar

'''

# Importação para o estudo de caso E4
import pandas as pd

train = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/train_local_BR.csv',
                    usecols=['ghi', 'relative_humidity', 'temperature'])
valid = pd.read_csv('https://github.com/fellypesb/projeto_PET_2021/raw/main/Dados/local_BR/valid_local_BR.csv',
                    usecols=['ghi', 'relative_humidity', 'temperature'])



#%%

# Divisão entre variáveis previsoras e variável alvo - E2, E3, E4

X_train = train.drop(['ghi'], axis=1)
y_train = train['ghi']
X_valid = valid.drop(['ghi'], axis=1)
y_valid = valid['ghi']

#%%

# Escalonamento das variáveis - Método MinMax - E2, E3, E4

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0.1,0.9))

X_train_norm = scaler.fit_transform(X_train)
X_valid_norm = scaler.transform(X_valid)

y_train_norm = scaler.fit_transform(y_train.to_numpy().reshape(-1,1))
y_valid_norm = scaler.transform(y_valid.to_numpy().reshape(-1,1))


#%%
# Aplicação da janela deslizante - E2, E3, E4

import numpy as np

def sliding_window(X_train, y_train, X_valid, delay):
    '''
        Função que aplica a técnica de transformação dos dados em função de uma janela de tempo (delay)
        
        Parâmetros:
            X_train: Conjunto de treinamento.
            y_train: Conjunto de saídas desejadas para o conjunto de treinamento.
            X_valid: Conjunto de validação.
            y_valid: Conjunto de saídas desejadas para o conjunto de validação.
            Delay: Tamanho da janela para ser aplicado aos dados.
    '''
    
    X_valid = np.concatenate((X_train[-delay:], X_valid))
    
    size_data = len(X_train)
    Xt, Xv, yt= [], [], []
    
    for i in range(delay, size_data):
        Xt.append(X_train[i-delay:i])
        yt.append(y_train[i])
        
    size_data = len(X_valid)
    for i in range(delay, size_data):
        Xv.append(X_valid[i-delay:i])
        
        
    return np.asarray(Xt), np.asarray(yt), np.asarray(Xv)

DELAY = 3 # Valores de delay -> [3,5,8,13]

X_train, y_train, X_valid = sliding_window(X_train_norm, y_train_norm, X_valid_norm, DELAY)

#%%
# Otimização dos hiperparâmetros - E2, E3, E4


from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from kerastuner import HyperParameters as hp
from kerastuner import HyperModel
import random as python_random
import tensorflow as tf
import time

SEED = 2021

np.random.seed(SEED)
tf.random.set_seed(SEED)
python_random.seed(SEED)


def build_model(hp):
        model = Sequential(name='Elman_RNN')
        
        
        model.add(SimpleRNN(units=hp.Int(name='units',
                                         min_value=2,
                                         max_value=50,
                                         step=2),
                            #[batch, timesteps, feature]
                            input_shape=(X_train.shape[1], X_train.shape[2]),
                            activation=hp.Choice('activation',
                                                 values=['relu', 'sigmoid', 'tanh']),
                            name='Recurrent_layer'))
        
        model.add(Dense(units=1,
                        activation='linear',
                        name='output_layer'))
        
        model.compile(optimizer=SGD(learning_rate=hp.Choice('learning_rate',
                                                            values=[1e-4, 1e-3, 1e-2, 3e-2, 5e-2, 7e-2,1e-1]),
                                    momentum=hp.Choice('momentum',
                                                       values=[9e-1, 4e-1, 2e-1, 1e-1, 1e-3, 5e-3, 1e-4])),
                     loss='mse',
                     metrics=[tf.keras.metrics.RootMeanSquaredError(),
                              tf.keras.metrics.MeanAbsolutePercentageError()])
    
        return model

tuner = RandomSearch(build_model,
                     objective='loss',
                     max_trials=60,
                     executions_per_trial=1,
                     seed=SEED)

tuner.search_space_summary()

callback = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5)

tuner.search(X_train,
             y_train,
             epochs=100,
             callbacks=[callback],
             use_multiprocessing=True)

tuner.results_summary()
#%%

# Avaliação dos melhores modelos - E2, E3, E4

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import random as python_random
import tensorflow as tf
import time

from sklearn.metrics import r2_score, \
mean_absolute_percentage_error, mean_squared_error

SEED = 2021

np.random.seed(SEED)
tf.random.set_seed(SEED)
python_random.seed(SEED)

model = Sequential(name='Elman_RNN')


model.add(SimpleRNN(units=48,     
                    activation='relu',  
                    input_shape=(X_train.shape[1], X_train.shape[2]), #[batch, timesteps, feature]
                    name='Recurrent_layer'))

model.add(Dense(units=1,
                activation='linear',
                name='output_layer'))

model.compile(optimizer=SGD(learning_rate=0.03, momentum=0.001),    
             loss='mse')

callback = EarlyStopping(monitor='loss', patience=10, min_delta=1e-5)

start = time.time()
hist = model.fit(X_train, y_train, epochs=100, callbacks=[callback], use_multiprocessing=True, verbose=2)
stop = time.time()

y_pred_train = model.predict(X_train, use_multiprocessing=True)
y_pred_valid = model.predict(X_valid, use_multiprocessing=True)

# Resultados para o conjunto de treino e validação

results_train = pd.DataFrame({'Metricas':['r2', 'RMSE', 'MAPE', 'TimeRun'], 
                    'Resultados':[r2_score(y_train, y_pred_train),
                                  mean_squared_error(y_train, y_pred_train, squared=False),
                                  mean_absolute_percentage_error(y_train, y_pred_train),
                                  stop-start]})

results_valid = pd.DataFrame({'Metricas':['r2', 'RMSE', 'MAPE', 'TimeRun'], 
                    'Resultados':[r2_score(y_valid_norm, y_pred_valid),
                                  mean_squared_error(y_valid_norm, y_pred_valid, squared=False),
                                  mean_absolute_percentage_error(y_valid_norm, y_pred_valid),
                                  stop-start]})

print('-------- Train ----------\n', results_train)
print('\n-------- Valid ----------\n', results_valid)




#%% 

# desnormalização dos dados - E2, E3, E4

x1 = y_valid
x2 = scaler.inverse_transform(y_pred_valid)


#%%

# plot dos gráficos de predições - E2, E3, E4

import matplotlib.pyplot as plt

plt.figure(figsize=(13,5), dpi=300)
#plt.title('Melhor modelo E1', fontsize=12, weight='bold')
plt.xlabel('Tempo (h)', fontsize=12, weight='bold')
plt.ylabel('Irradiância (w/m²)', fontsize=12, weight='bold')
plt.plot(x1,'-', label='DADOS REAIS', linewidth=2)
plt.plot(x2,'--', label='PREDIÇÕES ELMAN', linewidth=2)
#error = abs(x2 - x1)
# plt.plot(error, '-.', label='ERROR', linewidth=2)
plt.legend(loc='best', shadow=True)
plt.xlim(0, 427)
plt.ylim(top=1200)
plt.grid(alpha=0.4)
#plt.savefig('modelo.png', dpi=300)