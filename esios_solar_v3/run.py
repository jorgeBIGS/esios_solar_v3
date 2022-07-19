#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:31:32 2021

@author: tomas
"""
import time
from Utils import *
from CNN import *
from LR import *
from XGBoost import *
from SVR import *
import warnings
warnings.filterwarnings("ignore")

#Reset files and use the seed.
resetFiles()
getSeed()

# Load data for Univariant models, we only load the period when we have all the data aviable.
epochs = 100
batch_size = 128
forecast_horizon = 1
past_history = 48
shift = 24
cv = 10

all_data, df2 = load_data('U')
all_data = all_data.loc[:'2021-04-01 23:00:00']
df2 = df2.loc[:'2021-04-01 23:00:00']
folder_split = round(len(df2)/cv)
df_to_scale = df2[:folder_split].copy()
train_split = round(len(df_to_scale)*0.8)

# inicio = time.time()
# LR_Univariant(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon, past_history)

# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado LR_U:" + str(fin-inicio))

inicio = time.time()
CNN(all_data, df2, folder_split, cv, epochs, 
                    batch_size, train_split, 'U', forecast_horizon, past_history)

fin = time.time()
with open('datos_prueba/tiempos.txt', 'a') as f:
    f.write("\nTiempo procesado CNN_U:" + str(fin-inicio))
    
inicio = time.time()
CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                    batch_size, train_split, 'U', forecast_horizon, past_history)
fin = time.time()
with open('datos_prueba/tiempos.txt', 'a') as f:
    f.write("\nTiempo procesado CNN_LSTM_U:" + str(fin-inicio))

# inicio = time.time()
# XGBoost('U', cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history)
# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado XGBoost_U:" + str(fin-inicio))

# inicio = time.time()
# SVR_U(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon,
#              past_history)

# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado SVR_U:" + str(fin-inicio))

all_data, df2 = load_data("M")
all_data = all_data.loc[:'2021-04-01 23:00:00']
df2 = df2.loc[:'2021-04-01 23:00:00']
folder_split = round(len(df2)/cv)
df_to_scale = df2[:folder_split].copy()
train_split = round(len(df_to_scale)*0.8)

inicio = time.time()
CNN(all_data, df2, folder_split, cv, epochs, 
                    batch_size, train_split, 'M', forecast_horizon, past_history)
fin = time.time()
with open('datos_prueba/tiempos.txt', 'a') as f:
    f.write("Tiempo procesado CNN_M:" + str(fin-inicio))

inicio = time.time()
CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                    batch_size, train_split, 'M', forecast_horizon, past_history)
fin = time.time()
with open('datos_prueba/tiempos.txt', 'a') as f:
    f.write("\nTiempo procesado CNN_LSTM_M:" + str(fin-inicio))
    
# inicio = time.time()
# LR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
#              past_history, shift)
# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado LR_M:" + str(fin-inicio))

# inicio = time.time()
# XGBoost('M', cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history)
# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado XGBoost_M:" + str(fin-inicio))

# inicio = time.time()
# SVR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
#              past_history)
# fin = time.time()
# with open('datos_prueba/tiempos.txt', 'a') as f:
#     f.write("\nTiempo procesado SVR_M:" + str(fin-inicio))