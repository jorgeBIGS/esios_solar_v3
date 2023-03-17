import time
from fileUtils import resetFiles, load_data
from Utils import getSeed
from CNN import *
from LR import *
from XGBoost import *
from SVR import *
import warnings
warnings.filterwarnings("ignore")

def runExperimentUM(epochs, batch_size, shift, forecast_horizon=1, past_history=48, cv=10):
    #Reset files and use the seed.
    resetFiles()
    getSeed()

    # Load data for Univariant models, we only load the period when we have all the data aviable.
    file_data = 'solar_multi.csv'
    all_data, df2 = load_data('U', file_data)
    all_data = all_data.loc[:'2021-04-01 23:00:00']
    df2 = df2.loc[:'2021-04-01 23:00:00']
    folder_split = round(len(df2)/cv)
    df_to_scale = df2[:folder_split].copy()
    train_split = round(len(df_to_scale)*0.8)

    inicio = time.time()
    LR_Univariant(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon, past_history)

    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado LR_U:" + str(fin-inicio))

    inicio = time.time()
    CNN(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, 'U', forecast_horizon, past_history, shift)

    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado CNN_U:" + str(fin-inicio))
        
    inicio = time.time()
    CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, 'U', forecast_horizon, past_history, shift)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado CNN_LSTM_U:" + str(fin-inicio))

    inicio = time.time()
    XGBoost('U', cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado XGBoost_U:" + str(fin-inicio))

    inicio = time.time()
    SVR_U(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon,
                past_history)

    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado SVR_U:" + str(fin-inicio))

    all_data, df2 = load_data("M", file_data)
    all_data = all_data.loc[:'2021-04-01 23:00:00']
    df2 = df2.loc[:'2021-04-01 23:00:00']
    folder_split = round(len(df2)/cv)
    df_to_scale = df2[:folder_split].copy()
    train_split = round(len(df_to_scale)*0.8)

    inicio = time.time()
    CNN(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, 'M', forecast_horizon, past_history, shift)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("Tiempo procesado CNN_M:" + str(fin-inicio))

    inicio = time.time()
    CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, 'M', forecast_horizon, past_history, shift)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado CNN_LSTM_M:" + str(fin-inicio))
        
    inicio = time.time()
    LR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
                past_history, shift, 'M')
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado LR_M:" + str(fin-inicio))

    inicio = time.time()
    XGBoost('M', cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado XGBoost_M:" + str(fin-inicio))

    inicio = time.time()
    SVR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
                past_history, shift, 'M')
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write("\nTiempo procesado SVR_M:" + str(fin-inicio))

def runExperimentForecastedIrradiation(epochs, batch_size, shift, type, forecast_horizon=1, past_history=48, cv=10, resetFilesFlag=False):
    #Reset files and use the seed.
    if(resetFilesFlag):
        resetFiles()
    getSeed()

    hours = shift if shift >0 else 1
    # Load data for Univariant models, we only load the period when we have all the data aviable.
    file_data = type + str(hours) +'h.csv'
    all_data, df2 = load_data("M", file_data)
    all_data = all_data.loc[:'2021-04-01 23:00:00']
    df2 = df2.loc[:'2021-04-01 23:00:00']
    folder_split = round(len(df2)/cv)
    df_to_scale = df2[:folder_split].copy()
    train_split = round(len(df_to_scale)*0.8)

    inicio = time.time()
    CNN(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, type, forecast_horizon, past_history, shift)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write(f"Tiempo procesado CNN_{type}:" + str(fin-inicio))

    inicio = time.time()
    CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                        batch_size, train_split, type, forecast_horizon, past_history, shift)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write(f"\nTiempo procesado CNN_LSTM_{type}:" + str(fin-inicio))
        
    inicio = time.time()
    LR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
                past_history, shift, type)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write(f"\nTiempo procesado LR_{type}:" + str(fin-inicio))

    inicio = time.time()
    XGBoost(type, cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write(f"\nTiempo procesado XGBoost_{type}:" + str(fin-inicio))

    inicio = time.time()
    SVR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
                past_history, shift, type)
    fin = time.time()
    with open('datos_prueba/tiempos.txt', 'a') as f:
        f.write(f"\nTiempo procesado SVR_{type}:" + str(fin-inicio))