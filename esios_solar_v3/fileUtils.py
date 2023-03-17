import os
from Utils import media
import matplotlib.pyplot as plt
import csv
import pandas as pd
import datetime
import numpy as np
from normalization import denormalize, normalize

def dibujaGraph(train_split, folder_split, df2, iteration, 
                realData, forecastedData, esiosForecast, modelo, type,
                past_history):
  """
  Draw a graph with a comparison of the real values, the esios' predictions and 
  the proposed model's predictions for a folder.
  :param type: String that represents the kind of energy predicted.
  :param train_split: Integer that represents the point of the folder where 
  the test data start.
  :param folder_split: Integer that represents the size of a folder.
  :param df2: Dataframe to get the test part's dates of the folder.
  :param iteration: Integer that represents the folder we are drawing about.
  :param realData: List with the real data for every proposed model prediction.
  :param forecastedData: List with the proposed model predictions.
  :param esiosForecast: List with the esios predictions for every proposed model prediction.
  :param modelo: String that represents the name of the model used to predict.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  """
  colors = plt.get_cmap('Set2', 8)
  initial_moment = pd.to_datetime(df2.iloc[folder_split*iteration + train_split + past_history].name)
  numRegistrosAllData = len(realData)
  numRegistrosPredictions = len(forecastedData)
  x = [initial_moment + datetime.timedelta(hours=j) for j in range(0, numRegistrosAllData, 1)]
  y = [initial_moment + datetime.timedelta(hours=j) for j in range(0, numRegistrosPredictions, 1)]

  plt.figure(figsize=(32,8))
  plt.title(modelo)
  plt.xlabel('Fecha', fontsize=18)
  plt.ylabel('Predicción Energía Solar', fontsize=18)

  forecastedData = np.asarray(forecastedData)
  esiosForecast = np.asarray(esiosForecast)
    
  plt.plot(x, realData, c = colors(0), label = "Datos reales")
  plt.plot(y, forecastedData, c = colors(6), linestyle = "--", label = "Predicción modelo Solar")
  plt.plot(y, esiosForecast, c = colors(5), linestyle = "-.", label = "Predicción esios Solar")
  plt.legend()
    
  plt.savefig("datos_prueba/" + type + "/" + str(iteration) + "_" + modelo + 
              '.png', bbox_inches='tight')
  plt.show()

def resetFiles():
  """
  Reset the content of each result file.
  """
  archivos = ['wape_XGBOOST', 'wape_CNN', 'wape_CNN_LSTM', 'wape_XGBOOST_clean', 
              'wape_CNN_clean', 'wape_CNN_LSTM_clean','Valores_CNN', 
              'Valores_CNN_LSTM', 'Valores_LR', 'Valores_XGBOOST',
              'Valores_SVR', 'media', 'wape_LR',
              'wape_LR_clean', 'wape_SVR', 'wape_SVR_clean']
  
  if(os.path.isdir('datos_prueba') == False):
    os.mkdir('datos_prueba')
  if(os.path.isdir('datos_prueba/M') == False):
    os.mkdir('datos_prueba/M')
  if(os.path.isdir('datos_prueba/U') == False):
    os.mkdir('datos_prueba/U')
  if(os.path.isdir('datos_prueba/F') == False):
    os.mkdir('datos_prueba/F')
  if(os.path.isdir('datos_prueba/FM') == False):
    os.mkdir('datos_prueba/FM')

  with open('datos_prueba/' + 'tiempos.txt', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([])

  for archivo in archivos:
    with open('datos_prueba/M/' + archivo, 'w') as f:
      if("clean" in archivo):
        writer = csv.writer(f)
        writer.writerow(['MAE_MODEL', 'WAPE_MODEL', 'MAE_ESIOS', 'WAPE_ESIOS'])
      elif("Valores" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Real', 'ESIOS', 'Modelo', 'SEASON'])
      elif("media" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Modelo', 'Media MAE Modelo', 'Media WAPE Modelo', 'Media MAE ESIOS', 'Media WAPE ESIOS'])
    with open('datos_prueba/U/' + archivo, 'w') as f:
      if("clean" in archivo):
        writer = csv.writer(f)
        writer.writerow(['MAE_MODEL', 'WAPE_MODEL', 'MAE_ESIOS', 'WAPE_ESIOS'])
      elif("Valores" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Real', 'ESIOS', 'Modelo'])
      elif("media" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Modelo', 'Media MAE Modelo', 'Media WAPE Modelo', 'Media MAE ESIOS', 'Media WAPE ESIOS'])
    with open('datos_prueba/F/' + archivo, 'w') as f:
      if("clean" in archivo):
        writer = csv.writer(f)
        writer.writerow(['MAE_MODEL', 'WAPE_MODEL', 'MAE_ESIOS', 'WAPE_ESIOS'])
      elif("Valores" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Real', 'ESIOS', 'Modelo'])
      elif("media" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Modelo', 'Media MAE Modelo', 'Media WAPE Modelo', 'Media MAE ESIOS', 'Media WAPE ESIOS'])
    with open('datos_prueba/FM/' + archivo, 'w') as f:
      if("clean" in archivo):
        writer = csv.writer(f)
        writer.writerow(['MAE_MODEL', 'WAPE_MODEL', 'MAE_ESIOS', 'WAPE_ESIOS'])
      elif("Valores" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Real', 'ESIOS', 'Modelo'])
      elif("media" in archivo): 
        writer = csv.writer(f)
        writer.writerow(['Modelo', 'Media MAE Modelo', 'Media WAPE Modelo', 'Media MAE ESIOS', 'Media WAPE ESIOS'])
          
def saveResults(modelData, esiosData, model, type, shift):
  """
  Record in a file the prediction results for each folder and each model.
  :param modelData: List with the MAE and the WAPE error for the proposed model
  in the folder.
  :param esiosData: List with the MAE and the WAPE error for ESIOS in the 
  folder.
  :param datos: String that represents the name of the model.
  :param type: String that represents the kind of energy predicted.
  """
  media_modelo_mae = str(media(modelData[0][-1]))
  media_modelo_wape = str(media(modelData[1][-1]))
  media_esios_mae = str(media(esiosData[0][-1]))
  media_esios_wape = str(media(esiosData[1][-1]))

  if shift != 0 or shift != 1:
    media_esios_mae = '-'
    media_esios_wape = '-'
    
  with open('datos_prueba/' + type + '/wape_' + model, 'a') as f:
    f.write("MAES, WAPES " + str(model) + "\n")
        
    f.write("MAE, WAPE MEDIA "  + model + ": " + media_modelo_mae + ", " + media_modelo_wape)
    f.write("\n")
    f.write("MAE, WAPE MEDIA ESIOS " + ": " + media_esios_mae + ", " + media_esios_wape)
    f.write("\n")
    f.write("\n------------------------------------------------------------\n")
        
    f.write("\n\n")
        
  with open('datos_prueba/' + type + '/wape_' + model + "_clean", 'a') as f:        
    writer = csv.writer(f)
    writer.writerow([media_modelo_mae, media_modelo_wape, media_esios_mae, media_esios_wape])
        
def saveResultsAverage(maeWapesModel, maeWapesEsios, modelo, datos, shift):
  """
  Record in a file the average of the results for each folder and each model.
  :param maeWapesModel: List with Mae and Wape results for each folder predicted by the proposed model.
  :param maeWapesEsios: List with Mae and Wape results for each folder predicted by Esios.
  :param modelo: String that represents the name of the model.
  :param tipo: String that represents the kind of energy predicted.
  :param shift: Integer that represents the number of steps in the future from which we make predictions.
  """
  media_modelo_mae = str(media(maeWapesModel[0]))
  media_modelo_wape = str(media(maeWapesModel[1]))
  media_esios_mae = str(media(maeWapesEsios[0]))
  media_esios_wape = str(media(maeWapesEsios[1]))

  if shift > 1: 
    media_esios_mae = '-'
    media_esios_wape = '-'
    
  with open('datos_prueba/' + datos + '/media', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([modelo, media_modelo_mae, media_modelo_wape, media_esios_mae, media_esios_wape])
    
def saveValues(real, esios, predicted, modelo, type):
  """
  Write the comparison between the actual value, the one predicted by 
  esios and the one predicted by the proposed model for each prediction 
  in a file.
  :param real: Double. Real value for every prediction.
  :param esios: Double. Esios prediction for every model prediction.
  :param modelo: Double. Proposed model's predictions.
  :param type: String. U->Univariate, M->Multivariate
  """
  with open('datos_prueba/' + type + '/Valores_' + modelo, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([real[0], esios[0], predicted[0]])

def load_data_stacked(tipo, shift):
  """
  Load the data to train XGBoost and LR multivariate models.
  :param tipo: String, indicates the kind of XGBoost model that we will use, 
  this param can be U -> Univariate, or M -> Multivariate.
  :param shift: int, Specifies how many entries are between the last entry of 
  the past_history and the first of the forecast_horizon. 
  """
  if shift == 0: shift = 1
  path = os.getcwd() + '/datasets_proyecto/XGBOOST_SOLAR_' + tipo + '_' + str(shift) + 'H'
     
  lista_train = []
  lista_test = [] 
  lista_train_y = []
  lista_test_y = []
  
  ruta_actual = path + '/x_train'
  for archivo in sorted(os.listdir(ruta_actual)):
    lista_train.append(pd.read_csv(ruta_actual + "/" + archivo))
      
  ruta_actual = path + '/y_train'
  for archivo in sorted(os.listdir(ruta_actual)):
    lista_train_y.append(pd.read_csv(ruta_actual + "/" + archivo))
            
  ruta_actual = path + '/y_test'
  for archivo in sorted(os.listdir(ruta_actual)):
    lista_test_y.append(pd.read_csv(ruta_actual + "/" + archivo))

  ruta_actual = path + '/x_test'
  for archivo in sorted(os.listdir(ruta_actual)):
    lista_test.append(pd.read_csv(ruta_actual + "/" + archivo))

  return lista_train, lista_train_y, lista_test, lista_test_y

def storeForecastValues(prediction, norm_params, y_test, all_data,
                count, forecastedData, realData, train_split, 
                folder_split, esiosForecast, iteration, type, 
                past_history, forecast_horizon, modelo):
    """
    Stores values for every forecast value.
    :param prediction: float, forecasted value.
    :param norm_params: dict, min-max data for normalize and denormalize.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param y_test: int, specifies the size of the forest to be trained.
    :param count: int, index of the value.
    :param forecastedData: list, forecasted values.
    :param realData: list, real values.
    :param train_split: int, index to split data into train data and validate data.
    :param folder_split: int, index to split data into folders
    :param esiosForecast: list, esios forecasted values
    :param iteration: int, iteration we are processing.
    :param type: String, type of data we are using, M -> Multicariate, U-> Univariate
    :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
    :param forecast_horizon: int, the number of steps in the future that the model will forecast.
    """
    predictedf = denormalize(prediction, norm_params)
    forecastedData.append(predictedf)
    realesf = y_test[count]
    realData.append(realesf)
        
    predictedPoint = all_data[train_split + count + folder_split*iteration +
                              past_history:train_split + folder_split*iteration + 
                              count + past_history + forecast_horizon]
    
    esiosf = predictedPoint.iloc[0]['Generación prevista Solar']
    esiosForecast.append(esiosf)

    realesfl = []
    esiosfl = []
    predictedfl = []
    
    realesfl.append(realesf)
    esiosfl.append(esiosf)
    predictedfl.append(predictedf)

    saveValues(realesfl, esiosfl, predictedfl, modelo, type)
    
def load_data(tipo, dataset):
    """
      Load the data to train models and validate the results.
      :param tipo: Indicates the kind of XGBoost model that we will use, 
      this param can be U -> Univariate, or M -> Multivariate.
      """
    ruta_actual = os.getcwd()
    archivo = ruta_actual + '/datasets_proyecto/' + dataset
    all_data = pd.read_csv(archivo)
    all_data = all_data.set_index("datetime_utc")
    df2 = all_data.copy()
        
    if(tipo == "U"):
        df2 = df2.filter(['Generación T.Real Solar'])
    else:
        df2 = df2.drop('Generación prevista Solar', axis='columns')
        df2 = df2.fillna(df2.mean())
    return all_data, df2

def prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, iteration):
  x_train = lista_train[iteration]
  y_train = lista_train_y[iteration]['Generación T.Real Solar']
  x_test = lista_test[iteration]
  y_test = lista_test_y[iteration]['Generación T.Real Solar']

  norm_params = {}
  norm_params['mean'] = x_train.mean().mean()
  norm_params['std'] = x_train.std().std()
  norm_params['max'] = x_train.max().max()
  norm_params['min'] = x_train.min().min()
        
  x_train = normalize(x_train, norm_params)
  y_train = normalize(y_train, norm_params)
  x_test = normalize(x_test, norm_params)

  return x_train, y_train, x_test, y_test, norm_params

def prepareTrain(folder_split, df2, iteration, train_split, shifted, past_history, forecast_horizon):
  """
  Allows to save the predicted results by the proposed model for every subset
  of the test part inside the folder.
  :param folder_split: Integer that represents the size of a folder.
  :param df2: Dataframe with the solar energy data.
  :param iteration: Integer that represetns the folder we are using.
  :param train_split: Integer that represents the point of the folder where 
  the test data start.
  :return: x_train2, y_train2, x_test2 and y_test2 subsets to train and evaluate 
  the proposed model for a folder.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  :param shift: int, Specifies how many entries are between the last entry of 
  the past_history and the first of the forecast_horizon. 
  """
  df_to_scale = df2[folder_split*iteration:folder_split*(iteration+1)].copy()   
  df_to_scale_train = df_to_scale[:train_split].copy()
  df_to_scale_test = df_to_scale[train_split:].copy()
    
  norm_params = {}
  norm_params['mean'] = df_to_scale_train.mean().mean()
  norm_params['std'] = np.std(df_to_scale_train)[0]
  norm_params['max'] = df_to_scale_train.max().max()
  norm_params['min'] = df_to_scale_train.min().min()
        
  scaled_data = normalize(df_to_scale_train, norm_params)
  scaled_data_test = normalize(df_to_scale_test, norm_params)
    
  x_train, y_train = [], []
  x_test, y_test = [], []
    
  for i in range(0, len(scaled_data) - past_history - forecast_horizon - shifted):
    a = scaled_data[i:(i+past_history)]
    x_train.append(a)
    y_train.append(scaled_data[i + past_history + shifted: i + past_history+ shifted + forecast_horizon]['Generación T.Real Solar'])
    
  x_train = np.asarray(x_train)
  y_train = np.asarray(y_train)

  x_test, y_test = [], []

  for i in range(0, len(scaled_data_test) - past_history - forecast_horizon - shifted):
    a = scaled_data_test[i:(i+past_history)]
    x_test.append(a)
    y_test.append(scaled_data_test[i + past_history + shifted: i + past_history+ shifted + forecast_horizon]['Generación T.Real Solar'])
       
  x_test = np.asarray(x_test)
  y_test = np.asarray(y_test)
  
  return x_train, y_train, x_test, y_test, norm_params