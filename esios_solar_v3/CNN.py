#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:37:54 2021

@author: tomas
"""
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, LSTM, Input
from Utils import denormalize, prepareTrain, saveResults, saveResultsAverage, saveValues, getResults, dibujaGraph

def inicializaModelo_CNN(x_train2, forecast_horizon):
    """
  Initializes the CNN proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast
  """
    inp = Input(shape=(x_train2.shape[-2:]))
    
    x = Conv1D(128, 11, activation='relu', padding='same')(inp)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, 9, activation='relu', padding='same')(x)
    x = Conv1D(64, 9, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.1)(x)
    x = Conv1D(32, 7, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)

    x = Flatten()(x)
    x = Dense(50)(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
    
    model.compile(optimizer='adam', loss='mae')
    return model

def inicializaModelo_24h(x_train2, forecast_horizon):
    """
  Initializes the CNN proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast
  """
    inp = Input(shape=(x_train2.shape[-2:]))
    
    x = Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = Conv1D(128, 11, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.15)(x)
    x = Conv1D(128, 13, activation='relu', padding='same')(x)
    x = Conv1D(64, 7, activation='relu', padding='same')(x)
    x = Conv1D(64, 9, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Conv1D(64, 7, activation='relu', padding='same')(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)

    x = Flatten()(x)
    x = Dense(50)(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
    
    model.compile(optimizer='adam', loss='mae')
    return model

def inicializaModelo_CNN_LSTM(x_train2, forecast_horizon):
    """
  Initializes the CNN_LSTM proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  """
    inp = Input(shape=x_train2.shape[-2:])
    
    x = Conv1D(128, 7, activation='relu', padding='same')(inp)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = Conv1D(128, 11, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, 7, activation='relu', padding='same')(x)
    x = Conv1D(64, 9, activation='relu', padding='same')(x)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)

    x = LSTM(256, return_sequences=True)(x) 
    x = Dropout(0.1)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = LSTM(64, return_sequences=True)(x)
    
    x = Flatten()(x)
    x = Dense(50)(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
    
    model.compile(optimizer='adam', loss='mae')
    
    return model

def inicializaModelo_CNN_24h(x_train2, forecast_horizon):
    """
  Initializes the CNN proposed model.
  :param x_train2: List, contains the training data to check its shape.
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  """
    inp = Input(shape=(x_train2.shape[-2:]))
    
    x = Conv1D(128, 5, activation='relu', padding='same')(inp)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = Conv1D(128, 11, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, 7, activation='relu', padding='same')(x)
    x = Conv1D(64, 9, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(128, 9, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = Dropout(0.15)(x)
    x = Conv1D(64, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 7, activation='relu', padding='same')(x)
    x = Dropout(0.1)(x)
    x = Conv1D(32, 5, activation='relu', padding='same')(x)
    x = Conv1D(32, 7, activation='relu', padding='same')(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)

    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(32, return_sequences=True)(x)

    x = Flatten()(x)
    x = Dense(50)(x)
    x = Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inp, outputs=x)
    
    model.compile(optimizer='adam', loss='mae')
    return model

def storeForecastValues(prediction, norm_params, y_test, all_data, count, model, 
                forecastedData, realData, train_split, folder_split, esiosForecast, iteration,
                type, past_history, forecast_horizon):
    """
    Allows to save the predicted results by the proposed model for every subset
    of the test part inside the folder.
    :param prediction: float, predict value for the model.
    :param norm_params: tuple, contains params mean, std, max, min.
    :param y_test: List, contains the real data for every prediction.
    :param all_data: Dataframe, contains all the data.
    :param count: int, represents the prediction that we are saving.
    :param model: String,represents the name of the proposed model used.
    :param forecastedData: List, contains the proposed model predictions
    for every subset of the test data part inside the folder.
    :param realData: List, contains the real values for every subset 
    of the test data part inside the folder.
    :param train_split: int, represents the point of the folder where 
    the test data start.
    :param folder_split: int, represents the size of a folder.
    :param esiosForecast: List, contains the Esios predictions for every subset 
    of the test data part inside the folder.
    :param tipo: String, represents the kind of energy predicted.
    :param iteration: int, represents the folder we are using.
    :param type: String, indicates the kind of data we are using, U-> Univariate,
    M->Multivariate.
    :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
      :param forecast_horizon: int, the number of steps in the future that the model will forecast.
    """
    # Denormalizes the prediction value to compare it.
    predictedf = denormalize(prediction, norm_params)
    forecastedData.append(predictedf)
    real = y_test[count]
    realesf = denormalize(real, norm_params)
    realData.append(realesf)

    predictedPoint = all_data[train_split + count + folder_split*iteration +
                              past_history:train_split + folder_split*iteration + 
                              count + past_history + forecast_horizon]

    esiosf = predictedPoint['Generación prevista Solar']
    esiosForecast.append(esiosf)

    # Stores the values for this prediction.
    saveValues(realesf, esiosf, predictedf, model, type)
    
def comparePreds(norm_params, preds, y_test, all_data, model, 
                forecastedData, realData, train_split, folder_split,
                esiosForecast, iteration,
                maeWape, maeWape_esios, df2, type, past_history, forecast_horizon):
    """
    Allows to save the results of the expirements exploring their values and print
    the graph where the differences between real, esios and experiment values are compared. 
    :param preds: list, predictions of the model for x_test.
    :param norm_params: tuple, contains params mean, std, max, min.
    :param y_test: List, contains the real data for every prediction.
    :param all_data: Dataframe object, contains all the data.
    :param model: String, represents the name of the proposed model used.
    :param forecastedData: List, contains the proposed model predictions
    for every subset of the test data part inside the folder.
    :param realData: List, contains the real values for every subset s
    of the test data part inside the folder.
    :param train_split: int, represents the point of the folder where 
    the test data start.
    :param folder_split: int, represents the size of a folder.
    :param esiosForecast: List, contains the Esios predictions for every subset 
    of the test data part inside the folder.
    :param iteration: int, represetns the folder we are using.
    :param type: String, indicates the kind of data we are using, U-> Univariate,
    M->Multivariate.
    :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
    :param forecast_horizon: int, the number of steps in the future that the model will forecast.
    """

    # Validates the model using real data, forecasted test data and esios forecast data.
    # For every prediction, we denormalize the values and compare the real entry with the 
    # forecast entry and the esios predictions for that instant. Then we store the results 
    # of MAE and WAPE for every folder
    for count, prediction in enumerate(preds):
        storeForecastValues(prediction, norm_params, y_test, all_data, 
                    count, model, forecastedData, realData, train_split, folder_split, esiosForecast, iteration, 
                    type, past_history, forecast_horizon)
    
    # This method obtains the MAE and WAPE results from the data.
    getResults(realData, forecastedData, maeWape)
    getResults(realData, esiosForecast, maeWape_esios)
        
    # We store the MAE and WAPE for every folder and draw a gragh with this data.
    saveResults(maeWape, maeWape_esios, model, type)        
    dibujaGraph(train_split, folder_split, df2, iteration,
                realData, forecastedData, esiosForecast, model, type)



def CNN(all_data, df2, folder_split, cv, epochs, 
                   batch_size, train_split, type, forecast_horizon, past_history):
    """
  Allows to run the CNN_univariant experiment. 
  :param all_data: Dataframe object, contains all the data.
  :param df2: Dataframe object, contains only the train and test data.
  :param folder_split: int, that represents the size of a folder.
  :param cv: int, the number of folders that divides the data.
  :param epochs: int, the number of epochs to train our model.
  :param batch_size: int, the size of the batch during training.
  :param train_split: int, represents the point of the folder where 
  the test data start.
  :param type: String, represents type of model, U-> Univariate, M-> Multivariate .
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  """
    maeWape, maeWape_esios = [[], []], [[], []]
    realData, forecastedData, esiosForecast = [], [], []
    x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, 0, train_split)
    model = inicializaModelo_CNN(x_train, forecast_horizon)

    model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
                 validation_data=(x_test, y_test))
        
    preds = model.predict(x_test)
        
    comparePreds(norm_params, preds, y_test, all_data, 'CNN', 
                forecastedData, realData, 
                train_split, folder_split, esiosForecast, 0,
                maeWape, maeWape_esios, df2, type, past_history, forecast_horizon)

    for iteration in range(1, cv):
        realData, forecastedData, esiosForecast = [], [], []
        x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, iteration, train_split)
            
        model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
                 validation_data=(x_test, y_test))
        
        preds = model.predict(x_test)
        
        comparePreds(norm_params, preds, y_test, all_data, 'CNN', 
                     forecastedData, realData, 
                     train_split, folder_split, esiosForecast, iteration,
                     maeWape, maeWape_esios, df2, type, past_history, forecast_horizon)
    saveResultsAverage(maeWape, maeWape_esios, 'CNN', type)

def CNN_LSTM(all_data, df2, folder_split, cv, epochs, 
                   batch_size, train_split, type, forecast_horizon, past_history):
    """
  Allows to run the CNN_LSTM_univariant experiment. 
  :param all_data: Dataframe object, contains all the data.
  :param df2: Dataframe object, contains only the train and test data.
  :param folder_split: int, represents the size of a folder.
  :param cv: int, the number of folders that divides the data.
  :param epochs: int, the number of epochs to train our model.
  :param batch_size: int, the size of the batch during training.
  :param train_split: int, represents the point of the folder where 
  the test data start.
  :param type: String, represents type of model, U-> Univariate, M-> Multivariate 
  :param forecast_horizon: int, the number of steps in the future that the model will forecast.
  :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
  
  """
    x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, 0, train_split)
    model = inicializaModelo_CNN_LSTM(x_train, forecast_horizon)
    maeWape, maeWape_esios= [[], []], [[], []]

    realData, forecastedData, esiosForecast = [], [], []

    model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
                 validation_data=(x_test, y_test))
        
    preds = model.predict(x_test)
        
    comparePreds(norm_params, preds, y_test, all_data, 'CNN_LSTM', 
                forecastedData, realData, 
                train_split, folder_split, esiosForecast, 0,
                maeWape, maeWape_esios, df2, type, past_history, forecast_horizon)
    
    for iteration in range(1, cv):
        realData, forecastedData, esiosForecast = [], [], []
        x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, iteration, train_split)
            
        model.fit(x_train, y_train,
             batch_size=batch_size,
             epochs=epochs,
             verbose=1,
             validation_data=(x_test, y_test))
        
        preds = model.predict(x_test)

        comparePreds(norm_params, preds, y_test, all_data, 'CNN_LSTM', 
                     forecastedData, realData, 
                     train_split, folder_split, esiosForecast, iteration,
                     maeWape, maeWape_esios, df2, type, past_history, forecast_horizon)

    saveResultsAverage(maeWape, maeWape_esios, 'CNN_LSTM', type)
