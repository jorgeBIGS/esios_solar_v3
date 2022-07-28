#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:15:33 2021

@author: tomas
"""

import xgboost as xgb
import numpy as np
from Utils import getSeed, denormalize, load_data_stacked, saveValues, prepareStackedData, outlierDetector, getResults, dibujaGraph, saveResults, saveResultsAverage
seed = getSeed()

def inicializa_XGBoost():
    """
    Initializes XGBoost model with fixed hiperparameters selected by grid search.
    :param learning rate: float, step size shrinkage used in update to prevents overfitting. 
    :param max_depth: int, maximum depth of a tree.
    :param n_estimators: int, specifies the size of the forest to be trained.
    :param seed: int, seed to obtain replicable results
    :param verbosity: Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). 
    """
    reg = xgb.XGBRegressor(learning_rate=0.07, max_depth=4, n_estimators=200, seed = seed, verbosity=1)
    
    return reg

def storeForecastValues(prediction, norm_params, y_test, all_data,
                count, forecastedData, realData, train_split, 
                folder_split, esiosForecast, iteration, type, 
                past_history, forecast_horizon):
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

    saveValues(realesfl, esiosfl, predictedfl, type, "")
    
def XGBoost(tipo, cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history):
    """
    Runs XGBoost train and test for Multivariate or Univariate models.
    :param tipo: String, M -> Multivariate, U -> Univariate.
    :param cv: int, number of folders.
    :param shift: int, specifies how many entries are between the last entry of the past_history and the first of the forecast_horizon.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param folder_split: int, it specifies how many entries are in each folder.
    :param train_split: int, it specifies how many entries are in each train set for every folder.
    :param df2: DataFrame object, with the preprocessed data.
    :param forecast_horizon: int, the number of steps in the future that the model will forecast.
    :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data
    """
    # First, inializes the model and the variables where accuracy metrics will be stored.
    maeWape, maeWape_esios = [[], []], [[], []]

    # Loads stacked data to train the model. This is the best data format to train this model, a DataFrame object 
    # which every row contains all the training or test data for the model.
    if shift == 0: shifted = 1
    if(tipo == "M"):
        lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked('M', shift)
    else:
        lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked('U', shift)

    reg = inicializa_XGBoost()
    
    # For every folder we will train the model and obtain the MAE and WAPE accuracy metrics using the validate data.
    for iteration in range(0, cv):
        print('XGBoost ITERATION', iteration)

        # Prepares the data to train and validates the model in every folder.
        x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, iteration)

        # We train the model with the training data and use the test data to validate the model.
        reg = reg.fit(x_train, y_train)
        preds = reg.predict(x_test)   
    
        realData, forecastedData, esiosForecast = [], [], []

        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast enrty and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):
            storeForecastValues(prediction, norm_params, y_test, all_data,
                    count, forecastedData, realData, 
                    train_split, folder_split, esiosForecast, iteration, tipo, past_history, forecast_horizon)

        realData, forecastedData, esiosForecast = outlierDetector(realData, forecastedData, esiosForecast)
        
        # This method obtains the MAE and WAPE results from the data.
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # We write the real values, the forecasted values and the esios forecasted values in a file.
        saveResults(maeWape, maeWape_esios, 'XGBOOST', tipo)
        dibujaGraph(train_split, folder_split, df2, iteration,
                    realData, forecastedData, esiosForecast, 'XGBOOST', tipo, past_history)
    
    # Finally, we store the average value of MAE and WAPE for all the folders.
    saveResultsAverage(maeWape, maeWape_esios, 'XGBOOST', tipo)