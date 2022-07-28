#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:51:51 2021

@author: tomas
"""

from Utils import getSeed
import sklearn
import numpy as np
from Utils import select_var, denormalize, saveResults, saveValues, outlierDetector, getResults, dibujaGraph, saveResultsAverage, prepareTrain

def initializeSVR():
    """
  Initializes SVR model with fixed hiperparameters selected by grid search.
  :param C: float, regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
  :param degree: int, degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
  :param epsilon: float, it specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within,
   a distance epsilon from the actual value.
  :param gamma: {‘scale’, ‘auto’} or float, kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
  :param kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, specifies the kernel type to be used in the algorithm.
  """
    clf = sklearn.svm.SVR(C = 100.0, degree = 3, epsilon = 0.0001, gamma='auto', kernel = 'linear')
    return clf

def SVR_U(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon,
             past_history):
    """
    Runs SVR univariate model to forecast solar energy production throught esios data.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param df2: DataFrame object, with the preprocessed data.
    :param folder_split: int, it specifies how many entries are in each folder.
    :param cv: int, number of folders.
    :param train_split: int, it specifies how many entries are in each train set for every folder.
    :param shift: int, specifies how many entries are between the last entry of the past_history and the first of the forecast_horizon.
    :param forecast_horizon: The number of steps in the future that the model will forecast.
    :param past_history: The number of steps in the past that the model use as the size of the sliding window to create the train data.
    """
    # First, we initialize the model and the variables where accuracy metrics will be stored.
    maeWape, maeWape_esios = [[], []], [[], []]
    clf = initializeSVR()
    
    # For every folder we will train the model and obtain the MAE and WAPE accuracy metrics using the validate data.
    for iteration in range(0, cv):
        print("SVR ITERATION", iteration)
        realData, forecastedData, esiosForecast = [], [], []

        # We prepare the data to train the model.
        x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, iteration, train_split, shift, past_history, forecast_horizon)

        # Reshape the data for the model.
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
        
        # We use a selector to reduce the input data for this model. This is important because 
        # clasical regression models such as SVR cant handle large amounts of data.
        sel_ = select_var(x_train, y_train, iteration)        
        X_train_selected = sel_.transform(x_train)
        X_test_selected = sel_.transform(x_test)
        
        # We train the model with the training data and use the test data to validate the model.
        clf = clf.fit(X_train_selected, y_train)
        preds = clf.predict(X_test_selected)
        
        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast entry and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):
            predictedf = denormalize(prediction, norm_params)
            realesf = denormalize(y_test[count], norm_params)
            predictedPoint = all_data[train_split + count + folder_split*iteration +
                                  past_history + shift:train_split + folder_split*iteration +
                                  count + past_history + forecast_horizon + shift]
            esiosf = predictedPoint['Generación prevista Solar']
            
            realesfl,esiosfl,predictedfl = [], [], []
            realesfl.append(realesf)
            esiosfl.append(esiosf)
            predictedfl.append(predictedf)

            # We write the real values, the forecasted values and the esios forecasted values in a file.
            saveValues(realesfl, esiosfl, predictedfl, 'SVR', 'U')
            
            realData.append(realesf)
            forecastedData.append(predictedf)
            esiosForecast.append(esiosf)

        # Detects outlier and removes them.
        realData, forecastedData, esiosForecast = outlierDetector(realData, forecastedData, esiosForecast)

        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)
        
        # We store the MAE and WAPE for every folder and draw a gragh with this data.
        saveResults(maeWape, maeWape_esios, 'SVR', 'U')
        dibujaGraph(train_split, folder_split, all_data, iteration,
                    realData, forecastedData, esiosForecast, 'SVR', 'U', past_history)

    # Finally, we store the average value of MAE and WAPE for all the folders.
    saveResultsAverage(maeWape, maeWape_esios, 'SVR', 'U')
    
def SVR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
             past_history, shift):
    """
    Runs SVR multivariate model to forecast solar energy production throught esios data.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param folder_split: int, it specifies how many entries are in each folder.
    :param cv: int, number of folders.
    :param train_split: int, it specifies how many entries are in each train set for every folder.
    :param forecast_horizon: The number of steps in the future that the model will forecast.
    :param past_history: The number of steps in the past that the model use as the size of the sliding window to create the train data.
    """
    maeWape, maeWape_esios = [[], []], [[], []]
    clf = initializeSVR()
    
    for iteration in range(0, cv):
        print("SVR ITERATION", iteration)
        realData, forecastedData, esiosForecast = [], [], []
        x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, iteration, train_split, shift, past_history, forecast_horizon)
    
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
        
        # We use a selector to reduce the input data for this model. This is important because 
        # clasical regression models such as SVR cant handle large amounts of data.
        sel_ = select_var(x_train, y_train, iteration)        
        X_train_selected = sel_.transform(x_train)
        X_test_selected = sel_.transform(x_test)
        print("selected:", x_train.shape, X_train_selected.shape, x_test.shape, X_test_selected.shape)
        # We train the model with the training data and use the test data to validate the model.
        clf = clf.fit(X_train_selected, y_train)
        preds = clf.predict(X_test_selected)

        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast enrty and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):
            predictedf = denormalize(prediction, norm_params)
            realesf = denormalize(y_test[count] , norm_params)
            predictedPoint = all_data[train_split + count + folder_split*iteration +
                                  past_history:train_split + folder_split*iteration +
                                  count + past_history + forecast_horizon]
            esiosf = predictedPoint['Generación prevista Solar']
            
            realesfl,esiosfl,predictedfl = [], [], []
            realesfl.append(realesf)
            esiosfl.append(esiosf)
            predictedfl.append(predictedf)
        
            # We write the real values, the forecasted values and the esios forecasted values in a file 
            # and store the data to calculate the average error for the model in the future.
            saveValues(realesfl, esiosfl, predictedfl, 'SVR', "M")
            
            realData.append(realesf)
            forecastedData.append(predictedf)
            esiosForecast.append(esiosf)

        realData, forecastedData, esiosForecast = outlierDetector(realData, forecastedData, esiosForecast)
        
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # We store the MAE and WAPE for every folder and draw a gragh with this data.
        saveResults(maeWape, maeWape_esios, 'SVR', 'M')
        dibujaGraph(train_split, folder_split, all_data, iteration,
                    realData, forecastedData, esiosForecast, 'SVR', 'M', past_history)
            
    saveResultsAverage(maeWape, maeWape_esios, 'SVR', 'M')