from fileUtils import load_data_stacked, prepareStackedData, storeForecastValues, dibujaGraph, saveResults, saveResultsAverage
from outlierUtils import select_var
from Utils import getResults
import sklearn
import numpy as np

def initializeSVR(C, degree, epsilon, gamma):
    """
    Initializes SVR model with fixed hiperparameters selected by grid search.
    :param C: float, regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive.
    :param degree: int, degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    :param epsilon: float, it specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within,
    a distance epsilon from the actual value.
    :param gamma: {‘scale’, ‘auto’} or float, kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    :param kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, specifies the kernel type to be used in the algorithm.
    """
    return sklearn.svm.SVR(C = C, degree = degree, epsilon = epsilon, gamma = gamma, kernel = 'linear')

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
    realData, forecastedData, esiosForecast = [], [], []
    
    # First, we initialize the model and the variables where accuracy metrics will be stored.
    maeWape, maeWape_esios = [[], []], [[], []]
    if(shift == 0):
        clf = initializeSVR(100, 3, 0.0001, 'auto')
    if(shift == 24):
        clf = initializeSVR(0.1, 3, 0.0001, 0.0001)
    if(shift == 48):
        clf = initializeSVR(0.1, 3, 0.0001, 0.0001)

    if shift == 0: shift = 1
    lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked('U', shift)

    # Prepares the data to train and validates the model in every folder.
    x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, 0)            
    
    # We train the model with the training data and use the test data to validate the model.
    clf = clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
        
    # For every prediction, we denormalize the values and compare the real entry with the 
    # forecast entry and the esios predictions for that instant. Then we store the results 
    # of MAE and WAPE for every folder
    for count, prediction in enumerate(preds):
        storeForecastValues(prediction, norm_params, y_test, all_data,
                count, forecastedData, realData, 
                train_split, folder_split, esiosForecast, 0, "U", past_history, forecast_horizon, "SVR")
            
    # This method obtains the MAE and WAPE results from the data.
    getResults(realData, forecastedData, maeWape)
    getResults(realData, esiosForecast, maeWape_esios)

    # We write the real values, the forecasted values and the esios forecasted values in a file.
    saveResults(maeWape, maeWape_esios, 'SVR', "U", shift)
    dibujaGraph(train_split, folder_split, df2, 0,
                    realData, forecastedData, esiosForecast, 'SVR', "U", past_history)
    
    # For every folder we will train the model and obtain the MAE and WAPE accuracy metrics using the validate data.
    for iteration in range(1, cv):
        print("SVR Block", iteration)
        realData, forecastedData, esiosForecast = [], [], []

        # Prepares the data to train and validates the model in every folder.
        x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, iteration)
        
        # We train the model with the training data and use the test data to validate the model.
        clf = clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        
        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast entry and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):
            storeForecastValues(prediction, norm_params, y_test, all_data,
                count, forecastedData, realData, 
                train_split, folder_split, esiosForecast, iteration, "U", past_history, forecast_horizon, "SVR")
            
        # This method obtains the MAE and WAPE results from the data.
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # We write the real values, the forecasted values and the esios forecasted values in a file.
        saveResults(maeWape, maeWape_esios, 'SVR', "U", shift)
        dibujaGraph(train_split, folder_split, df2, iteration,
                        realData, forecastedData, esiosForecast, 'SVR', "U", past_history)

    # Finally, we store the average value of MAE and WAPE for all the folders.
    saveResultsAverage(maeWape, maeWape_esios, 'SVR', 'U', shift)
    
def SVR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
             past_history, shift, type):
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

    if(shift == 0):
        clf = initializeSVR(100, 3, 0.0001, 'auto')
    if(shift == 24):
        clf = initializeSVR(0.1, 3, 0.0001, 0.0001)
    if(shift == 48):
        clf = initializeSVR(0.1, 3, 0.0001, 0.0001)

    realData, forecastedData, esiosForecast = [], [], []

    if shift == 0: shift = 1
    lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked(type, shift)
    
    # Prepares the data to train and validates the model in every folder.
    x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, 0)

    # We use a selector to reduce the input data for this model. This is important because 
    # clasical regression models such as SVR cant handle large amounts of data.
    sel_ = select_var(x_train, y_train)
    X_train_selected = sel_.transform(x_train)
    X_test_selected = sel_.transform(x_test)
        
    # We train the model with the training data and use the test data to validate the model.
    clf = clf.fit(X_train_selected, y_train)
    preds = clf.predict(X_test_selected)

    # For every prediction, we denormalize the values and compare the real entry with the 
    # forecast enrty and the esios predictions for that instant. Then we store the results 
    # of MAE and WAPE for every folder
    for count, prediction in enumerate(preds):
        storeForecastValues(prediction, norm_params, y_test, all_data, count, forecastedData, realData, 
                train_split, folder_split, esiosForecast, 0, type, past_history, forecast_horizon, "SVR")
            
    # This method obtains the MAE and WAPE results from the data.
    getResults(realData, forecastedData, maeWape)
    getResults(realData, esiosForecast, maeWape_esios)

    # We write the real values, the forecasted values and the esios forecasted values in a file.
    saveResults(maeWape, maeWape_esios, 'SVR', type, shift)
    dibujaGraph(train_split, folder_split, df2, 0,
                    realData, forecastedData, esiosForecast, 'SVR', type, past_history)

    
    for iteration in range(1, cv):
        print("SVR Block", iteration)
        realData, forecastedData, esiosForecast = [], [], []
        # Prepares the data to train and validates the model in every folder.
        x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, iteration)
             
        X_train_selected = sel_.transform(x_train)
        X_test_selected = sel_.transform(x_test)
        
        # We train the model with the training data and use the test data to validate the model.
        clf = clf.fit(X_train_selected, y_train)
        preds = clf.predict(X_test_selected)

        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast enrty and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):
            storeForecastValues(prediction, norm_params, y_test, all_data,
                    count, forecastedData, realData, 
                    train_split, folder_split, esiosForecast, 0, type, past_history, forecast_horizon, "SVR")
            
        # This method obtains the MAE and WAPE results from the data.
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # We write the real values, the forecasted values and the esios forecasted values in a file.
        saveResults(maeWape, maeWape_esios, 'SVR', type, shift)
        dibujaGraph(train_split, folder_split, df2, iteration,
                        realData, forecastedData, esiosForecast, 'SVR', type, past_history)
            
    saveResultsAverage(maeWape, maeWape_esios, 'SVR', type, shift)