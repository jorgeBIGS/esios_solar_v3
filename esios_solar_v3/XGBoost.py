import xgboost as xgb
import numpy as np
from fileUtils import load_data_stacked, prepareStackedData, storeForecastValues, saveResultsAverage, saveResults, dibujaGraph
from Utils import getResults, getSeed

seed = getSeed()
def inicializa_XGBoost(n_estimators, lr):
    """
    Initializes XGBoost model with fixed hiperparameters selected by grid search.
    :param learning rate: float, step size shrinkage used in update to prevents overfitting. 
    :param max_depth: int, maximum depth of a tree.
    :param n_estimators: int, specifies the size of the forest to be trained.
    :param seed: int, seed to obtain replicable results
    :param verbosity: Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug). 
    """
    reg = xgb.XGBRegressor(learning_rate=lr, max_depth=4, n_estimators=n_estimators, seed = seed, verbosity=1)
    
    return reg

    
def XGBoost(type, cv, shift, all_data, folder_split, train_split, df2, forecast_horizon, past_history):
    """
    Runs XGBoost train and test for Multivariate or Univariate models.
    :param type: String, M -> Multivariate, U -> Univariate.
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

    if(shift==0):
        reg = inicializa_XGBoost(200, 0.07)
    if(shift==24):
        reg = inicializa_XGBoost(50, 0.07)
    if(shift==48):
        reg = inicializa_XGBoost(100, 0.03)

    # Loads stacked data to train the model. This is the best data format to train this model, a DataFrame object 
    # which every row contains all the training or test data for the model.
    if shift == 0: shift = 1
    lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked(type, shift)
    
    # For every folder we will train the model and obtain the MAE and WAPE accuracy metrics using the validate data.
    for iteration in range(0, cv):
        print('XGBoost Block', iteration)

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
            storeForecastValues(prediction, norm_params, y_test, all_data, count, forecastedData, realData, 
            train_split, folder_split, esiosForecast, iteration, type, past_history, forecast_horizon, "XGBOOST")

        #realData, forecastedData, esiosForecast = outlierDetector(realData, forecastedData, esiosForecast)
        
        # This method obtains the MAE and WAPE results from the data.
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # We write the real values, the forecasted values and the esios forecasted values in a file.
        saveResults(maeWape, maeWape_esios, 'XGBOOST', type, shift)
        dibujaGraph(train_split, folder_split, df2, iteration,
                    realData, forecastedData, esiosForecast, 'XGBOOST', type, past_history)
    
    # Finally, we store the average value of MAE and WAPE for all the folders.
    saveResultsAverage(maeWape, maeWape_esios, 'XGBOOST', type, shift)