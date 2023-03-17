from sklearn.linear_model import LinearRegression
from fileUtils import prepareStackedData, prepareTrain, saveResults, saveResultsAverage,saveValues, load_data_stacked, dibujaGraph
from outlierUtils import select_var, outlierDetector
from Utils import getResults
from normalization import denormalize
import numpy as np

def LR_Univariant(all_data, df2, folder_split, cv, train_split, shift, forecast_horizon,
             past_history):  
    """
    Initializes and runs a LR univariate model to forecast solar energy production throught esios data.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param df2: DataFrame object, with the preprocessed data.
    :param folder_split: int, it specifies how many entries are in each folder.
    :param cv: int, number of folders.
    :param train_split: int, it specifies how many entries are in each train set for every folder.
    :param shift: int, specifies how many entries are between the last entry of the past_history and the first of the forecast_horizon.
    :param forecast_horizon: int, the number of steps in the future that the model will forecast.
    :param past_history: int, the number of steps in the past that the model use as the size of the sliding window to create the train data.
    """
    # First, we initialize the model and the variables where accuracy metrics will be stored.
    maeWape, maeWape_esios = [[], []], [[], []]
    if(shift == 0 or shift == 48):
        regressor = LinearRegression(copy_X=(True), fit_intercept=(True), normalize=True)
    if(shift == 24):
        regressor = LinearRegression(copy_X=(True), fit_intercept=(False), normalize=True)

    # For every folder we will train the model and obtain the MAE and WAPE accuracy metrics using the validate data.
    for iteration in range(0, cv):
        print('LR Block', iteration)
        
        x_train, y_train, x_test, y_test, norm_params = prepareTrain(
            folder_split, df2, iteration, train_split, shift, past_history, forecast_horizon)
        
        x_train = np.reshape(x_train, (x_train.shape[0], past_history))
        x_test = np.reshape(x_test, (x_test.shape[0], past_history))

        # We train the model with the training data and use the test data to validate the model.
        regressor.fit(x_train, y_train)
        preds = regressor.predict(x_test)
        realData, forecastedData, esiosForecast = [], [], []

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

            # We write the real values, the forecasted values and the esios forecasted values in a file.         
            saveValues(realesf, esiosf.values, predictedf, 'LR', 'U')
            
            realData.append(realesf)
            forecastedData.append(predictedf)
            esiosForecast.append(esiosf)
        
        # This method obtains the MAE and WAPE results from the data.
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)
        
        # We store the MAE and WAPE for every folder and draw a gragh with this data.
        saveResults(maeWape, maeWape_esios, 'LR', 'U', shift)
        dibujaGraph(train_split, folder_split, all_data, iteration,
                    realData, forecastedData, esiosForecast, 'LR', 'U', past_history)
                    
    # Finally, we store the average value of MAE and WAPE for all the folders.        
    saveResultsAverage(maeWape, maeWape_esios, 'LR', 'U', shift)
    
def LR_M(all_data, df2, folder_split, cv, train_split, forecast_horizon,
             past_history, shift, type):
    """
    Initializes and runs a LR univariate model to forecast solar energy production throught esios data.
    :param all_data: DataFrame object, returned from pandas read_csv function.
    :param df2: DataFrame object, with the preprocessed data.
    :param cv: int, number of folders.
    :param shift: int, specifies how many entries are between the last entry of the past_history and the first of the forecast_horizon.
    :param type: str, specifies if we are using Multivariate data with (F) or without forecasted irradiation data (M).
    """
    realData, forecastedData, esiosForecast = [], [], []
    
    # First, we initialize the model and the variables where accuracy metrics will be stored.
    maeWape, maeWape_esios = [[], []], [[], []]

    if(shift == 0 or shift == 48):
        regressor = LinearRegression(copy_X=(True), fit_intercept=(True), normalize=True)
    if(shift == 24):
        regressor = LinearRegression(copy_X=(True), fit_intercept=(False), normalize=True)

    # Loads stacked data to train the model. This is the best data format to train this model, a DataFrame object 
    # which every row contains all the training or test data for the model.
    if shift == 0: shift = 1
    lista_train, lista_train_y, lista_test, lista_test_y = load_data_stacked(type, shift)
    # Prepares the data to train and validates the model in every folder.
    x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, 0)
    # Selects only the best parameters to train and validates the model.
    sel_ = select_var(x_train, y_train)
    
    for iteration in range(0, cv):
        print('LR Block', iteration)
        # Prepares the data to train and validates the model in every folder.
        x_train, y_train, x_test, y_test, norm_params = prepareStackedData(lista_train, lista_train_y, lista_test, lista_test_y, iteration)
        
        X_train_selected = sel_.transform(x_train)
        X_test_selected = sel_.transform(x_test)

        # Trains the model and forecasts the values for the test data.
        regressor.fit(X_train_selected, y_train)
        preds = regressor.predict(X_test_selected)
        realData, forecastedData, esiosForecast = [], [], []
        
        # Validates the model using real data, forecasted test data and esios forecast data.
        # For every prediction, we denormalize the values and compare the real entry with the 
        # forecast entry and the esios predictions for that instant. Then we store the results 
        # of MAE and WAPE for every folder
        for count, prediction in enumerate(preds):   
            predictedf = denormalize(prediction, norm_params)
            realesf = y_test[count]
            predictedPoint = all_data[train_split + count + folder_split*iteration +
                                  past_history:train_split + folder_split*iteration +
                                  count + past_history + forecast_horizon]
            esiosf = predictedPoint.iloc[0]['Generación prevista Solar']
            
            realesfl = []
            esiosfl = []
            predictedfl = []
            
            realesfl.append(realesf)
            esiosfl.append(esiosf)
            predictedfl.append(predictedf)
            
            # Writes the real values, the forecasted values and the esios forecasted values in a file.
            saveValues(realesfl, esiosfl, predictedfl, 'LR', type)
            
            realData.append(realesf)
            forecastedData.append(predictedf)
            esiosForecast.append(esiosf)
        
        # Detects outlier and removes them.
        realData, forecastedData, esiosForecast = outlierDetector(realData, forecastedData, esiosForecast)
            
        getResults(realData, forecastedData, maeWape)
        getResults(realData, esiosForecast, maeWape_esios)

        # Stores the MAE and WAPE for every folder and draw a gragh with this data.
        saveResults(maeWape, maeWape_esios, 'LR', type, shift)
        dibujaGraph(train_split, folder_split, df2, iteration,
                    realData, forecastedData, esiosForecast, 'LR', type, past_history)
                    
    # Finally, we store the average value of MAE and WAPE for all the folders.        
    saveResultsAverage(maeWape, maeWape_esios, 'LR', type, shift)