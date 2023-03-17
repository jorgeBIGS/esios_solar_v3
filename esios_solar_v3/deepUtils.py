from normalization import denormalize
from Utils import getResults
from fileUtils import saveResults, saveValues, dibujaGraph

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
    
def comparePreds(norm_params, preds, y_test, all_data, model, forecastedData, realData, train_split, 
folder_split, esiosForecast, iteration, maeWape, maeWape_esios, df2, type, past_history, 
forecast_horizon, shift):
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
    :param shift: int, Specifies how many entries are between the last entry of 
    the past_history and the first of the forecast_horizon.
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
    saveResults(maeWape, maeWape_esios, model, type, shift)        
    dibujaGraph(train_split, folder_split, df2, iteration,
                realData, forecastedData, esiosForecast, model, type, past_history)