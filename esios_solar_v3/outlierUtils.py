import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model

def select_var(x_train, y_train):
  sel_ = SelectFromModel(linear_model.Lars())
  sel_.fit(x_train, y_train)
  sel_.get_support()

  return sel_

def outlierDetector(realData, forecastedData, esiosForecast):
  """
  We use IQR method to detect and remove outliers from the data. First we 
  detect the outliers, we divide them in high value outliers and low 
  value outliers. Then we flip both sets to remove them from the last to 
  the first, avoiding order problems.
  :param forecastedData: predictions obtained from our model.
  :param realData: real data to validate the forecastData and the esiosForecast.
  :param esiosForecast: forecasting entries from esios.
   
  It returns the new results data without outliers. 
  """
  Q1 = np.percentile(forecastedData, 25,
                 interpolation = 'midpoint')
  Q3 = np.percentile(forecastedData, 75,
                 interpolation = 'midpoint')
  IQR = Q3 - Q1

  # Upper bound
  upper = np.where(forecastedData >= (Q3+1.5*IQR))
  if(len(upper[0])>0):
    for num in np.flip(upper[0]):
      forecastedData.pop(num)
      realData.pop(num)
      esiosForecast.pop(num)
  # Lower bound
  lower = np.where(forecastedData <= (Q1-1.5*IQR))
  if(len(lower[0])>0):
    for num in np.flip(lower[0]):
      forecastedData.pop(num)
      realData.pop(num)
      esiosForecast.pop(num)
  return realData, forecastedData, esiosForecast