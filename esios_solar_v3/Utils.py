import numpy as np
import tensorflow as tf
import os
import random

def getSeed():
  """
  Initialize the seed to make experiments reproducible.
  returns seed.
  """
  seed = 6
  os.environ['PYTHONHASHSEED']=str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
  tf.compat.v1.keras.backend.set_session(sess)
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

  # tf.config.threading.set_inter_op_parallelism_threads(1)
  # tf.config.threading.set_intra_op_parallelism_threads(1)

  return seed

def media(lista):
  """
  Calculate the average of a list.
  :param lista: List used to do the average.

  return avg, average value for the list
  """
  lista = np.asarray(lista)
  avg = np.mean(lista)
  return avg

def evaluate_error(actual, predicted):
    """
  Evaluate the MAE and WAPE error for a folder
  :param actual: List with the real values.
  :param predicted: List with the predicted values.
  :return: List with the MAE and WAPE error for the folder.
  """
    metrics = []
    EPSILON = 1e-10
    mae = np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))
    metrics.append(mae)
    wape = mae / (np.mean(actual) + EPSILON)
    metrics.append(wape)
    
    return metrics

def getResults(realData, forecasts, maeWape):
  """
  Calculate MAE and WAPE error between the real values and the one you want to compare.
  :param realData: List with the real values for the folder.
  :param forecasts: List with the Esios' predictions or our model's predictions
  values for the folder.
  :param maeWape: List with the MAE and WAPE errors for the fodler.
  
  """
  results_metrics = evaluate_error(np.asarray(realData), np.asarray(forecasts))
  result_mae = results_metrics[0]
  result_wape = results_metrics[1]
  maeWape[0].append(result_mae)
  maeWape[1].append(result_wape)