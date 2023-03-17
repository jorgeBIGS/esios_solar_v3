def normalize(ts, norm_params):
  """
  Apply min-max normalization
  :param ts: time series
  :param norm_params: tuple with params mean, std, max, min
  :return: normalized time series
  """

  denominator = norm_params["max"] - norm_params["min"]

  if denominator == 0.0:
    denominator = 1e-10
        
  return (ts - norm_params['min']) / denominator

def normalize_zscore(ts, norm_params):
  std = norm_params["std"]
  if std == 0.0:
    std = 1e-10
  return (ts - norm_params["mean"]) / norm_params["std"]

def denormalize_zscore(ts, norm_params):
  return (ts * norm_params["std"]) + norm_params["mean"]
  
def denormalize(ts, norm_params):
  """
  Apply min-max normalization
  :param data: time series
  :param norm_params: tuple with params mean, std, max, min
  :return: normalized time series
  """
  return ts * (norm_params["max"] - norm_params["min"]) + norm_params["min"]