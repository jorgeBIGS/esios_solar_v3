[![TensorFlow 2.9.1](https://img.shields.io/badge/TensorFlow-2.9.1-orange?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.9.1)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)

In our repository we can find a folder called esios_solar_v3, containing the results and source code files that have been used to build the tests in the article, and two Dockerfile and requierements files to generate a Docker environment capable of executing these files and obtaining the results that appear in the article. Inside the esios_solar_v3 folder we find 4 folders with the test results shown in the article. In addition, there are 13 Python scripts that contain all the code of the project. These scripts can be classified into 3 types. On the one hand, the scripts designed to facilitate the execution of the experiments seen in the article in a quick and simple manner, such as:
- run.py: Contains the calls to the models used in the paper already configured with the parameters that appear in the article.
- 1-h_script.py: Calls the run file to run the tests corresponding to the 1-h prediction horizon.
- 24-h_script.py: Calls the run file to run the tests corresponding to the 24-h prediction horizon.
- 48-h_script.py: Calls the run file to run the tests corresponding to the 48-h prediction horizon. These tests have been separated so that they can be run independently.

On the other hand, we have the scripts that implement the models used in the article, each one is named after the model implemented in that file:
- CNN.py: In this script we implement the CNN and CNN_LSTM models.
- LR.py.
- SVR.py.
- XGBoost.py.

Finally, we have the classes that handle the auxiliary functions used by the models and their validation processes:
- Utils.py: Contains the auxiliary functions that allow to calculate the errors of the models and which establish a seed for the reproducibility of the experiments.
- deepUtils.py: Contains the auxiliary functions for the calculation of the error and the validation process of the Deep learning models.
- fileUtils.py: Contains the auxiliary functions for storing the results of the experiments.
- normalization.py: Contains the auxiliary functions for the normalization of the data.
- outlierUtils.py: Contains the auxiliary functions for the treatment of outliers in the models.
 
