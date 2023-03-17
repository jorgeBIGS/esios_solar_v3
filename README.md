[![TensorFlow 2.9.1](https://img.shields.io/badge/TensorFlow-2.9.1-orange?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.9.1)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)](https://www.python.org/downloads/release/python-380/)

Para realizar las pruebas que aparecen en el artículo solo debe ejecutarse los scripts llamados 1-h_script, 24-h_script y 48-h_script, 
en los que se encuentran algunos parámetros que pueden ser modificados:
  
  1. Epochs, por defecto las selecciondas para cada horizonte de predicción.
  2. El batch_size por, defecto las selecciondas para cada horizonte de predicción.
  3. El forecast_horizon, por defecto a 1.
  4. El past_history, por defecto a 48.
  5. El shift, indica cual es la diferencia de pasos de tiempo entre el último registro que se usa 
  como past_hisotry al que se va a predecir, por defecto a 24.
  6. El cv, número de carpetas en las que se dividen los datosm, por defecto a 10.

Si se quieren realizar modificaciones sobre la arquitectura de las redes se debe ir a los métodos 
inicializaModelo_X que se encuentran en el archivo CNN.py. En estas funciones se puede observar la
arquitectura de las redes y se puede modificar todo lo referente a ellas.

Los módulos de este proyecto son:
  -CNN.py: Cuenta con todas las funciones que inicializan los modelos de redes neuronales convolucionales y realizan las pruebas de validación.
  -LR.py: Cuenta con todas las funciones que inicializan los modelos de LinearRegression y realizan las pruebas de validación.
  -SVR.py: Cuenta con todas las funciones que inicializan los modelos de SVR y realizan las pruebas de validación.
  -XGBoost: Cuenta con todas las funciones que inicializan los modelos de XGBoost y realizan las pruebas de validación.
  -Utils: Contiene las funciones que se usan en los otros módulos para registrar los resultados, organizar la información, pintar las gráficas, normalizar
          los datos, seleccionar las mejores características para modelos multivariantes y evaluar el error.
  -fileUtils: Contiene las funciones que se usan en los otros módulos para registrar los resultados, organizar la información, pintar las gráficas y                     modificar los archivos csv con los resultados.
  -normalization.py: Contiene las funciones que se usan en los otros módulos para normalizar los datos.
  -outlierUtils.py: Contiene las funciones que se usan en los otros módulos para tratar los outliers. 
