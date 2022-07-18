Para realizar las pruebas de las redes CNN y CNN_LSTM del artículo solo debe ejecutarse el archivo run.py, 
en el se pueden encontrar desde la línea 22 a la 27 algunos parámetros que pueden ser modificados:
  
  1. Las epochs, por defecto a 100.
  2. El batch_size, por defecto a 64.
  3. El forecast_horizon, por defecto a 1.
  4. El past_history, por defecto a 48.
  5. El shift, indica cual es la diferencia de pasos de tiempo entre el último registro que se usa 
  como past_hisotry al que se va a predecir, por defecto a 24.
  6. El cv, número de carpetas en las que se dividen los datosm, por defecto a 10.

Si se quieren realizar modificaciones sobre la arquitectura de las redes se debe ir a los métodos 
inicializaModelo_X que se encuentran en el archivo CNN.py. En estas funciones se puede observar la
arquitectura de las redes y se puede modificar todo lo referente a ellas.
