from run import runExperimentForecastedIrradiation, runExperimentUM

shift = 48
runExperimentUM(epochs=50, batch_size=128, shift=shift)
runExperimentForecastedIrradiation(epochs=50, batch_size=128, shift=shift, type='F')
runExperimentForecastedIrradiation(epochs=50, batch_size=128, shift=shift, type='FM')