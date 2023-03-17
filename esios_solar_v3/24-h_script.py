from run import runExperimentForecastedIrradiation, runExperimentUM

shift=24
runExperimentUM(epochs=25, batch_size=64, shift=shift)
runExperimentForecastedIrradiation(epochs=100, batch_size=64, shift=shift, type='F')
runExperimentForecastedIrradiation(epochs=100, batch_size=64, shift=shift, type='FM')