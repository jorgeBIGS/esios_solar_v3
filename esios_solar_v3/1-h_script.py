from run import runExperimentForecastedIrradiation, runExperimentUM

runExperimentUM(epochs=100, batch_size=64, shift=0)
runExperimentForecastedIrradiation(epochs=100, batch_size=64, shift=0, type='F')