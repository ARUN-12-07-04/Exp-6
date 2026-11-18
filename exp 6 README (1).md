# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate stationary time series data
np.random.seed(42)

data = [0]

for i in range(1, 100):
    value = 0.6 * data[i-1] + np.random.normal(0, 1)
    data.append(value)

# Convert into NumPy array
data = np.array(data)

# Split training and testing data
train = data[:80]
test = data[80:]

# Plot ACF and PACF
plot_acf(train, lags=20)
plt.title("Autocorrelation Function (ACF)")
plt.show()

plot_pacf(train, lags=20)
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

# Fit ARIMA model
model = ARIMA(train, order=(1, 0, 1))
model_fit = model.fit()

# Forecast future values
forecast = model_fit.forecast(steps=len(test))

# Evaluate model
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("ARIMA Model Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot actual and predicted values
plt.figure(figsize=(10, 5))

plt.plot(range(len(train)), train, label="Training Data")
plt.plot(range(80, 100), test, label="Actual Values")
plt.plot(range(80, 100), forecast, label="Forecast Values")

plt.title("ARIMA Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.show()

## Output

 ARIMA Model Evaluation
MAE: 0.6687258909497752
MSE: 0.7349968854849636
RMSE: 0.857319593550132


## Result
Thus, the ARIMA time-series forecasting model was successfully implemented using Python. A stationary time series was prepared, ARIMA parameters were selected, and future values were forecasted. Model performance was evaluated using MAE, MSE and RMSE.
