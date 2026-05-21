import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pdb
import holidays

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error


raw_fc_data = pd.read_csv("electricity_price_forecast_assignment-quant_assignment-data/data/data_forecast.csv")
raw_act_data = pd.read_csv("electricity_price_forecast_assignment-quant_assignment-data/data/data_actual.csv")

dk_holidays = holidays.Denmark(years=range(2020, 2022))

raw_fc_data["valuetime"] = pd.to_datetime(raw_fc_data["valuetime"])
raw_fc_data["import"] =raw_fc_data["flow"].clip(lower=0)
raw_fc_data["export"] = (-raw_fc_data["flow"]).clip(lower=0)

raw_fc_data["is_weekend"] = (raw_fc_data["valuetime"].dt.dayofweek >= 5).astype(int)
raw_fc_data["is_holiday"] = raw_fc_data["valuetime"].isin(dk_holidays).astype(int)
raw_fc_data["hour_sin"] = np.sin(2 * np.pi * raw_fc_data["valuetime"].dt.hour / 24)
raw_fc_data["hour_cos"] = np.cos(2 * np.pi * raw_fc_data["valuetime"].dt.hour / 24)
raw_fc_data["dayofweek_sin"] = np.sin(2 * np.pi * raw_fc_data["valuetime"].dt.dayofweek / 7)
raw_fc_data["dayofweek_cos"] = np.cos(2 * np.pi * raw_fc_data["valuetime"].dt.dayofweek / 7)


raw_act_data["valuetime"] = pd.to_datetime(raw_act_data["valuetime"])
raw_act_data["import"] =raw_act_data["flow"].clip(lower=0)
raw_act_data["export"] = (-raw_act_data["flow"]).clip(lower=0)
raw_act_data["wind_fc_error"] = (1 - (raw_fc_data["wind"]/raw_act_data["wind"])).fillna(0)
raw_act_data["solar_fc_error"] = (1 - (raw_fc_data["solar"]/raw_act_data["solar"])).fillna(0)
raw_act_data["load_fc_error"] = (1 - (raw_fc_data["load"]/raw_act_data["load"])).fillna(0)

raw_act_data["is_weekend"] = (raw_act_data["valuetime"].dt.dayofweek >= 5).astype(int) 
raw_act_data["is_holiday"] = raw_act_data["valuetime"].isin(dk_holidays).astype(int)
raw_act_data["hour_sin"] = np.sin(2 * np.pi * raw_act_data["valuetime"].dt.hour / 24)
raw_act_data["hour_cos"] = np.cos(2 * np.pi * raw_act_data["valuetime"].dt.hour / 24)
raw_act_data["dayofweek_sin"] = np.sin(2 * np.pi * raw_act_data["valuetime"].dt.dayofweek / 7)
raw_act_data["dayofweek_cos"] = np.cos(2 * np.pi * raw_act_data["valuetime"].dt.dayofweek / 7)

'''
Training data 60 days
Testing data 60-90 days
Validating data 90-12 days
'''

train_data = raw_act_data[0:60*24]
test_data_act = raw_act_data[60*24:90*24]
test_data_fc = raw_fc_data[60*24:90*24]




# train = train_data['spot']
# kpi_list = []
# orders = [(p, d, q) for p in [0,1,2] for d in [0,1,2] for q in [0,1,2]]
# seasonal_orders = [(p, d, q, 24) for p in [0,1] for d in [0,1] for q in [0,1]]
# for order in orders:
#     for seasonal_order in seasonal_orders:
#         try:
#             with warnings.catch_warnings():
#                 model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
#                 fitted = model.fit()

#                 # forecast = fitted.get_forecast(steps=24, alpha=0.05)  # 95% CI
#                 # fc_mean = forecast.predicted_mean
#                 # fc_ci = forecast.conf_int()
#                 # # residuals = test_data[:len(fc_mean)]["spot"] - fc_mean
#                 # # forecast_df = pd.DataFrame({'actual':test_data[:len(fc_mean)]["spot"], 'forecast': fc_mean.values, 'lower_95': fc_ci['lower spot'].values, 'upper_95': fc_ci['upper spot'].values}, index=test_data.index[:24])

#                 # mae  = mean_absolute_error(test_data[:24]["spot"], fc_mean)
#                 # rmse = np.sqrt(mean_squared_error(test_data[:24]["spot"], fc_mean))
#                 # mape = np.mean(np.abs((test_data[:24]["spot"] - fc_mean) / test_data[:24]["spot"])) * 100

#                 row_list = {"order" : order, "seasonal_order": seasonal_order, "aic" : fitted.aic, "bic" : fitted.bic}
#                 print(row_list)
#                 kpi_list.append(row_list)
#         except Exception as e:
#             print(f"FAILED — order={order}, seasonal_order={seasonal_order} | Error: {e}")

# kpis = pd.DataFrame.from_dict(kpi_list)

#* Rolling origin method SARIMA
# all_forecasts = []
# start_idx = 0

# while start_idx + h <= len(test):
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
#             fitted = model.fit(disp=False)

#         fc = fitted.get_forecast(steps=h)
#         fc_mean = fc.predicted_mean
#         fc_ci = fc.conf_int(alpha=0.05)

#         actual = test.iloc[start_idx:start_idx + h]

#         forecast_chunk = pd.DataFrame({
#             "actual": actual.values,
#             "forecast": fc_mean.values,
#             "lower_95": fc_ci.iloc[:, 0].values,
#             "upper_95": fc_ci.iloc[:, 1].values
#                                       }, index=actual.index)

#         all_forecasts.append(forecast_chunk)

#         train = pd.concat([train, actual])
#         start_idx += h

#     except Exception as e:
#         print(f"FAILED at start_idx={start_idx} | order={order}, seasonal_order={seasonal_order} | Error: {e}")
#         start_idx += h
    
#     print(f"Days Completed: {start_idx/24}")
    

# forecast_df = pd.concat(all_forecasts)

# mae  = mean_absolute_error(forecast_df["actual"], forecast_df["forecast"])
# rmse = np.sqrt(mean_squared_error(forecast_df["actual"], forecast_df["forecast"]))
# mape = ((forecast_df["actual"] - forecast_df["forecast"]) / forecast_df["actual"]).abs().mean() * 100

'''
#* SARIMAX 
order = (1, 1, 2)
seasonal_order = (0, 1, 1, 24)
feature_cols =["wind", "solar", "load", "import", "export", "is_weekend", "is_holiday", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]

h = 24
train = train_data.copy()
test = test_data_fc.copy()


model = SARIMAX(
    endog=train['spot'],
    exog=train[feature_cols],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted = model.fit(disp=True, method_kwargs={'maxiter': 500})
fc = fitted.get_forecast(steps=h, exog=test[feature_cols].iloc[:h])
fc_mean = fc.predicted_mean

forecasted_prices = pd.DataFrame({"actual": test_data_act["spot"].iloc[:h].values, "forecast": fc_mean.values})
forecast_df = forecasted_prices.copy()
'''

#* Rolling origin SARIMAX

order = (1, 1, 2)
seasonal_order = (0, 1, 1, 24)
feature_cols = ["wind", "solar", "load", "import", "export", "is_weekend", "is_holiday", "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]

h = 24

# Initialize
train = train_data.copy()  # Initial training set
test = test_data_fc.copy()
actual = test_data_act.copy()

all_forecasts = []
all_actual_prices = []
all_conf_ints = []

i = 0  # Position in test set

while i < len(test):
    print(f"\n=== Rolling Forecast Iteration {i//h + 1} ===")
    print(f"Training set size: {len(train)} observations")
    print(f"Forecasting hours: {i} to {min(i+h, len(test))}")
    
    # Fit model on current training set
    model = SARIMAX(
        endog=train['spot'],
        exog=train[feature_cols],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    try:
        fitted = model.fit(disp=False, method_kwargs={'maxiter': 500})
    except:
        fitted = model.fit(disp=False, method='lbfgs')
    
    # Forecast next h steps
    steps = min(h, len(test) - i)
    exog_future = test[feature_cols].iloc[i:i+steps]
    
    fc = fitted.get_forecast(steps=steps, exog=exog_future, alpha=0.05)
    fc_mean = fc.predicted_mean
    conf_int = fc.conf_int()
    
    # Store results
    all_forecasts.append(fc_mean)
    all_conf_ints.append(conf_int)
    
    # Get actual values
    actual_slice = actual["spot"].iloc[i:i+steps]
    all_actual_prices.append(actual_slice)
    
    # === ROLLING: Add observed data to training set ===
    # Get the actual observations for this period (if available)
    observed_data = actual.iloc[i:i+steps].copy()
    
    # Append to training set for next iteration
    train = pd.concat([train, observed_data], axis=0)
    
    i += h

pdb.set_trace()

# Combine results
forecasted_prices = pd.DataFrame({ "actual": pd.concat(all_actual_prices, ignore_index=True), "forecast": pd.concat(all_forecasts, ignore_index=True)})

all_conf_int_df = pd.concat(all_conf_ints, ignore_index=True)
forecasted_prices["lower_95"] = all_conf_int_df.iloc[:, 0].values
forecasted_prices["upper_95"] = all_conf_int_df.iloc[:, 1].values

final_mae = mean_absolute_error(forecasted_prices["actual"], forecasted_prices["forecast"])
final_rmse = np.sqrt(mean_squared_error(forecasted_prices["actual"], forecasted_prices["forecast"]))
final_mape = ((forecasted_prices["actual"] - forecasted_prices["forecast"]) / forecasted_prices["actual"]).abs().mean() * 100

print(f"\n=== Overall Performance ===")
print(f"Final MAE: {final_mae:.2f}")
print(f"Final RMSE: {final_rmse:.2f}")


pdb.set_trace()