'''
Experiments of all forecasting models in one place. Easier for performance comparison. 
'''
#%%
import pandas as pd
import numpy as np
from experiment import Experiment
from utils import plot 

#%% Configure settings
data_filename = 'data_complete.csv'
assets_filename = 'asset_details.csv'
history_start = '2021-06-12 00:01:00'
forecast_start = '2021-06-13 23:59:00'
forecast_end ='2021-06-14 01:00:00'
attribute_cols = ['Asset_ID']
timestamp_col = 'timestamp'
target_col = 'Target'
feature_cols = ['timestamp','Count','Open','High','Low','Close','Volume','VWAP']
#%% Read data
df = pd.read_csv(data_filename)
assets = pd.read_csv(assets_filename)

#%% Setup experiment data
experiment = Experiment(
                df,
                history_start,
                forecast_start,
                forecast_end,
                timestamp_col,
                attribute_cols,
                feature_cols,
                target_col
                )
                
# for col in ['Open','High','Low','Close']:
#     experiment.factory.add_rolling_max(col)
#     experiment.factory.add_rolling_mean(col)
#     experiment.factory.add_rolling_median(col)

# experiment.factory.clean_features()
print(experiment.factory.data)

# %% Linear model per each Asset
import statsmodels.api as sm
for name, group in experiment.factory.data.groupby(attribute_cols):
    X_train, X_test, y_train = experiment.get_X_y(group)
    model = sm.OLS(y_train,X_train)
    results = model.fit()
    print(name)
    # print(results.summary())
    experiment.factory.data['forecast'] = results.predict(pd.concat([X_train, X_test]))

# %% Arima model per each Asset
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest

# for name, group in experiment.factory.data.groupby(attribute_cols):

btc = experiment.factory.data[experiment.factory.data['Asset_ID'] == 1]
adf_test = ADFTest(alpha = 0.05)
adf_test.should_diff(btc['Target'])

btc_train,btc_test = experiment.get_train_test(btc)
btc_train,btc_test = btc_train[target_col],btc_test[target_col]
arima_model =  auto_arima(btc_train,start_p=0, d=1, start_q=0, 
                          max_p=5, max_d=5, max_q=5, start_P=0, 
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True, 
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 50 )


arima_model.summary()
prediction = pd.Series(arima_model.predict(n_periods = len(btc)),index=btc.index)
from sklearn.metrics import r2_score
r2_score(btc_test, prediction)

result = pd.DataFrame()
result['actual'] = btc[target_col]
result['forecast'] = prediction
plot(result,'index',['actual','forecast'])
# %%
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(btc_train)
pyplot.show()