'''
Experiments of all forecasting models in one place. Easier for performance comparison. 
'''
#%%
import pandas as pd
import numpy as np
from experiment import Experiment
from features import FeatureFactory 

#%% Configure settings
data_filename = 'data_complete.csv'
assets_filename = 'asset_details.csv'
history_start = '2019-04-12'
forecast_start = '2021-06-13'
forecast_end ='2021-09-13'
attribute_cols = ['Asset_ID']
timestamp_col = 'timestamp'
forecast_col = 'Target'
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
                forecast_col
                )
                
# train,test = experiment.get_train_test()
for col in ['Open','High','Low','Close']:
    experiment.factory.add_rolling_max(col)
    experiment.factory.add_rolling_mean(col)
    experiment.factory.add_rolling_median(col)

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

# %%
