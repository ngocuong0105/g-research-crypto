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
experiment.factory.clean_features()
for col in ['Open','High','Low','Close']:
    experiment.factory.add_rolling_max(col)
    experiment.factory.add_rolling_mean(col)
    experiment.factory.add_rolling_median(col)

print(experiment.factory.data)

# %% Linear model per each Asset
# for ids in assets['Asset_ID']:
ids = 1
X_train, X_test, y_train = experiment.get_X_y(Asset_ID = ids)
factory = FeatureFactory(X_train, X_test, y_train)
for col in ['Open','High','Low','Close']:
    factory.add_rolling_max(col)
    factory.add_rolling_mean(col)
    factory.add_rolling_median(col)
factory.standard_feature_prep()
# %%
