'''
This file is for testing ideas.
'''
#%%
import numpy as np
import pandas as pd
from datetime import datetime
from utils import get_time_range, compute_targets, compare_targets, plot, complete_single, \
    log_return, shifted_return, complete

#%% Read Data
data = pd.read_csv("train.csv")
data = data.fillna(0)
assets = pd.read_csv("asset_details.csv")
data.insert(loc=0, column='Time', value=pd.to_datetime(data['timestamp'],unit='s'))
data.info(show_counts=True)
train = data[data['timestamp']<1623542400]
test = data[data['timestamp']>=1623542400]

#%% Reproduce targets (returns)
targets = compute_targets(data,assets,'Close')
asset_times = get_time_range(train,assets)
compare_targets(data,targets,assets)

#%% Bitcoin example
btc = data[data['Asset_ID']==1]
btc.reset_index(inplace = True,drop=True)
plot(btc[-1000:], 'Time', ['Open','Close'])
btc_complete = complete_single(btc, 'timestamp', 60)
plot(btc_complete[-2000:],'Time',['High','Low','Open'],'BTC')
# btc returns
btc_complete['log_return'] = log_return(btc_complete['Close'],periods=15)
btc_complete['shifted_return'] = shifted_return(btc_complete['Close'],periods=15)
plot(btc_complete,'Time',['shifted_return','log_return'])

#%% Complete time-series data
data_complete = complete(data,['Asset_ID'],'timestamp',60)
data_complete.to_csv('data_complete.csv',index=False)

#%%
prices = train.pivot(index=["timestamp"], columns=["Asset_ID"], values=["Close"])
prices.columns = [f"A{a}" for a in range(14)]

prices = prices.reindex(range(prices.index[0], prices.index[-1]+60,60), method='pad')
prices.index = prices.index.map(lambda x: datetime.fromtimestamp(x))
prices.sort_index(inplace=True)


#%% TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST TEST

from sklearn import linear_model, preprocessing, decomposition, pipeline, feature_selection, impute
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin

from utils import totimestamp, log_return, plot
#%% Configure settings
data_filename = 'data_complete.csv'
history_start = '2019-04-12'
forecast_start = '2021-06-13'
forecast_end ='2021-09-13'
attribute_cols = ['Asset_ID']
timestamp_col = 'timestamp'
forecast_col = 'Target'

feature_cols = ['timestamp','Count','Open','High','Low','Close','Volume','VWAP']
#%% Read data
df = pd.read_csv(data_filename)


# %%
train = df[
    (df[timestamp_col]>=totimestamp(history_start)) &
    (df[timestamp_col]<totimestamp(forecast_start))
    ]
test = df[
    (df[timestamp_col]>=totimestamp(forecast_start)) &
    (df[timestamp_col]<totimestamp(forecast_end))
    ]
all_data = pd.concat([train,test])

btc_train = train[train['Asset_ID'] == 1]
X_btc_train = btc_train[feature_cols]
y_btc_train = btc_train[forecast_col]

btc_all = all_data[all_data['Asset_ID'] == 1]
X_btc_all = btc_all[feature_cols]
y_btc_all = btc_all[forecast_col]

btc_test = test[test['Asset_ID'] == 1]
X_btc_test= btc_test[feature_cols]
y_btc_test = btc_test[forecast_col]


#%%

def debug(array):
    return array

pipe = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('debug', preprocessing.FunctionTransformer(debug)),
    ('impute', impute.SimpleImputer()),
    ('pca', decomposition.PCA()),
    ('filter constants', feature_selection.VarianceThreshold(0.05)),
    ('model', linear_model.LinearRegression())
])

model = pipe.fit(X_btc_train,y_btc_train)
pred = model.predict(X_btc_all)

print('R2 on train:', model.score(X_btc_train,y_btc_train))
print('R2 on test:', model.score(X_btc_test,y_btc_test))
print('R2 on all:', model.score(X_btc_all,y_btc_all))
# %%
def debug(array):
    return array

def log_return_feature(series, periods=1):
    return np.log(series).diff(periods=periods)

pipe = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('pca', decomposition.PCA()),
    ('filter constants', feature_selection.VarianceThreshold(0.05))
])

X_btc_train_transformed = pipe.fit_transform(X_btc_train,y_btc_train)
X_btc_all_transformed = pipe.transform(X_btc_all)
X_btc_all_transformed = sm.add_constant(X_btc_all_transformed)

# Tune Linear model, pipeline,
# Try Statsmodels
# %%
model = sm.OLS(y_btc_all,X_btc_all_transformed,hasconst=False)
results = model.fit()
results.summary()
# %%
# EDA
data = pd.DataFrame(X_btc_train_transformed)
data['Target'] = y_btc_train.reset_index(drop = True)

# %%

upper_shadow = lambda asset: asset.High - np.maximum(asset.Close,asset.Open)
lower_shadow = lambda asset: np.minimum(asset.Close,asset.Open)- asset.Low

X_btc = pd.concat([log_return_feature(X_btc_train.VWAP,periods=5), log_return_feature(X_btc_train.VWAP,periods=1).abs(), 
               upper_shadow(X_btc_train), lower_shadow(X_btc_train)], axis=1).fillna(0)
# %%

model = sm.OLS(y_btc_train,X_btc)
results = model.fit()
results.summary()
# %%
