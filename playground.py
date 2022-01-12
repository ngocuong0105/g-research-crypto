#%%
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
train = pd.read_csv("train.csv")
assets = pd.read_csv("asset_details.csv")
train.insert(loc=0, column='Time', value=pd.to_datetime(train['timestamp'],unit='s'))
train.info(show_counts=True)


# %%
# Reproducing the target column
def ResidualizeMarket(df, mktColumn, window):
    if mktColumn not in df.columns:
        return df

    mkt = df[mktColumn]

    num = df.multiply(mkt.values, axis=0).rolling(window).mean().values  #numerator of linear regression coefficient
    denom = mkt.multiply(mkt.values, axis=0).rolling(window).mean().values  #denominator of linear regression coefficient
    beta = np.nan_to_num( num.T / denom, nan=0., posinf=0., neginf=0.)  #if regression fell over, use beta of 0
    resultRet = df - (beta * mkt.values).T  #perform residualization
    resultBeta = 0.*df + beta.T  #shape beta

    return resultRet.drop(columns=[mktColumn]), resultBeta.drop(columns=[mktColumn])

# EVALUATION METRIC
# 'a' and 'b' are the expected and predicted targets, and ' weights' include the weight of each row, determined by its asset.
def weighted_correlation(a, b, weights):

  w = np.ravel(weights)
  a = np.ravel(a)
  b = np.ravel(b)

  sum_w = np.sum(w)
  mean_a = np.sum(a * w) / sum_w
  mean_b = np.sum(b * w) / sum_w
  var_a = np.sum(w * np.square(a - mean_a)) / sum_w
  var_b = np.sum(w * np.square(b - mean_b)) / sum_w

  cov = np.sum((a * b * w)) / np.sum(w) - mean_a * mean_b
  corr = cov / np.sqrt(var_a * var_b)

  return corr

# Function log_return_ahead computes R_t = log(P_{t+16} / P_{t+1})
# define function to compute log returns
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)



# %%
ids = list(assets.Asset_ID)
asset_names = list(assets.Asset_Name)

# times = data['timestamp'].agg(['min', 'max']).to_dict()
# all_timestamps = np.arange(times['min'], times['max'] + 60, 60)

all_timestamps = np.sort(train['timestamp'].unique())
targets = pd.DataFrame(index=all_timestamps)
# %%

price_column = 'Close'
for i, id in enumerate(ids):
    asset = train[train.Asset_ID == id].set_index(keys='timestamp')
    price = pd.Series(index=all_timestamps, data=asset[price_column])
#   targets[asset_names[i]] = np.log(
#         price.shift(periods=-16) /
#         price.shift(periods=-1)
#     )
    targets[asset_names[i]] = (
        price.shift(periods=-16) /
        price.shift(periods=-1)
    ) - 1


# %%
# Next calculate M as weighted mean of columns (currently holding R value for each asset) for each row. 
# This implementation count unavailable R values as zero. 

weights = np.array(list(assets.Weight))
targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)
#%%
targets = ResidualizeMarket(targets,'m',3750)[0]
# %%
diffs = []

for i, id in enumerate(ids):
    print(asset_names[i])
    # type: pd.DataFrame
    asset = train[train.Asset_ID == id].set_index(keys='timestamp')
    print(f'asset size {asset.shape[0]}')
    recreated = pd.Series(index=asset.index, data=targets[asset_names[i]])
    diff = np.abs(asset['Target'] - recreated)
    diffs.append(diff[~pd.isna(diff)].values)
    print(f'Average absolute error {diff.mean():8.6f}')
    print(f'Max absolute error {diff.max():8.6f}')
    print(f'Standard deviation {diff.std():8.6f}')
    print(f'Target na {pd.isna(asset.Target).sum()}')
    print(f'Target_calculated na {pd.isna(recreated).sum()}')
    print()

diffs = np.concatenate(diffs, axis=0)
print('For all assets')
print(f'Average absolute error {diffs.mean():8.6f}')
print(f'Max absolute error {diffs.max():8.6f}')
print(f'Standard deviation {diffs.std():8.6f}')

#%%
prices = train.pivot(index=["timestamp"], columns=["Asset_ID"], values=["Close"])
prices.columns = [f"A{a}" for a in range(14)]

prices = prices.reindex(range(prices.index[0], prices.index[-1]+60,60), method='pad')
prices.index = prices.index.map(lambda x: datetime.fromtimestamp(x))
prices.sort_index(inplace=True)
