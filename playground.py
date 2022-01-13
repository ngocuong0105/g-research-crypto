#%%
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Read Data
data = pd.read_csv("train.csv")
assets = pd.read_csv("asset_details.csv")
data.insert(loc=0, column='Time', value=pd.to_datetime(data['timestamp'],unit='s'))
data.info(show_counts=True)
train = data[data['timestamp']<1623542400]
test = data[data['timestamp']>1623542400]

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

# Function log_return_ahead computes R_t = log(P_{t+16} / P_{t+1})
# define function to compute log returns
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

# evaluation metric
def weighted_correlation(actual, pred, weights):

  w = np.ravel(weights)
  actual = np.ravel(actual)
  pred = np.ravel(pred)

  sum_w = np.sum(w)
  mean_actual = np.sum(actual * w) / sum_w
  mean_pred = np.sum(pred * w) / sum_w
  var_actual = np.sum(w * np.square(actual - mean_actual)) / sum_w
  var_pred = np.sum(w * np.square(pred - mean_pred)) / sum_w

  cov = np.sum((actual * pred * w)) / np.sum(w) - mean_actual * mean_pred
  corr = cov / np.sqrt(var_actual * var_pred)
  return corr

# target column computation
def compute_targets(data:pd.DataFrame, assets:pd.DataFrame, price_col:str):
    ids = list(assets.Asset_ID)
    asset_names = list(assets.Asset_Name)
    # times = data['timestamp'].agg(['min', 'max']).to_dict()
    # all_timestamps = np.arange(times['min'] times['max'] + 60, 60)
    all_timestamps = np.sort(data['timestamp'].unique())
    targets = pd.DataFrame(index=all_timestamps)
    for i, id in enumerate(ids):
        asset = data[data.Asset_ID == id].set_index(keys='timestamp')
        price = pd.Series(index=all_timestamps, data=asset[price_col])
    #   targets[asset_names[i]] = np.log(
    #         price.shift(periods=-16) /
    #         price.shift(periods=-1)
    #     )
        targets[asset_names[i]] = (
            price.shift(periods=-16) /
            price.shift(periods=-1)
        ) - 1

    # Next calculate M as weighted mean of columns (currently holding R value for each asset) for each row. 
    # This implementation count unavailable R values as zero. 
    weights = np.array(list(assets.Weight))
    targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)
    targets = ResidualizeMarket(targets,'m',3750)[0]
    return targets

# sanity check
def compare_targets(data,targets,assets):
    diffs = []
    ids = list(assets.Asset_ID)
    asset_names = list(assets.Asset_Name)
    for i, id in enumerate(ids):
        print(asset_names[i])
        # type: pd.DataFrame
        asset = data[data.Asset_ID == id].set_index(keys='timestamp')
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

def get_time_range(data, assets):
    df = pd.DataFrame()
    mins,maxs = [],[]
    for ids in assets['Asset_ID']:
        sub = data[data['Asset_ID']==ids]
        mins.append(min(sub['Time']))
        maxs.append(max(sub['Time']))
    df['Asset_ID'] = assets['Asset_ID']
    df['Min_date'] = mins
    df['Max_date'] = maxs
    return df

def plot_candle(df_coin,title='', last_minutes = 5000):
    df_coin = df_coin[-last_minutes:]
    fig = go.Figure(data=[go.Candlestick(
        x=df_coin.index,
        open=df_coin['Open'], 
        high=df_coin['High'], 
        low=df_coin['Low'], 
        close=df_coin['Close'])])
    fig.update_layout(title_text=f'{title}', title_x=0.5)
    fig.show()

def plot_candles(data, assets, last_minutes = 5000):
    for ids in assets['Asset_ID']:
        coin_df = data[data['Asset_ID'] == ids]
        coin_df.set_index('Time',inplace = True)
        plot_candle(coin_df[-last_minutes:],f'Asset {ids}')

def plot(data, time_col, value_cols, title = '', last_minutes = 5000):
    data = data[-last_minutes:]
    DEFAULT_LAYOUT = dict(
    xaxis=dict(
        type='date',
        rangeselector=dict(
            buttons=list([
                dict(count=7,
                    label='1w',
                    step='day',
                    stepmode='backward'),
                dict(count=1,
                    label='1m',
                    step='month',
                    stepmode='backward'),
                dict(count=3,
                    label='3m',
                    step='month',
                    stepmode='backward'),
                dict(count=6,
                    label='6m',
                    step='month',
                    stepmode='backward'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ]),
            bgcolor = '#7792E3',
            font=dict(
                color = 'white',
                size = 13
            ),
        ),
        rangeslider=dict(
            visible=True
        ),
    ),
    height = 550
)
    fig = go.Figure(
        layout=DEFAULT_LAYOUT
    )
    fig.update_layout(title = title)
    if isinstance(value_cols,str):
        value_cols = [value_cols]

    for value_col in value_cols:
        fig.add_scatter(
            x=data[time_col],
            y=data[value_col], 
            name=f'{value_col}'
        )
    fig.show()

def complete_single(df, timestamp_col:str, time_freq:int, method = 'pad'):
    df = df.set_index(timestamp_col)
    df = df.reindex(range(min(df.index),max(df.index)+time_freq,time_freq),method=method)
    df = df.reset_index()
    return df

#%%
targets = compute_targets(data,assets,'Close')
compare_targets(data,targets,assets)
#%%
prices = train.pivot(index=["timestamp"], columns=["Asset_ID"], values=["Close"])
prices.columns = [f"A{a}" for a in range(14)]

prices = prices.reindex(range(prices.index[0], prices.index[-1]+60,60), method='pad')
prices.index = prices.index.map(lambda x: datetime.fromtimestamp(x))
prices.sort_index(inplace=True)

