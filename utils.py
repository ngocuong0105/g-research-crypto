'''
This file serves as a placeholder for common functions used in the project.
'''
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns

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

# auxiliary function, from datetime to timestamp
def totimestamp(s):
    return np.int32(time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()))

# Function log_return_ahead computes R_t = log(P_{t+16} / P_{t+1})
# define function to compute log returns
def log_return(series, periods=1):
    return -np.log(series).diff(periods=-periods)

def shifted_return(series, periods=1):
    return series.shift(periods=-periods-1)/series.shift(periods=-1) - 1

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

def plot(data, timestamp_col, value_cols, title = '', last_minutes = 5000, line_date = None, renderer = 'notebook'):
    data = data[-last_minutes:]
    if timestamp_col in data:
        data.insert(loc=0, column='t', value=pd.to_datetime(data[timestamp_col],unit='s'))
    else:
        data.insert(loc=0, column='t', value=data.index)
    DEFAULT_LAYOUT = dict(
    xaxis=dict(
        type='date',
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                    label='1s',
                    step='second',
                    stepmode='backward'),
                dict(count=10,
                    label='10s',
                    step='second',
                    stepmode='backward'),
                dict(count=1,
                    label='60s',
                    step='minute',
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
            x=data['t'],
            y=data[value_col], 
            name=f'{value_col}'
        )
    if line_date:
        forecast_start = pd.to_datetime(line_date)
        fig.add_shape(go.layout.Shape(
                type="line", yref="paper", x0=forecast_start, y0=0, x1=forecast_start, y1=1,
                line=dict(color="Red", width=1),
            ))
    fig.show(renderer = renderer)

def complete_single(df, timestamp_col:str, time_freq:int, method = 'pad'):
    df = df.set_index(timestamp_col)
    df = df.reindex(range(min(df.index),max(df.index)+time_freq,time_freq),method=method)
    df = df.reset_index()
    return df

def complete(data:pd.DataFrame, attribute_cols:'list[str]', timestamp_col:str, time_freq:int) -> pd.DataFrame:
    dfs = []
    for _,df in data.groupby(attribute_cols):
        df = complete_single(df,timestamp_col,time_freq)
        dfs.append(df)
    return pd.concat(dfs)


def corr_map(data_complete, assets, timestamp_col = 'timestamp', start='01/01/2021', end='01/05/2021'):
    it = 0
    all_assets = pd.DataFrame([])
    for asset_id, asset_name in zip(assets.Asset_ID, assets.Asset_Name):
        asset = data_complete[data_complete["Asset_ID"]==asset_id].set_index(timestamp_col)
        asset = asset.loc[totimestamp(start):totimestamp(end)]
        lret = pd.DataFrame({f'{asset_name}': log_return(asset.Close)[1:]})
        if it == 0: all_assets = lret
        else: all_assets = pd.concat([all_assets, lret], axis=1)
        it += 1

    corr_df = all_assets.corr()
    mask_ut = np.triu(np.ones(corr_df.shape)).astype(np.bool)
    sns.heatmap(corr_df, mask=mask_ut, cmap="Spectral")

def makelist(itr):
    if itr is None:
        return None
    if isinstance(itr, str):
        return [itr]
    try:
        return list(itr)
    except TypeError:
        return [itr]