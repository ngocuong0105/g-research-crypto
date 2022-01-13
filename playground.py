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
