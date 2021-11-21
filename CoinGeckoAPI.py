#%%
from numpy.core.arrayprint import DatetimeFormat
from pandas.core.arrays import categorical
from pandas.core.indexing import _iLocIndexer
import requests
from datetime import datetime as dt
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import csv
import datetime
#%matplotlib inline


class CoinGecko_API:
    def __init__(self):
        self.url = 'https://api.coingecko.com/api/v3'
    def get_coin_historical_price(self,coin_name,starting_timestamp,ending_timestamp):
        path = self.url + '/coins/{}/market_chart/range?vs_currency=usd&from={}&to={}'.format(coin_name,starting_timestamp,ending_timestamp)
        response = requests.get(path)
        data_dict = json.loads(response.text)
        return data_dict['prices'],data_dict['market_caps'],data_dict['total_volumes']
def add_monthly_and_weekly_moving_avg(dataframe):
    monthly_moving_average = []
    weekly_moving_average = []
    for i in range(len(dataframe)-1,30,-1):
        monthly_sum = dataframe['Price'].iloc[i-31:i-1].sum()
        weekly_sum = dataframe['Price'].iloc[i-8:i-1].sum()
        monthly_avg = monthly_sum/30
        weekly_avg = weekly_sum/7
        monthly_moving_average.append(monthly_avg)
        weekly_moving_average.append(weekly_avg)
    monthly_moving_average.reverse()
    weekly_moving_average.reverse()
    return monthly_moving_average,weekly_moving_average
def moving_avg_for_prediction(dataframe):
    a = len(dataframe)
    monthly_sum = dataframe['Price'].iloc[a-30:a].sum() 
    weekly_sum  = dataframe['Price'].iloc[a-7:a].sum()
    return monthly_sum/30,weekly_sum/7
def convert_to_dataframe_for_prediction(timestamp,monthly_avg,weekly_avg):
    return pd.DataFrame({'TimeStamp':[timestamp],'MAmonth':[monthly_avg],'MAweek':[weekly_avg]})
def convert_to_dict(coin_price,coin_market_cap,coin_volume):
    price = []
    timestamp = []
    volume = []
    market_cap = []
    min_index = min(len(coin_price),len(coin_market_cap),len(coin_volume))
    for i in range(0,min_index,1):
        price.append(coin_price[i][1])
        timestamp.append(coin_price[i][0])
    for i in range(0,min_index,1):
        market_cap.append(coin_market_cap[i][1])
    for i in range(0,min_index,1):
        volume.append(coin_volume[i][1])
    data_dict = {'TimeStamp':timestamp,'Market_Cap':market_cap,'Volume':volume,'Price':price}
    return data_dict
cg_api = CoinGecko_API()

ct = datetime.datetime.now()
timestamp_now = ct.timestamp()
timestamp_five_year_ago = timestamp_now - (5*365*24*60*60)
print(timestamp_five_year_ago, timestamp_now)
bitcoin_price,bitcoin_matket_cap,bitcoin_volume = cg_api.get_coin_historical_price('bitcoin',timestamp_five_year_ago,timestamp_now)
ethereum_price,ethereum_market_cap,ethereum_volume = cg_api.get_coin_historical_price('ethereum', timestamp_five_year_ago, timestamp_now)
cardano_price,cardano_market_cap,cardano_volume = cg_api.get_coin_historical_price('cardano', timestamp_five_year_ago, timestamp_now)
dogecoin_price,dogecoin_market_cap,dogecoin_volume = cg_api.get_coin_historical_price('dogecoin', timestamp_five_year_ago, timestamp_now)
xrp_price,xrp_market_cap,xrp_volume = cg_api.get_coin_historical_price('ripple',timestamp_five_year_ago,timestamp_now)

ethereumDataFrame = pd.DataFrame(convert_to_dict(ethereum_price,ethereum_market_cap,ethereum_volume))

bitcoinDataFrame = pd.DataFrame(convert_to_dict(bitcoin_price,bitcoin_matket_cap,bitcoin_volume))
monthly_avg,weekly_avg = add_monthly_and_weekly_moving_avg(bitcoinDataFrame)
bitcoinDataFrame = bitcoinDataFrame.drop(labels=range(0,31))
bitcoinDataFrame['MAmonth'],bitcoinDataFrame['MAweek'] = monthly_avg,weekly_avg
timestamp_for_tomorrow = ethereumDataFrame['TimeStamp'][1823]+(24*60*60*1000)
month,week = moving_avg_for_prediction(bitcoinDataFrame)
bitcoinDataFrame_for_predicition = convert_to_dataframe_for_prediction(timestamp_for_tomorrow,month,week)



monthly_avg,weekly_avg = add_monthly_and_weekly_moving_avg(ethereumDataFrame)
ethereumDataFrame = ethereumDataFrame.drop(labels=range(0,31))
ethereumDataFrame['MAmonth'],ethereumDataFrame['MAweek'] = monthly_avg,weekly_avg
month,week = moving_avg_for_prediction(ethereumDataFrame)
ethereumDataFrame_for_predicition = convert_to_dataframe_for_prediction(timestamp_for_tomorrow,month,week)

cardanoDataFrame = pd.DataFrame(convert_to_dict(cardano_price,cardano_market_cap,cardano_volume))
monthly_avg,weekly_avg = add_monthly_and_weekly_moving_avg(cardanoDataFrame)
cardanoDataFrame = cardanoDataFrame.drop(labels=range(0,31))
cardanoDataFrame['MAmonth'],cardanoDataFrame['MAweek'] = monthly_avg,weekly_avg
month,week = moving_avg_for_prediction(cardanoDataFrame)
cardanoDataFrame_for_prediction = convert_to_dataframe_for_prediction(timestamp_for_tomorrow,month,week)

dogecoinDataFrame = pd.DataFrame(convert_to_dict(dogecoin_price,dogecoin_volume,dogecoin_volume))
monthly_avg,weekly_avg = add_monthly_and_weekly_moving_avg(dogecoinDataFrame)
dogecoinDataFrame = dogecoinDataFrame.drop(labels=range(0,31))
dogecoinDataFrame['MAmonth'],dogecoinDataFrame['MAweek'] = monthly_avg,weekly_avg
month,week = moving_avg_for_prediction(dogecoinDataFrame)
dogecoinDataFrame_for_prediction = convert_to_dataframe_for_prediction(timestamp_for_tomorrow,month,week)

xrpDataFrame = pd.DataFrame(convert_to_dict(xrp_price,xrp_market_cap,xrp_volume))
monthly_avg,weekly_avg = add_monthly_and_weekly_moving_avg(xrpDataFrame)
xrpDataFrame = xrpDataFrame.drop(labels=range(0,31))
xrpDataFrame['MAmonth'],xrpDataFrame['MAweek'] = monthly_avg,weekly_avg
month,week = moving_avg_for_prediction(xrpDataFrame)
xrpDataFrame_for_prediction = convert_to_dataframe_for_prediction(timestamp_for_tomorrow,month,week)

all_the_dates = bitcoinDataFrame['TimeStamp'].values.tolist()
dates = []
for date in all_the_dates:
    date = dt.fromtimestamp(date/1000)
    date = date.strftime("%m/%d/%Y")
    print(date)
    dates.append(date)
# %%
# %%
