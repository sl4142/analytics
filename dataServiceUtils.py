import requests
import numpy as np
import pandas as pd
import sched, time
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

base_url = 'https://www.bitstamp.net/api/v2/ticker_hour/'
daily_url = 'https://www.bitstamp.net/api/ticker/'

DAY_LIMIT = 24*60
TICKERS = ['btc', 'ltc', 'eth', 'bch', 'xrp']
currPair = ['btcusd', 'ltcusd', 'ethusd', 'bchusd', 'xrpusd']
url_hr = 'https://www.bitstamp.net/api/ticker_hour/'
urls = [url_hr + curr + '/' for curr in currPair]
cols = ['high', 'last', 'timestamp', 'bid', 'vwap', 'volume', 'low', 'ask', 'open']

def tsToDtEst(ts):
    return datetime.utcfromtimestamp(ts)-timedelta(seconds=3600*4)
    
def dtToString(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def initializeOneDict(cols):
    output = {}
    for col in cols:
        output[col] = []
    return output

def initializeDataDict(TICKERS, cols):
    output = {}
    for ticker in TICKERS:
        output[ticker] = initializeOneDict(cols)
    return output

def collectOneData(url):
    r = requests.get(url) 
    return r.json()

def collectAllData(sc, urls, allData, cols, DAY_LIMIT, interval=60):
    curLimit = 0
    for ticker, url in urls.items():
        curData = collectOneData(url)
        for col in cols:
            allData[ticker][col].append(curData[col])
            curLimit = max(curLimit, len(allData[ticker][col]))
    if  curLimit % DAY_LIMIT == 0:
        ts = datetime.now().timestamp()
        curDate = datetime.utcfromtimestamp(ts).strftime('%Y%m%d')
        for ticker in urls.keys():
            curDf = pd.DataFrame.from_dict(allData[ticker])
            curDf.to_csv('data/' + ticker + '/' + ticker + curDate + '.csv')
        allData = initializeDataDict(TICKERS, cols)
        print(curDate)
    sc.enter(interval, 1, collectAllData, (sc,urls, allData, cols, DAY_LIMIT, interval))