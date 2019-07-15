import requests
import os
import errno
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import numpy as np
import pandas as pd
import sched, time
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

base_url = 'https://www.bitstamp.net/api/v2/ticker_hour/'
daily_url = 'https://www.bitstamp.net/api/ticker/'

DAY_LIMIT = 24*60
TWO_HOUR_LIMIT = 60*2
MINUTE_INTERVAL = 60
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

#def collectOneData(url):
#    r = requests.get(url) 
#    return r.json()

def collectOneData(url):
    s = requests.Session()
    retries = Retry(total=10000,
                backoff_factor=0.1,
                status_forcelist=[ 500, 502, 503, 504 ])

    s.mount('http://', HTTPAdapter(max_retries=retries))
    r = s.get(url)
    return r.json()

def writeFiles(dirName, filename, df):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    df.to_csv(dirName+filename)

def collectAllData(sc, urls, allData, cols, HOUR_LIMIT, DAY_LIMIT, i=1, interval=60):
    curLimit = 0
    for ticker, url in urls.items():
        curData = collectOneData(url)
        for col in cols:
            allData[ticker][col].append(curData[col])
            curLimit = max(curLimit, len(allData[ticker][col]))
    if  curLimit % HOUR_LIMIT == 0:
        ts = datetime.now().timestamp()
        curDate = datetime.utcfromtimestamp(ts).strftime('%Y%m%d')
        for ticker in urls.keys():
            curDf = pd.DataFrame.from_dict(allData[ticker])
            dirName = 'data/' + ticker + '/' + curDate + '/'
            fileName = ticker + curDate + '_' + str(i) + '.csv'
            writeFiles(dirName, fileName, curDf)
        allData = initializeDataDict(TICKERS, cols)
        i += 1
        print(curDate + str(i))
    if i % (DAY_LIMIT / HOUR_LIMIT) == 0:
        i = 0
    sc.enter(interval, 1, collectAllData, (sc,urls, allData, cols, HOUR_LIMIT, DAY_LIMIT, i, interval))