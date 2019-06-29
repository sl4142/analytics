#!/usr/bin/env python
# coding: utf-8

# In[9]:


import yfinance as yf
import datetime
import requests
import urllib.request
import time
import pandas as pd
import numpy as np
import re
import scipy as sp
from scipy import stats
from bs4 import BeautifulSoup

def get_data(symbols, start, end):
    start = start.split("/")
    end = end.split("/")
    start = datetime.datetime(int(start[0]), int(start[1]), int(start[2]))
    end = datetime.datetime(int(end[0]), int(end[1]), int(end[2]))
    return yf.download(symbols, start=start, end=end)

def get_symbols(url, tag="a", attr="", fromText=False):
    response = requests.get(url)
    if not response:
        print("Status code > 400, web connection failed.")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    symbols = []
    for cur in soup.findAll(tag):
        if fromText:
            if cur.get_text() != None:
                symbols.append(cur.get_text())
        else:
            if cur.get(attr) != None:
                symbols.append(cur.get(attr))
    return symbols

def handle_one_df(data, symbol, cols):
    cur_col = [col for col in data.columns if symbol in col]
    return pd.DataFrame(np.array(data[cur_col]), columns=cols, index=data[cur_col].index)

def get_data_dict(symbols, data, cols, batch=10):
    df_dict = {}
    batch_steps = np.arange(0, len(symbols), batch)
    for i in range(1, len(batch_steps)+1):
        cur_symbols = symbols[batch_steps[i-1]:batch_steps[i]] if i != len(batch_steps) else symbols[batch_steps[i-1]:]
        for symbol in cur_symbols:
            df_dict[symbol] = handle_one_df(data, symbol, cols)
    print("Finished")
    return df_dict

def get_symbols_single_page(url, tag="a", attr="", fromText=False):
    response = requests.get(url)
    if not response:
        print("Status code > 400, web connection failed.")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    symbols = []
    for cur in soup.findAll(tag):
        if fromText:
            if cur.get_text() != None:
                symbols.append(cur.get_text())
        else:
            if cur.get(attr) != None:
                symbols.append(cur.get(attr))
    return symbols

def get_symbols_multiple_pages(url, tag="a", attr="", fromText=False):
    response = requests.get(url)
    if not response:
        print("Status code > 400, web connection failed.")
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    total_results = 0
    for cur in soup.findAll('span'):
        search = re.search(r"1-(.|..|...) of (.|..|...|....|.....|......) results", cur.get_text())
        if search != None:
            total_results = cur.get_text().split(" ")[2]
            break
    total_urls = int(int(total_results)/100)
    urls = []
    for i in range(total_urls+1):
        urls.append(url + "?count=100&offset=" + str(i*100))
    if not urls:
        return get_symbols_single_page(url, tag, attr, fromText)
    else:
        symbols = []
        for cur_url in urls:
            symbols += get_symbols_single_page(cur_url, tag, attr, fromText)
    print("Get Symbol Finished")
    return symbols
    
def get_rank(data_dict, symbols, target_col, measure):
    target_list, target_rank = [], []
    for symbol in symbols:
        target_list.append(data_dict[symbol].describe()[target_col][measure])
    df_target_col = pd.DataFrame(np.array(target_list), index=symbols, columns=[target_col])
    for cur in target_list:
        target_rank.append(stats.percentileofscore(target_list, cur))
    df_target_col[target_col+" Rank"] = target_rank
    return df_target_col

