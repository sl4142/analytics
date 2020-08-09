# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import datetime
%matplotlib inline

path_amzn = 'D:\\github\\analytics\\data\\AMZN.csv'
amzn = pd.read_csv(path_amzn)
amzn.set_index('Date', inplace=True)
amzn.index = pd.to_datetime(amzn.index)

print(amzn)
