#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:36:50 2018

@author: aayushgadia
"""


from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
import numpy
 
# monkey patch around bug in ARIMA class
def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
#for adjusting the Error in Statsmodel    
 
ARIMA.__getnewargs__ = __getnewargs__
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
 
# load data
series = Series.from_csv('final_dataset.csv')
# prepare data
X = series.values
X = X.astype('float32')
# difference data
daily_records = 1
diff = difference(X, daily_records)

# fit model
model = ARIMA(diff, order=(1,0,0))
model_fit = model.fit(trend='nc', disp=0)

# save model
model_fit.save('final_model.pkl')
