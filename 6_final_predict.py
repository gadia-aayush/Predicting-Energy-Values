#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:36:50 2018

@author: aayushgadia
"""
#ONLY FOR MAY' 17

from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
 
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
    
 
# load and prepare datasets
dataset = Series.from_csv('final_dataset.csv')
X = dataset.values.astype('float32')
history = [x for x in X]
daily_records = 1
validation = Series.from_csv('final_validation.csv')
y = validation.values.astype('float32')

# load model
model_fit = ARIMAResults.load('final_model.pkl')

# make first prediction
predictions = list()
yhat = float(model_fit.forecast()[0])
yhat = inverse_difference(history, yhat, daily_records) #remove & also check
predictions.append(yhat)
history.append(y[0])
print('>Predicted=%.3f, Expected=%3.f' % (yhat, y[0]))

# rolling forecasts
for i in range(1, len(y)):
    # difference data
    daily_records = 1
    diff = difference(history, daily_records)
    
    # predict
    model = ARIMA(diff, order=(1,0,0))
    model_fit = model.fit(trend='nc', disp=0)
    yhat = model_fit.forecast()[0]
    yhat = inverse_difference(history, yhat,daily_records)
    predictions.append(yhat)
    
    # observation
    obs = y[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))

# report performance
mse = mean_squared_error(y, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)
pyplot.plot(y) 
pyplot.plot(predictions, color='red')
pyplot.show()