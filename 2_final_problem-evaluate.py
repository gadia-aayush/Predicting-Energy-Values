#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:21:55 2018

@author: aayushgadia
"""
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy 


#Problem Understand & Model Evaluate:
#1. Baseline Prediction
#2. Different Plots


#BASE-LINE PREDICTION

# load data
series = Series.from_csv('final_dataset.csv')

# prepare data
X = series.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # predict
    yhat = history[-1]
    predictions.append(yhat)
    # observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%3.f' % (yhat, obs))
# report performance
mse = mean_squared_error(test, predictions)
rmse = sqrt(mse)
print('RMSE: %.3f' % rmse)


# Different Plots

#Line Plot
series_d = Series.from_csv('final_dataset.csv') #using only the Dataset seperated for Modelling
series_d.plot()
pyplot.show()

#Density Plot
pyplot.figure(1)
pyplot.subplot(211)
series_d.hist()
pyplot.subplot(212)
series_d.plot(kind='kde')
pyplot.show()
#Well we can see that it is not Normal Distribution, it is Binomial Distribution implying Non- Stationarity.
