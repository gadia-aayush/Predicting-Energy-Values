#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:21:55 2018

@author: aayushgadia
"""
from pandas import Series


#Data Loading & Dividing it for Modelling & Validation
series=Series.from_csv('/home/aayushgadia/Desktop/final.csv',header=0)
split_point=len(series)-1488
dataset, validation= series[0:split_point-1], series[split_point-1:len(series)]
dataset.to_csv('final_dataset.csv')  #For Modelling.
validation.to_csv('final_validation.csv')
