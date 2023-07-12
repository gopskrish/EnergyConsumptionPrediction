#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 22:20:44 2023

@author: gopalakrishnan
"""

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
           
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import math

class EnergyConsumptionForecast:
  
    def __init__(self):
        self.EnergyConsumptionDataset = pd.read_csv('dataset.csv').dropna().sort_values(by="date")
        (self.shortTermDF, self.LongTermDF) = self.create_DataFrames()
        
    def create_DataFrames(self):
        data = self.EnergyConsumptionDataset
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        short_term = data
        long_term = pd.DataFrame(data.resample("QS-JUN")['total'].sum())
        return (short_term, long_term)

    def train_shortTermModel(self):
        mod = sm.tsa.statespace.SARIMAX(endog=self.shortTermDataset['total'], order=(4,0,5),seasonal_order=(12,1, 6, 12),trend='c')
        self.shortTermModel = mod.fit()
        return self.shortTermModel.fittedvalues
    
    def train_longTermModel(self):
        mod = ARIMA(self.LongTermDataset['total'], order=(3, 1, 1))
        self.longTermModel = mod.fit()
        return self.longTermModel.fittedvalues
        
    def train_model(self):
        return (self.train_shortTermModel(),  self.train_longTermModel())
    
    def shortTerm_Forecast(self, numberOfSteps = 3):
        return self.shortTermModel.forecast(steps = numberOfSteps)
    
    def longTerm_Forecast(self, numberOfSteps = 2):
        return self.longTermModel.forecast(steps = numberOfSteps)
    
