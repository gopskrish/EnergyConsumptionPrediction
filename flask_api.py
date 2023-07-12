#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 22:44:59 2023

@author: gopalakrishnan
"""

from flask import Flask, request
import EnergyConsumptionForecast
app = Flask(__name__)
energyConsumptionForecast = EnergyConsumptionForecast.EnergyConsumptionForecast()

# Creating helath check method
@app.route('/', methods=['GET'])
def getstatus():
    return 'ok'

# route to get longTerm Forecast
@app.route('/longTerm', methods=['GET'])
def get_LongTermForecast(numberOfPreditions = 2):
    return energyConsumptionForecast.longTerm_Forecast(numberOfPreditions)

# route to get shortTerm Forecast
@app.route('/shortTerm', methods=['GET'])
def get_shortTermForecast(numberOfPreditions = 3):
    return energyConsumptionForecast.shortTerm_Forecast(numberOfPreditions)

# route to get food recommendation
@app.route('/train', methods=['POST'])
def train():
    return energyConsumptionForecast.train_model()

if __name__ == '__main__':
    app.run()
