#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')
# preprocessing #
# categorical attiributes
ohe = OneHotEncoder(categorical_features='all')

outlook = ohe.fit_transform(veriler.iloc[:,0:1].apply(LabelEncoder().fit_transform)).toarray()
play = veriler.iloc[:,4:5].apply(LabelEncoder().fit_transform)
windy = veriler.iloc[:,3:4].apply(LabelEncoder().fit_transform)

# numerical attiributes
humidity = veriler.iloc[:,2:3]
temperature = veriler.iloc[:,1:2]

outlook = pd.DataFrame(data=outlook, index=range(14), columns=['overcast', 'rainy', 'sunny'])
windy = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
play = pd.DataFrame(data=play, index=range(14), columns=['play'])

# standardizing
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# humidity = sc.fit_transform(humidity)
# temperature = sc.fit_transform(temperature)

# humidity = pd.DataFrame(data=humidity, index=range(14), columns=['humidity'])
# temperature = pd.DataFrame(data=temperature, index=range(14), columns=['temperature'])

# joining frames on data
data = pd.concat([outlook, windy, temperature, play] , axis=1)

# dividing datas for training and testing with cross-validation
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train, y_test = train_test_split(data, humidity, test_size=0.33, random_state=0)

# creating linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

predict = lr.predict(x_test)
lr.score(predict, y_test.iloc[:,0:].values)
