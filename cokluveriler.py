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

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')

#encoder:  Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()

# ----
cinsiyet = veriler.iloc[:,-1:].values
le = LabelEncoder()
cinsiyet[:,0] = le.fit_transform(cinsiyet[:,0])
ohe = OneHotEncoder(categorical_features='all')
cinsiyet = ohe.fit_transform(cinsiyet).toarray()


#numpy dizileri dataframe donusumu
cinsiyet_erkek_dataframe = pd.DataFrame(data=cinsiyet[:,0], index=range(22), columns=['cinsiyet'])
cinsiyet_kadin_dataframe = pd.DataFrame(data=cinsiyet[:,1], index=range(22), columns=['cinsiyet'])

ulke_dataframe = pd.DataFrame(data=ulke, index=range(22), columns=['us', 'tr', 'fr'])

boy_dataframe = pd.DataFrame(veriler.iloc[:, 1:2])
kilo_dataframe = pd.DataFrame(veriler.iloc[:, 2:3])
yas_dataframe = pd.DataFrame(veriler.iloc[:, 3:4])

#dataframe birlestirme islemi
sonuc = pd.concat([ulke_dataframe, kilo_dataframe, yas_dataframe, cinsiyet_erkek_dataframe], axis=1)

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sonuc, boy_dataframe, test_size=0.33, random_state=0)

'''
    Modelleme, tahmin etme ve skorlama
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

predict = lr.predict(x_test)