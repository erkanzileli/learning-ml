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
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")

#veri on isleme
aylar = veriler[['Aylar']]

satislar = veriler[['Satislar']]

#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler, Normalizer

'''
    Verilerin Standardizasyonu
        Verilerin aynı dili konuşması amaçlanır
'''
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test= sc.fit_transform(y_test)

'''
    Modelleme, tahmin etme ve skorlama
'''
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)

predict = lr.predict(X_test)
score = lr.score(predict, Y_test)
print('predict', predict)
print('score', score)

'''
    Grafik oluşturma
    Önce indexlerine göre sıralanmalı
'''
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(X_train, Y_train)
plt.plot(X_test, lr.predict(Y_test))
plt.title('Aylara Göre Satışlar')
plt.xlabel('Aylar')
plt.ylabel('Satışlar')
