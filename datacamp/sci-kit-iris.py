#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:14:57 2019

@author: erkanzileli
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(iris_df.head())
pd.scatter_matrix(iris_df, c=iris.target, figsize=[8,8], s=150, marker='D')