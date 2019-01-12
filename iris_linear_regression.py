import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import recall_score, precision_score, mean_squared_error

iris_data = pd.read_csv('../data/iris.csv')
ohe = OneHotEncoder(categorical_features='all')

x = iris_data.iloc[:, 0:4].values
y = iris_data.iloc[:,4:5]
# preprocessing

normalized_x = Normalizer().fit_transform(x)
encoded_y = ohe.fit_transform(y.apply(LabelEncoder().fit_transform)).toarray()


x_train, x_test, y_train, y_test = train_test_split(normalized_x, encoded_y[:,0:1], random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)
print(cross_val_score(lr, x_train, y_train, cv=3))
# print('recall ', recall_score(y_test, prediction))
print('precision ', precision_score(y_test, prediction))
print('mean squared error ', mean_squared_error(y_test, prediction))
