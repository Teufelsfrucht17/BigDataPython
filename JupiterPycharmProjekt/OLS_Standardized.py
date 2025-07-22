from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Read original Data from CSV
data = pd.read_csv('UsedCarSellingPrices.csv')
print(data.head())

data_numeric = pd.get_dummies(data)
print(data_numeric.head())

nscaler = preprocessing.MinMaxScaler()
data = nscaler.fit_transform(data_numeric)
data_df = pd.DataFrame(data, columns=data_numeric.columns)
print(data_df.head())

data_df.to_csv('DataNormed.csv', index=False)

X = data_df.drop(columns=['selling_price']) # Feature
Y = data_df['selling_price'] # Variable

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(Y_train.shape)
print(Y_test.shape)

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_train_pred = lm.predict(X_train)

R2 = r2_score(Y_train, Y_train_pred)
# Y_train_dev = sum((Y_train - Y_train_pred) ** 2)
# R2 = 1 - (Y_train_dev / Y_train_meandev)
print(R2)

Y_test_pred = lm.predict(X_test)
PseudoR2 = r2_score(Y_test, Y_test_pred)

# Y_test_dev = sum((Y_test - Y_test_pred) ** 2)
# PseudoR2 = 1 - (Y_test_dev / Y_train_meandev)
print(PseudoR2)