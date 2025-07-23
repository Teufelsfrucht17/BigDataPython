from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


# Read original Data from CSV
data = pd.read_csv('UsedCarSellingPrices.csv')
#print(data.head())

data['Brand'] = data['name'].str.split().str[0]
#print(data)

data = data.drop(columns=['name'])

data.to_csv('AllData.csv', index=False)

#################
# Lable Encoded #
#################

data_LE = data.copy()
categorical_columns = data_LE.select_dtypes(include=['object']).columns

label_encoder = LabelEncoder()
for column in categorical_columns:
    data_LE[column] = label_encoder.fit_transform(data_LE[column])

nscaler = preprocessing.MinMaxScaler()
data_scaled_LE = nscaler.fit_transform(data_LE)

data_df_LE = pd.DataFrame(data_scaled_LE, columns=data.columns)
#print(data_df_LE.shape)
#print(data_df_LE)

X_LE = data_df_LE.drop(columns=['selling_price']) # Feature
Y_LE = data_df_LE['selling_price'] # Variable




###########
# One-Hot #
###########

data_OH = pd.get_dummies(data)
#print(data_OH.head())

nscaler = preprocessing.MinMaxScaler()
data_scaled_OH = nscaler.fit_transform(data_OH)
# data = (data_numeric - data_numeric.min()) / (data_numeric.max() - data_numeric.min())

#print(data_scaled_OH.shape)
#print(data_scaled_OH.shape)
data_df_OH = pd.DataFrame(data_scaled_OH, columns=data_OH.columns)
#print(data_df_OH.head())

data_df_OH.to_csv('One_Hot_Data.csv', index=False)

X_OH = data_df_OH.drop(columns=['selling_price']) # Feature
Y_OH = data_df_OH['selling_price'] # Variable

