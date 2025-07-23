from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

###########
# One-Hot #
###########

# Read original Data from CSV
data = pd.read_csv('UsedCarSellingPrices.csv')
#print(data.head())

data['Brand'] = data['name'].str.split().str[0]
#print(data)

data = data.drop(columns=['name'])

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

X = data_df_OH.drop(columns=['selling_price']) # Feature
Y = data_df_OH['selling_price'] # Variable

X_train_OH, X_test_OH, Y_train_OH, Y_test_OH = train_test_split(X, Y, test_size=0.2, random_state=42)
print(Y_train_OH.shape)
print(Y_test_OH.shape)

lm = LinearRegression()
lm.fit(X_train_OH, Y_train_OH)

Y_train_pred_OH = lm.predict(X_train_OH)

R2_OH = r2_score(Y_train_OH, Y_train_pred_OH)
# Y_train_dev = sum((Y_train - Y_train_pred) ** 2)
# R2 = 1 - (Y_train_dev / Y_train_meandev)
print("R2 OH: ")
print(R2_OH)

Y_test_pred_OH = lm.predict(X_test_OH)
PseudoR2_OH = r2_score(Y_test_OH, Y_test_pred_OH)

# Y_test_dev = sum((Y_test - Y_test_pred) ** 2)
# PseudoR2 = 1 - (Y_test_dev / Y_train_meandev)
print("PseudoR2 OH: ")
print(PseudoR2_OH)

