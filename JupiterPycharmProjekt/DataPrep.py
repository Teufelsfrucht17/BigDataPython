from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

################
# Prepare Data #
################

# Read original Data from CSV
data = pd.read_csv('UsedCarSellingPrices.csv')
#print(data.head())

# Introduce Variable (Column) "Brand" - Car brand that is mentioned in Columne "Name"
data['Brand'] = data['name'].str.split().str[0]
#print(data)

# Drop Variable "name" to make Lable encoding and One-Hot-Encoding possible - Regression can only handle numerical values
data = data.drop(columns=['name'])

# Save non-Encoded data as CSV
data.to_csv('AllData.csv', index=False)

#################
# Lable Encoded #
#################

# Select all variables (columns) that have data in type "Object" - non-numeric values - and save them in variable "categorical_columns"
data_LE = data.copy()
categorical_columns = data_LE.select_dtypes(include=['object']).columns

# Introduce Lable Encoder function and let is run thrughe all columns defined in variable "categorical_columns", changing them into numerical values
label_encoder = LabelEncoder()
for column in categorical_columns:
    data_LE[column] = label_encoder.fit_transform(data_LE[column])

# scale the numerical values via Min-Max-scaler
nscaler = preprocessing.MinMaxScaler()
data_scaled_LE = nscaler.fit_transform(data_LE)

# Reeitroduce the Columne names using dataframe
data_df_LE = pd.DataFrame(data_scaled_LE, columns=data.columns)
#print(data_df_LE.shape)
#print(data_df_LE)

# Save Lable Encoded Data as CSV
data_df_LE.to_csv('Lable_Encoded_Data.csv', index=False)

# Define the Feature (X_LE) and the Variables (Y_LE) for Regression
X_LE = data_df_LE.drop(columns=['selling_price']) # Feature
Y_LE = data_df_LE['selling_price'] # Variable


###########
# One-Hot #
###########

# Create dataset with One-Hot-Encoding for non-numerical data columns
data_OH = pd.get_dummies(data)
#print(data_OH.head())

# Scale One-Hot-Encoded data using a Min-Max-Scaler
scaler = preprocessing.MinMaxScaler()
data_scaled_OH = scaler.fit_transform(data_OH)
# data = (data_numeric - data_numeric.min()) / (data_numeric.max() - data_numeric.min())

#print(data_scaled_OH.shape)
#print(data_scaled_OH.shape)

# Reintroduce the Columne names using dataframe
data_df_OH = pd.DataFrame(data_scaled_OH, columns=data_OH.columns)
#print(data_df_OH.head())

# Save One-Hot-Encoded Data as CSV
data_df_OH.to_csv('One_Hot_Data.csv', index=False)

# Define the Feature (X_OH) and the Variables (Y_OH) for Regression
X_OH = data_df_OH.drop(columns=['selling_price']) # Feature
Y_OH = data_df_OH['selling_price'] # Variable

## creating a report sheam

report = pd.DataFrame(columns=['Model','R2.Train','R2.Test','RMSE','R2_Mean_CV','R2_Std_CV'])
