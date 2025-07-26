####################
# Data Preperation #
####################

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read original Data from CSV
data = pd.read_csv('UsedCarSellingPrices.csv')


####################################################
# Label-Encoding for Visualisation before cleaning #
####################################################

# Create 'brand' as new columne before any encoding
data['brand'] = data['name'].str.split().str[0]  # [nicht in PDF]

# Define columns that need to be lable-encoded
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

# Copy data into var label_encoded_data; create empty array lable_encoders
label_encoded_data = data.copy()
label_encoders = {}

# Run lable-encoder for every previosly defined columne (function imported from sklearn)xx
for col in categorical_columns:
    le = LabelEncoder()
    label_encoded_data[col] = le.fit_transform(label_encoded_data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# [nicht in PDF] Falls Label-Encoding NaNs erzeugt (z. B. durch unerwartete Werte), mit 0 füllen
label_encoded_data = label_encoded_data.fillna(0)

print("\nLabel-Encoded data for visualisation:")
print(label_encoded_data.head())

# concatinates x and y into one point to be visualized
all_data_LableEncoded = pd.concat([label_encoded_data], axis=1)

# Drops all columes that are non-numeric to make scaling possible
all_data_LableEncoded = all_data_LableEncoded.select_dtypes(include=['number'])

# Scale data (normalized via MinMaxScaler - between 0 and 1)
#sscaler = preprocessing.StandardScaler()
#all_data_LableEncoded = sscaler.fit_transform(all_data_LableEncoded)
nscaler = preprocessing.MinMaxScaler()
all_data_LableEncoded = nscaler.fit_transform(all_data_LableEncoded)


#######################################
# Visualisation before cleaining Data #
#######################################

# Reintegrates Column name for boxplot
scaled_df = pd.DataFrame(all_data_LableEncoded, columns=label_encoded_data.select_dtypes(include='number').columns)

# Boxplot with readable x-achsis
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df, orient='v', palette='Set2')
plt.xticks(rotation=45)
plt.title("Normed boxplot")
plt.tight_layout()
plt.show()

# Boxplot for only numerical data
selected_cols = ['selling_price', 'km_driven', 'year']
plt.figure(figsize=(8, 5))
sns.boxplot(data=scaled_df[selected_cols], orient='v', palette='Set3')
plt.title("Normed boxplot for numerical data only")
plt.tight_layout()
plt.show()

# Pairplot to show correlation
sns.pairplot(scaled_df[selected_cols])
plt.suptitle("Pairplot for select charactaristics", y=1.02)
plt.show()


######################################
# Clean Data & Create variable Brand #
######################################

# Remove missing data
data = data.dropna()

# Show how much data was removed
print("Data after removing data:")
print(data.isnull().sum())
print(f"Remaining rows: {len(data)}")

# [not in PDF] IQR-based removal of outlires
# Source: https://medium.com/@karthickrajaraja424/write-a-function-to-detect-outliers-in-a-dataset-using-the-iqr-method-6141cb9b0b91
def remove_outliers_iqr(df, column):
    Q1 = np.percentile(df[column], 25)
    Q3 = np.percentile(df[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Use IQR on relavant columns
data = remove_outliers_iqr(data, 'selling_price')
data = remove_outliers_iqr(data, 'km_driven')

print("\nData after IQR based data cleaning:")
print(f"Max. Selling price: {data['selling_price'].max()}")
print(f"Max. kilometer driven: {data['km_driven'].max()}")
print(f"Remaining rows after IQR: {len(data)}")

print(data)


###################################################
# Label-Encoding for Visualisation after cleaning #
###################################################

# Define columns that need to be lable-encoded
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

# Copy data into var label_encoded_data; create empty array lable_encoders
label_encoded_data = data.copy()
label_encoders = {}

# Run lable-encoder for every previosly defined columne (function imported from sklearn)
for col in categorical_columns:
    le = LabelEncoder()
    label_encoded_data[col] = le.fit_transform(label_encoded_data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# [nicht in PDF] Falls Label-Encoding NaNs erzeugt (z. B. durch unerwartete Werte), mit 0 füllen
label_encoded_data = label_encoded_data.fillna(0)

print("\nLabel-Encoded Data:")
print(label_encoded_data.head())

# concatinates x and y into one point to be visualized
all_data_LableEncoded = pd.concat([label_encoded_data], axis=1)

# Drops all columes that are non-numeric to make scaling possible
all_data_LableEncoded = all_data_LableEncoded.select_dtypes(include=['number'])

# Scale data (normalized)
# sscaler = preprocessing.StandardScaler()
# all_data_LableEncoded = sscaler.fit_transform(all_data_LableEncoded)
nscaler = preprocessing.MinMaxScaler()
all_data_LableEncoded = nscaler.fit_transform(all_data_LableEncoded)


#######################################
# Visualisation after cleaining Data #
#######################################

# Reintegrates Column name for boxplot
scaled_df = pd.DataFrame(all_data_LableEncoded, columns=label_encoded_data.select_dtypes(include='number').columns)

# Boxplot with readable x-achsis
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df, orient='v', palette='Set2')
plt.xticks(rotation=45)
plt.title("Boxplot of scaled numerical features")
plt.tight_layout()
plt.show()

# Boxplot for only numerical data
selected_cols = ['selling_price', 'km_driven', 'year']
plt.figure(figsize=(8, 5))
sns.boxplot(data=scaled_df[selected_cols], orient='v', palette='Set3')
plt.title("Boxplot of scaled numerical features")
plt.tight_layout()
plt.show()

# Pairplot to show correlation
sns.pairplot(scaled_df[selected_cols])
plt.suptitle("Pairplot of selected features", y=1.02)
plt.show()


############################################################
# Hybride Kodierung für Modelltraining                    #
# One-Hot: fuel, seller_type, transmission                #
# Label-Encoding: owner, brand                            #
############################################################

# 1. One-Hot-Encoding für ausgewählte Features
onehot_features = ['fuel', 'seller_type', 'transmission', 'brand']
encoded_data = pd.get_dummies(data, columns=onehot_features, drop_first=True)

# 2. Label-Encoding für owner und brand
label_columns = ['owner']
label_encoders = {}

for col in label_columns:
    le = LabelEncoder()
    encoded_data[col] = le.fit_transform(encoded_data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# [nicht in PDF] Fehlende Werte absichern
encoded_data = encoded_data.fillna(0)

print("\nHybrid codierte Daten für das Modelltraining:")
print(encoded_data.head())

# Sortiere nach Jahr und Kilometerstand
encoded_data_sorted = encoded_data.sort_values(by=['year', 'km_driven'], ascending=[False, True])

# Features und Zielspalte definieren
features = encoded_data_sorted.columns.drop(['name', 'selling_price']) if 'name' in encoded_data_sorted.columns else encoded_data_sorted.columns.drop('selling_price')
target = 'selling_price'

X = encoded_data_sorted[features]
y = encoded_data_sorted[target]

# Speichere vollständige Daten
all_data = pd.concat([X, y], axis=1)
# all_data.to_csv('prepared_used_car_data_all.csv', index=False)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Speichern
train_data = pd.concat([X_train, y_train], axis=1)
# train_data.to_csv('prepared_used_car_data_train.csv', index=False)

test_data = pd.concat([X_test, y_test], axis=1)
# test_data.to_csv('prepared_used_car_data_test.csv', index=False)

# print("\nTrainingsdaten gespeichert unter 'prepared_used_car_data_train.csv'")
# print("Testdaten gespeichert unter 'prepared_used_car_data_test.csv'")