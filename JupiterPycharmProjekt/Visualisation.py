import pandas as pd
import DataPrep
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

# define all numerical data from orininal file
selected_cols = ['selling_price', 'km_driven', 'year']

#######################################
# Visualisation BEFORE cleaining Data #
#######################################

# Visualisation of all data Lable encoded
plt.figure(figsize=(12, 6))
sns.boxplot(data=DataPrep.data_df_LE_before_IQR, orient='v', palette='Set2')
plt.title("Boxplot - Original - All Label Encoded Data")
plt.show()

# Visualisation of only nomerical values Lable encoded
plt.figure(figsize=(12, 6))
sns.boxplot(data=DataPrep.data_df_LE_before_IQR[selected_cols], orient='v', palette='Set2')
plt.title("Boxplot - Original - Numerical values Label Encoded Data")
plt.show()

# Visualisation of all data Lable encoded
sns.pairplot(data=DataPrep.data_df_LE_before_IQR[selected_cols], palette='Set3')
plt.title("Pairplot - Original - All Label Encoded Data")
plt.show()


#######################################
# Visualisation AFTER cleaining Data #
#######################################

# Visualisation of all data Lable encoded
plt.figure(figsize=(12, 6))
sns.boxplot(data=DataPrep.data_df_LE, orient='v', palette='Set2')
plt.title("Boxplot - Cleaned - All Label Encoded Data")
plt.show()

# Visualisation of only nomerical values Lable encoded
plt.figure(figsize=(12, 6))
sns.boxplot(data=DataPrep.data_df_LE[selected_cols], orient='v', palette='Set2')
plt.title("Boxplot - Cleaned - Numerical values Label Encoded Data")
plt.show()

# Visualisation of all data Lable encoded
sns.pairplot(data=DataPrep.data_df_LE[selected_cols], palette='Set3')
plt.title("Pairplot - Cleaned - All Label Encoded Data")
plt.show()