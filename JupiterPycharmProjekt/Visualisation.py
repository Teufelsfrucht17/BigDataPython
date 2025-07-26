import pandas as pd
import DataPrep
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

#######################################
# Visualisation before cleaining Data #
#######################################

# All Data containes NO names but brands
data = pd.read_csv('AllData.csv')

'''
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, orient='v', palette='Set2')
plt.show()
'''

# Visualisation of all data Lable encoded
plt.figure(figsize=(12, 6))
sns.boxplot(data=DataPrep.data_df_LE, orient='v', palette='Set2')
plt.title("Boxplot - All Label Encoded Data")
plt.show()

#DataFocus = DataPrep.data_df_LE[]
# Visualisation of all data Lable encoded
plt.figure(figsize=(12, 6))
selected_cols = ['selling_price', 'km_driven', 'year']
sns.pairplot(data=DataPrep.data_df_LE[selected_cols], palette='Set3')
plt.show()