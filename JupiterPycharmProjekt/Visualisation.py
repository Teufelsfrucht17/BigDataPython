import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Read original Data from CSV


#######################################
# Visualisation before cleaining Data #
#######################################

# All Data containes NO names but brands
data = pd.read_csv('AllData.csv')

# Boxplot with readable x-achsis
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, orient='v', palette='Set2')
plt.xticks(rotation=45)
plt.title("Normed boxplot")
plt.tight_layout()
plt.show()

data = pd.read_csv('Lable_Encoded_Data.csv')

# Boxplot for only numerical data
selected_cols = ['selling_price', 'km_driven', 'year']

plt.figure(figsize=(8, 5))
#sns.boxplot(data=data[selected_cols], orient='v', palette='Set3')
sns.boxplot(data=data, orient='v', palette='Set2')
plt.title("Normed boxplot for numerical data only")
plt.tight_layout()
plt.show()

# Pairplot to show correlation
sns.pairplot(data[selected_cols])
plt.suptitle("Pairplot for select charactaristics", y=1.02)
plt.show()