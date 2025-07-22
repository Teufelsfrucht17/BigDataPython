##################
# OLS Regression #
##################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# read training and test data - prepaired in 'DataPreperation'
data_ols_train = pd.read_csv("prepared_used_car_data_all.csv")
data_ols_test = pd.read_csv("prepared_used_car_data_test.csv")

# Split train and test data into x and y for regression
X_OLS_train = data_ols_train[['year', 'km_driven']]  # [nicht in PDF]
Y_OLS_train = data_ols_train['selling_price']  # [nicht in PDF]

X_OLS_test= data_ols_test[['year', 'km_driven']]  # [nicht in PDF]
Y_OLS_test = data_ols_test['selling_price']  # [nicht in PDF]

# only keep numerical columnes
X_OLS_train = X_OLS_train.select_dtypes(include=['number']) # [nicht in PDF]
X_OLS_test= (X_OLS_test.select_dtypes(include=['number'])) # [nicht in PDF]



# normalisierung auf 0/1
X_OLS_train = (X_OLS_train - X_OLS_train.min()) / (X_OLS_train.max() - X_OLS_train.min())
X_OLS_test= (X_OLS_test- X_OLS_test.min()) / (X_OLS_test.max() - X_OLS_test.min())

print (X_OLS_train)

#################
# Run OLS Model #
#################

# Set model specs
ols_model = LinearRegression()
ols_model.fit(X_OLS_train, Y_OLS_train)

# Prediction and result
y_train_pred = ols_model.predict(X_OLS_train)
y_test_pred = ols_model.predict(X_OLS_test)

r2_train = r2_score(Y_OLS_train, y_train_pred)
r2_test = r2_score(Y_OLS_test, y_test_pred)

print("\nOLS Regression result:")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mean_squared_error(Y_OLS_test, y_test_pred)):.2f}")


#############################
# Visulise OLS model result #
#############################

# Residual plot based on OLS result
residuals = Y_OLS_test - y_test_pred
plt.scatter(Y_OLS_test, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Selling price")
plt.ylabel("Residual")
plt.title("Residual plot - OLS")
plt.tight_layout()
plt.show()