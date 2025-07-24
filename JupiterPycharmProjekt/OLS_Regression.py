##################
# OLS Regression #
##################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import DataPrep

########################################
# Run OLS Model with Lable Encoded Data#
########################################

X_train_OLS, X_test_OLS, Y_train_OLS, Y_test_OLS = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

# Set model specs
ols_model = LinearRegression()
ols_model.fit(X_train_OLS, Y_train_OLS)

# Prediction and result
y_train_pred = ols_model.predict(X_train_OLS)
y_test_pred = ols_model.predict(X_test_OLS)

r2_train = r2_score(Y_train_OLS, y_train_pred)
r2_test = r2_score(Y_test_OLS, y_test_pred)


DataPrep.report.loc[len(DataPrep.report)] = ['OLS RegressionLC', r2_train, r2_test,np.sqrt(mean_squared_error(Y_test_OLS, y_test_pred)), "", ""]

print(DataPrep.report.head())

#############################
# Visulise OLS model result #
#############################

# Residual plot based on OLS result
residuals = Y_test_OLS - y_test_pred
plt.scatter(Y_test_OLS, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Selling price")
plt.ylabel("Residual")
plt.title("Residual plot - OLS")
plt.tight_layout()
plt.show()

##########################################
# Run OLS Model with One-Hot-Encoded Data#
##########################################

X_train_OLS, X_test_OLS, Y_train_OLS, Y_test_OLS = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)
# Set model specs
ols_model = LinearRegression()
ols_model.fit(X_train_OLS, Y_train_OLS)

# Prediction and result
y_train_pred = ols_model.predict(X_train_OLS)
y_test_pred = ols_model.predict(X_test_OLS)

r2_train = r2_score(Y_train_OLS, y_train_pred)
r2_test = r2_score(Y_test_OLS, y_test_pred)

print("\nOLS Regression result with one-hot-encoding:")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mean_squared_error(Y_test_OLS, y_test_pred)):.2f}")

