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
data_ols_train = pd.read_csv("prepared_used_car_data_train.csv")
data_ols_test = pd.read_csv("prepared_used_car_data_test.csv")

'''
# [nicht in PDF] Standardisierung der Eingabedaten
from sklearn.preprocessing import StandardScaler  # [nicht in PDF]
scaler = StandardScaler()  # [nicht in PDF]
X_ols = data_ols.drop(columns=['selling_price'])  # [nicht in PDF]
X_ols = scaler.fit_transform(X_ols)  # [nicht in PDF]
y_ols = data_ols['selling_price']  # [nicht in PDF]
'''

# Split train and test data into x and y for regression
X_train_ols = data_ols_train.drop(columns=['selling_price'])  # [nicht in PDF]
y_train_ols = data_ols_train['selling_price']  # [nicht in PDF]
X_test_ols = data_ols_test.drop(columns=['selling_price'])  # [nicht in PDF]
y_test_ols = data_ols_test['selling_price']  # [nicht in PDF]

# only keep numerical columnes
X_train_ols = X_train_ols.select_dtypes(include=['number']) # [nicht in PDF]
X_test_ols = X_test_ols.select_dtypes(include=['number']) # [nicht in PDF]

# [nicht in PDF] Standardisierung wie im BostonHousing-PDF (S. 10)
scaler = StandardScaler()  # [nicht in PDF]
X_train_ols = scaler.fit_transform(X_train_ols)  # [nicht in PDF]
X_test_ols = scaler.transform(X_test_ols)  # [nicht in PDF]

'''
# [nicht in PDF] Entferne Spalten mit 0-Varianz im Trainingsset
Sollten wir eh nicht haben - 0 varianz impelziert das es eine Zeihle gibt mit immer den gleichen wärten die damit nichts zu dem modell beiträgt. Haben wir nicht gemacht, entsprechend lassen wir das erstmal raus und gucken uns sonnst online nochmal die quellen dazu an.
from sklearn.feature_selection import VarianceThreshold  # [nicht in PDF]
var_filter = VarianceThreshold(threshold=0.0)  # [nicht in PDF]
X_train_ols = var_filter.fit_transform(X_train_ols)  # [nicht in PDF]
X_test_ols = var_filter.transform(X_test_ols)  # [nicht in PDF]
'''

'''
# [nicht in PDF] NaN-/Inf-Prüfung (not a number & infinite)
Checen ob nicht numerische Zahlen vorhanden sind oder Zahlen die unendlich sind - beides funktioniert nicht mit OLS Regression 
print("NaN in X_train:", np.isnan(X_train_ols).sum())  # [nicht in PDF]
print("Inf in X_train:", np.isinf(X_train_ols).sum())  # [nicht in PDF]
'''


#################
# Run OLS Model #
#################

# Set model specs
ols_model = LinearRegression()
ols_model.fit(X_train_ols, y_train_ols)

# Prediction and result
y_train_pred = ols_model.predict(X_train_ols)
y_test_pred = ols_model.predict(X_test_ols)

r2_train = r2_score(y_train_ols, y_train_pred)
r2_test = r2_score(y_test_ols, y_test_pred)

print("\nOLS Regression result:")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mean_squared_error(y_test_ols, y_test_pred)):.2f}")


#############################
# Visulise OLS model result #
#############################

# Residual plot based on OLS result
residuals = y_test_ols - y_test_pred
plt.scatter(y_test_ols, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Selling price")
plt.ylabel("Residual")
plt.title("Residual plot - OLS")
plt.tight_layout()
plt.show()