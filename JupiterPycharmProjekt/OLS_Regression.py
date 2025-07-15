


# OLS Regression – Start gemäß BostonHousing-PDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # [nicht in PDF]
import pandas as pd  # [nicht in PDF]
from sklearn.model_selection import train_test_split

#
# [nicht in PDF] CSV-Daten für OLS aus vorbereiteten Trainingsdaten laden
data_ols_train = pd.read_csv("prepared_used_car_data_train.csv")  # [nicht in PDF]
data_ols_test = pd.read_csv("prepared_used_car_data_test.csv")

'''
# [nicht in PDF] Standardisierung der Eingabedaten
from sklearn.preprocessing import StandardScaler  # [nicht in PDF]
scaler = StandardScaler()  # [nicht in PDF]
X_ols = data_ols.drop(columns=['selling_price'])  # [nicht in PDF]
X_ols = scaler.fit_transform(X_ols)  # [nicht in PDF]
y_ols = data_ols['selling_price']  # [nicht in PDF]
'''
# [nicht in PDF] Train-Test-Split wie zuvor
X_train_ols = data_ols_train.drop(columns=['selling_price'])  # [nicht in PDF]
y_train_ols = data_ols_train['selling_price']  # [nicht in PDF]
X_test_ols = data_ols_test.drop(columns=['selling_price'])  # [nicht in PDF]
y_test_ols = data_ols_test['selling_price']  # [nicht in PDF]
# [nicht in PDF] Nur numerische Spalten behalten
X_train_ols = X_train_ols.select_dtypes(include=['number'])
X_test_ols = X_test_ols.select_dtypes(include=['number'])

# [nicht in PDF] Standardisierung wie im BostonHousing-PDF (S. 10)

from sklearn.preprocessing import StandardScaler  # [nicht in PDF]
scaler = StandardScaler()  # [nicht in PDF]
X_train_ols = scaler.fit_transform(X_train_ols)  # [nicht in PDF]
X_test_ols = scaler.transform(X_test_ols)  # [nicht in PDF]

# [nicht in PDF] Entferne Spalten mit 0-Varianz im Trainingsset
from sklearn.feature_selection import VarianceThreshold  # [nicht in PDF]
var_filter = VarianceThreshold(threshold=0.0)  # [nicht in PDF]
X_train_ols = var_filter.fit_transform(X_train_ols)  # [nicht in PDF]
X_test_ols = var_filter.transform(X_test_ols)  # [nicht in PDF]

# [nicht in PDF] NaN-/Inf-Prüfung
print("NaN in X_train:", np.isnan(X_train_ols).sum())  # [nicht in PDF]
print("Inf in X_train:", np.isinf(X_train_ols).sum())  # [nicht in PDF]

# Modell trainieren
ols_model = LinearRegression()
ols_model.fit(X_train_ols, y_train_ols)

# Vorhersagen und Bewertung
y_train_pred = ols_model.predict(X_train_ols)
y_test_pred = ols_model.predict(X_test_ols)

r2_train = r2_score(y_train_ols, y_train_pred)
r2_test = r2_score(y_test_ols, y_test_pred)

print("\nOLS Regression Ergebnisse:")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mean_squared_error(y_test_ols, y_test_pred)):.2f}")

# [nicht in PDF] Residuenplot (optional)
import matplotlib.pyplot as plt  # [nicht in PDF]
residuals = y_test_ols - y_test_pred
plt.scatter(y_test_ols, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Tatsächlicher Verkaufspreis")
plt.ylabel("Residuum")
plt.title("Residuenplot: OLS")
plt.tight_layout()
plt.show()