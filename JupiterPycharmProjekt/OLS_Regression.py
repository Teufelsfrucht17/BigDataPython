


# OLS Regression – Start gemäß BostonHousing-PDF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # [nicht in PDF]
import pandas as pd  # [nicht in PDF]
from sklearn.model_selection import train_test_split

#
# [nicht in PDF] CSV-Daten für OLS aus vorbereiteten Trainingsdaten laden
data_ols = pd.read_csv("prepared_used_car_data_train.csv")  # [nicht in PDF]

# [nicht in PDF] Standardisierung der Eingabedaten
from sklearn.preprocessing import StandardScaler  # [nicht in PDF]
scaler = StandardScaler()  # [nicht in PDF]
X_ols = data_ols.drop(columns=['selling_price'])  # [nicht in PDF]
X_ols = scaler.fit_transform(X_ols)  # [nicht in PDF]
y_ols = data_ols['selling_price']  # [nicht in PDF]

# [nicht in PDF] Train-Test-Split wie zuvor
X_train_ols, X_test_ols, y_train_ols, y_test_ols = train_test_split(X_ols, y_ols, test_size=0.2, random_state=42)

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