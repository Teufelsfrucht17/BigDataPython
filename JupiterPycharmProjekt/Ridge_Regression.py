from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt



data_rig_train = pd.read_csv("prepared_used_car_data_train.csv")  # [nicht in PDF]
data_rig_test = pd.read_csv("prepared_used_car_data_test.csv")


X_train_rig = data_rig_train.drop(columns=['selling_price'])  # [nicht in PDF]
y_train_rig = data_rig_train['selling_price']  # [nicht in PDF]
X_test_rig = data_rig_test.drop(columns=['selling_price'])  # [nicht in PDF]
y_test_rig = data_rig_test['selling_price']  # [nicht in PDF]
# [nicht in PDF] Nur numerische Spalten behalten
X_train_rig = X_train_rig.select_dtypes(include=['number'])
X_test_rig = X_test_rig.select_dtypes(include=['number'])

rige = Ridge(alpha=0.5)

rige.fit(X_train_rig, y_train_rig)
Y_train_pred = rige.predict(X_train_rig)
Y_train_dev = sum((y_train_rig - Y_train_pred)**2)
Y_train_meandev = sum((y_train_rig - y_train_rig.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]
print("R² on training set =", round(r2, 4))

# Predict on test data and compute Pseudo-R²
Y_test_pred = rige.predict(X_test_rig)
Y_test_dev = sum((y_test_rig - Y_test_pred)**2)
Y_train_meandev = sum((y_test_rig - y_test_rig.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev
print("Pseudo-R² on test set =", round(pseudor2, 4))




# [nicht in PDF] Scatterplot: Tatsächlicher vs. Vorhergesagter Preis
plt.figure(figsize=(8, 5))
plt.scatter(y_train_rig, Y_train_pred, alpha=0.5)
plt.plot([y_train_rig.min(), y_train_rig.max()],
         [y_train_rig.min(), y_train_rig.max()],
         color='red', linestyle='--')  # 45° Referenzlinie
plt.xlabel("Tatsächlicher Verkaufspreis")
plt.ylabel("Vorhergesagter Verkaufspreis")
plt.title("Vorhersagegenauigkeit – Ridge Regression (Trainingsdaten)")
plt.tight_layout()
plt.show()

