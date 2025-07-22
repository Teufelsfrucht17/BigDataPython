####################
# Ridge Regression #
####################

from sklearn.linear_model import Ridge
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_df = pd.read_csv('DataNormed.csv')
# read training and test data - prepaired in 'DataPreperation'


X = data_df.drop(columns=['selling_price']) # Feature
Y = data_df['selling_price'] # Variable

X_train_rig, X_test_rig, Y_train_rig, Y_test_rig = train_test_split(X, Y, test_size=0.2, random_state=42)

print (X_train_rig)

ridge = Ridge(alpha=0.1)

# Run ridge regression
ridge.fit(X_train_rig, Y_train_rig)


Y_train_pred = ridge.predict(X_train_rig)
Y_train_dev = sum((Y_train_rig - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_rig - Y_train_rig.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]
print("R² on training set =", round(r2, 4))

# Predict based on test data and compute Pseudo-R²
Y_test_pred = ridge.predict(X_test_rig)
Y_test_dev = sum((Y_test_rig - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_rig - Y_test_rig.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev
print("Pseudo-R² on test set =", round(pseudor2, 4))

##########################################
# Visulise Ridge Regression model result #
##########################################

# Residualplot based on Ridge Regression result
residuals = Y_test_rig - Y_test_pred
plt.scatter(Y_test_rig, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Selling price")
plt.ylabel("Residual")
plt.title("Residual plot - Ridge Regression")
plt.tight_layout()
plt.show()

# Scatterplot based on Ridge Regression compairing actual and predicted price
plt.figure(figsize=(8, 5))
plt.scatter(Y_train_rig, Y_train_pred, alpha=0.5)
plt.plot([Y_train_rig.min(), Y_train_rig.max()],
         [Y_train_rig.min(), Y_train_rig.max()],
         color='red', linestyle='--')  # 45° Reference linie
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Prediction accuracy - Ridge Regression")
plt.tight_layout()
plt.show()
