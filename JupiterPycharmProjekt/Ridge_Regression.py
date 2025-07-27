####################
# Ridge Regression #
####################
import warnings
import numpy as np
warnings.filterwarnings("ignore")
np.seterr(all='ignore')
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import DataPrep

########################################
# Ridge Regression with Label Encoding #
########################################

X_train_rig, X_test_rig, Y_train_rig, Y_test_rig = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

# Define Ridge Regression function and parameters (panalty parameter alpha)
ridge = Ridge()
param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1,0.5, 1, 2, 3, 5, 10, 50, 100, 1000]}

CV_rrmodel = GridSearchCV(estimator=ridge,param_grid=param_grid, cv=10)
with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore', all='ignore'):
    CV_rrmodel.fit(X_train_rig, Y_train_rig)

print("Best parameters set values:", CV_rrmodel.best_params_)

Y_train_pred = CV_rrmodel.predict(X_train_rig)
Y_train_dev = sum((Y_train_rig - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_rig - Y_train_rig.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

# Predict based on test data and compute Pseudo-R²
Y_test_pred = CV_rrmodel.predict(X_test_rig)
Y_test_dev = sum((Y_test_rig - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_rig - Y_test_rig.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev

DataPrep.report.loc[len(DataPrep.report)] = ['Rige RegressionLE', r2, pseudor2,"", CV_rrmodel.cv_results_['mean_test_score'][CV_rrmodel.best_index_], CV_rrmodel.cv_results_['std_test_score'][CV_rrmodel.best_index_]]

###########################################
# Visualise Ridge Regression model result #
###########################################

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


##########################################
# Ridge Regression with One-Hot-Encoding #
##########################################

X_train_rig, X_test_rig, Y_train_rig, Y_test_rig = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

# Define Ridge Regression function and parameters (panalty parameter alpha)
ridge = Ridge()
param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1,0.5, 1, 2, 3, 5, 10, 50, 100, 1000]}

CV_rrmodel = GridSearchCV(estimator=ridge,param_grid=param_grid, cv=10)
with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore', all='ignore'):
    CV_rrmodel.fit(X_train_rig, Y_train_rig)

print("Best parameters set values:", CV_rrmodel.best_params_)

Y_train_pred = CV_rrmodel.predict(X_train_rig)
Y_train_dev = sum((Y_train_rig - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_rig - Y_train_rig.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

# Predict based on test data and compute Pseudo-R²
Y_test_pred = CV_rrmodel.predict(X_test_rig)
Y_test_dev = sum((Y_test_rig - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_rig - Y_test_rig.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev

DataPrep.report.loc[len(DataPrep.report)] = ["Rige RegressionOH ", r2, pseudor2,"", CV_rrmodel.cv_results_['mean_test_score'][CV_rrmodel.best_index_], CV_rrmodel.cv_results_['std_test_score'][CV_rrmodel.best_index_]]

print(DataPrep.report.head())

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
