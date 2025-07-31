from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataPrep


param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [10, 50, 100, 150, 200],
    'criterion': ['squared_error', 'absolute_error']
}

(X_train_RF, X_test_RF, Y_train_RF, Y_test_RF) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

RForregCV = RandomForestRegressor(random_state=42)

CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=4, n_jobs=-1)
CV_rfmodel.fit(X_train_RF, Y_train_RF)

Y_train_pred = CV_rfmodel.predict(X_train_RF)
Y_train_dev = sum((Y_train_RF - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_RF - Y_train_RF.mean())**2)
r2 = 1 - Y_train_dev / Y_train_meandev

# Predict on test set
Y_test_pred = CV_rfmodel.predict(X_test_RF)
Y_test_dev = sum((Y_test_RF - Y_test_pred)**2)
Y_test_meandev = sum((Y_test_RF - Y_test_RF.mean())**2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev


# Show best parameters
DataPrep.report.loc[len(DataPrep.report)] = ["RF_LC ", r2, pseudor2,"", CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_]]

print(CV_rfmodel.best_params_)



plt.figure(figsize=(8, 6))

for criterion in param_grid['criterion']:
    RForregCV = RandomForestRegressor(random_state=42, criterion=criterion)
    CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid={k: param_grid[k] for k in ['max_depth', 'n_estimators']}, cv=4, n_jobs=-1)
    CV_rfmodel.fit(X_train_RF, Y_train_RF)
    results_df = pd.DataFrame(CV_rfmodel.cv_results_)

    for depth in param_grid['max_depth']:
        subset = results_df[results_df['param_max_depth'] == depth]
        linestyle = '-' if criterion == 'squared_error' else '--'
        plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o',
                 linestyle=linestyle, label=f"{criterion}, max_depth: {depth}")

plt.xlabel('n_estimators')
plt.ylabel('CV Average Validation Accuracy')
plt.title('Grid Search Scores - Label Encoded')
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()





##Onehot

(X_train_RF, X_test_RF, Y_train_RF, Y_test_RF) = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

RForregCV = RandomForestRegressor(random_state=42)

CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=4, n_jobs=-1)
CV_rfmodel.fit(X_train_RF, Y_train_RF)

Y_train_pred = CV_rfmodel.predict(X_train_RF)
Y_train_dev = sum((Y_train_RF - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_RF - Y_train_RF.mean())**2)
r2 = 1 - Y_train_dev / Y_train_meandev

# Predict on test set
Y_test_pred = CV_rfmodel.predict(X_test_RF)
Y_test_dev = sum((Y_test_RF - Y_test_pred)**2)
Y_test_meandev = sum((Y_test_RF - Y_test_RF.mean())**2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev


# Show best parameters
DataPrep.report.loc[len(DataPrep.report)] = ["RF_OH ", r2, pseudor2,"", CV_rfmodel.cv_results_['mean_test_score'][CV_rfmodel.best_index_], CV_rfmodel.cv_results_['std_test_score'][CV_rfmodel.best_index_]]

print(CV_rfmodel.best_params_)
print(DataPrep.report.head())

plt.figure(figsize=(8, 6))

for criterion in param_grid['criterion']:
    RForregCV = RandomForestRegressor(random_state=42, criterion=criterion)
    CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid={k: param_grid[k] for k in ['max_depth', 'n_estimators']}, cv=4, n_jobs=-1)
    CV_rfmodel.fit(X_train_RF, Y_train_RF)
    results_df = pd.DataFrame(CV_rfmodel.cv_results_)

    for depth in param_grid['max_depth']:
        subset = results_df[results_df['param_max_depth'] == depth]
        linestyle = '-' if criterion == 'squared_error' else '--'
        plt.plot(subset['param_n_estimators'], subset['mean_test_score'], marker='o',
                 linestyle=linestyle, label=f"{criterion}, max_depth: {depth}")

plt.xlabel('n_estimators')
plt.ylabel('CV Average Validation Accuracy')
plt.title('Grid Search Scores - One-Hot Encoded')
plt.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
