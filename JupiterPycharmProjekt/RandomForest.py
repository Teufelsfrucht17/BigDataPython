from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import matplotlib.pyplot as plt

from JupiterPycharmProjekt import DataPrep


### Function to visualize the results of Grid Search with 2 hyperparameters ###
def plot_grid_search_2d(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_1),len(grid_param_2))
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line
    for idx, val in enumerate(grid_param_1):
        ax.plot(grid_param_2, scores_mean[idx,:], '-o', label= name_param_1 + ': ')
    ax.set_title("Grid Search Scores", fontsize=12, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=10)
    ax.set_ylabel('CV Average Validation Accuracy', fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid('on')
    return print("ja")

def plot_grid_search_2d(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Filter auf festen Wert für dritten Parameter (z. B. 'criterion')
    if 'param_criterion' in cv_results:
        cv_results = cv_results[cv_results['param_criterion'] == 'squared_error']

    # Get Test Scores Mean and reshape
    scores_mean = np.array(cv_results['mean_test_score']).reshape(len(grid_param_1), len(grid_param_2))

    # Plot
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, val in enumerate(grid_param_1):
        ax.plot(grid_param_2, scores_mean[idx, :], '-o', label=f"{name_param_1}: {val}")
    ax.set_title("Grid Search Scores", fontsize=12, fontweight='bold')
    ax.set_xlabel(name_param_2, fontsize=10)
    ax.set_ylabel('CV Average Validation Score', fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return print("ja")


param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [10, 50, 100, 150, 200],
    'criterion': ['squared_error', 'absolute_error']  # [Fix] gültige Kriterien für RandomForestRegressor
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


# Unpack parameter names and values
plot_model = CV_rfmodel.cv_results_
param_name1 = list(param_grid.keys())[0] # 'max_depth'
param_name2 = list(param_grid.keys())[1] # 'n_estimators'
param1_values = param_grid[param_name1]
param2_values = param_grid[param_name2]
# Call plot function (assuming it's defined elsewhere)
plot_grid_search_2d(plot_model, param1_values, param2_values, param_name1, param_name2)





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
