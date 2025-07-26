from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from JupiterPycharmProjekt import DataPrep

param_grid = {
'max_depth': [4, 5, 6, 7, 8],
'n_estimators': [10, 50, 100, 150, 200],
'criterion': ['gini', 'entropy'],
'random_state': [0]
}

(X_train_RF, X_test_RF, Y_train_RF, Y_test_RF) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

RForregCV = RandomForestRegressor()

CV_rfmodel = GridSearchCV(estimator=RForregCV, param_grid=param_grid, cv=10)
CV_rfmodel.fit(X_train_RF, Y_train_RF)
# Show best parameters
print("Best parameters found:", CV_rfmodel.best_params_)