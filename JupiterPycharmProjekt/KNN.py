from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from JupiterPycharmProjekt import DataPrep

(X_train_KNN, X_test_KNN, Y_train_KNN, Y_test_KNN) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

# Initialize KNN model
knnmodelCV = KNeighborsRegressor()
# Define grid of neighbor counts
param_grid = {
'n_neighbors': range(3, 22, 2),
'weights': ['uniform', 'distance'],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
'leaf_size': [8, 16, 32, 64, 128, 256, 512],
'p': [2, 3, 4, 5, 6, 7, 8]

}

# Run 10-fold cross-validation across neighbor settings
CV_knnmodel = GridSearchCV(estimator=knnmodelCV, param_grid=param_grid, cv=10,n_jobs=-1)
CV_knnmodel.fit(X_train_KNN, Y_train_KNN)
# Output the best number of neighbors
print("Best parameters found:", CV_knnmodel.best_params_)


Y_train_pred = CV_knnmodel.predict(X_train_KNN)
Y_train_dev = sum((Y_train_KNN - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_KNN - Y_train_KNN.mean())**2)
r2 = 1 - Y_train_dev / Y_train_meandev

# Predict on test set
Y_test_pred = CV_knnmodel.predict(X_test_KNN)
Y_test_dev = sum((Y_test_KNN - Y_test_pred)**2)
Y_test_meandev = sum((Y_test_KNN - Y_test_KNN.mean())**2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev

DataPrep.report.loc[len(DataPrep.report)] = ["KNN_LE ", r2, pseudor2,"", CV_knnmodel.cv_results_['mean_test_score'][CV_knnmodel.best_index_], CV_knnmodel.cv_results_['std_test_score'][CV_knnmodel.best_index_]]


## One

(X_train_KNN, X_test_KNN, Y_train_KNN, Y_test_KNN) = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

# Run 10-fold cross-validation across neighbor settings
CV_knnmodel = GridSearchCV(estimator=knnmodelCV, param_grid=param_grid, cv=10,n_jobs=-1)
CV_knnmodel.fit(X_train_KNN, Y_train_KNN)
# Output the best number of neighbors
print("Best parameters found:", CV_knnmodel.best_params_)


Y_train_pred = CV_knnmodel.predict(X_train_KNN)
Y_train_dev = sum((Y_train_KNN - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_KNN - Y_train_KNN.mean())**2)
r2 = 1 - Y_train_dev / Y_train_meandev

# Predict on test set
Y_test_pred = CV_knnmodel.predict(X_test_KNN)
Y_test_dev = sum((Y_test_KNN - Y_test_pred)**2)
Y_test_meandev = sum((Y_test_KNN - Y_test_KNN.mean())**2)
pseudor2 = 1 - Y_test_dev / Y_test_meandev

DataPrep.report.loc[len(DataPrep.report)] = ["KNN_OH ", r2, pseudor2,"", CV_knnmodel.cv_results_['mean_test_score'][CV_knnmodel.best_index_], CV_knnmodel.cv_results_['std_test_score'][CV_knnmodel.best_index_]]
print(DataPrep.report.head())
print(CV_knnmodel.best_params_)