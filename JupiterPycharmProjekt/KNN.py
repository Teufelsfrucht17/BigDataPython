from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from JupiterPycharmProjekt import DataPrep

(X_train_RF, X_test_RF, Y_train_RF, Y_test_RF) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

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
CV_knnmodel.fit(X_train_RF, Y_train_RF)
# Output the best number of neighbors
print("Best parameters found:", CV_knnmodel.best_params_)