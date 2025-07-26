from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from JupiterPycharmProjekt import DataPrep
import warnings
warnings.filterwarnings("ignore")


(X_train_nn, X_test_nn, Y_train_nn, Y_test_nn) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)


param_grid = {
'hidden_layer_sizes': [(5,), (8,), (10,), (13,)],
'alpha': [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.1],
'activation': ['logistic', 'tanh', 'relu'],
'solver': ['sgd', 'adam', 'lbfgs'],
'max_iter': [5000],
'random_state': [0],
'learning_rate': ['constant', 'invscaling', 'adaptive'],
}

NNetRregCV = MLPRegressor()
CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=10,n_jobs=-1)
CV_nnmodel.fit(X_train_nn, Y_train_nn)


#print("Best parameters set values:", CV_rrmodel.best_params_)

Y_train_pred = CV_nnmodel.predict(X_train_nn)
Y_train_dev = sum((Y_train_nn - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_nn - Y_train_nn.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

# Predict based on test data and compute Pseudo-R²
Y_test_pred = CV_nnmodel.predict(X_test_nn)
Y_test_dev = sum((Y_test_nn - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_nn - Y_test_nn.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev


DataPrep.report.loc[len(DataPrep.report)] = ["NN_LE ", r2, pseudor2,"", CV_nnmodel.cv_results_['mean_test_score'][CV_nnmodel.best_index_], CV_nnmodel.cv_results_['std_test_score'][CV_nnmodel.best_index_]]

print(CV_nnmodel.best_params_)

(X_train_nn, X_test_nn, Y_train_nn, Y_test_nn) = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=10,n_jobs=-1)
CV_nnmodel.fit(X_train_nn, Y_train_nn)


#print("Best parameters set values:", CV_rrmodel.best_params_)

Y_train_pred = CV_nnmodel.predict(X_train_nn)
Y_train_dev = sum((Y_train_nn - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_nn - Y_train_nn.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

# Predict based on test data and compute Pseudo-R²
Y_test_pred = CV_nnmodel.predict(X_test_nn)
Y_test_dev = sum((Y_test_nn - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_nn - Y_test_nn.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev


DataPrep.report.loc[len(DataPrep.report)] = ["NN_OH ", r2, pseudor2,"", CV_nnmodel.cv_results_['mean_test_score'][CV_nnmodel.best_index_], CV_nnmodel.cv_results_['std_test_score'][CV_nnmodel.best_index_]]
print(DataPrep.report.head())
print(CV_nnmodel.best_params_)