from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from JupiterPycharmProjekt import DataPrep

(X_train_svr, X_test_svr, Y_train_svr, Y_test_svr) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)


param_grid = {
'kernel': ["linear", "rbf"],
'C': [1, 3, 5, 8, 10],
'epsilon': [0.0, 0.025, 0.05, 0.075, 0.1],
'gamma': [0., 1., 2., 3., 4.]
}

LinSVRreg = SVR() # Wichtig für notizen Jan erklären es gibt mehrer kernals bzw alle werte erklären wichtig
CV_svrmodel = GridSearchCV(estimator=LinSVRreg, param_grid=param_grid, cv=4,n_jobs=-1) # n_jobs verwendet nun alle CPU Kerner erhöht die lesitung
CV_svrmodel.fit(X_train_svr, Y_train_svr)



Y_train_pred = CV_svrmodel.predict(X_train_svr)
Y_train_dev = sum((Y_train_svr - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_svr - Y_train_svr.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

print(r2)


Y_test_pred = CV_svrmodel.predict(X_test_svr)
Y_test_dev = sum((Y_test_svr - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_svr - Y_test_svr.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev
print(pseudor2)

print(CV_svrmodel.best_params_)