from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from JupiterPycharmProjekt import DataPrep

###########################################
# Support Vector Regression Lable Encoded #
###########################################

# Braucht lange selbst bei meiner CPU das liegt an der anzahl an möglichen combinationen

(X_train_svr, X_test_svr, Y_train_svr, Y_test_svr) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)

param_grid = {
'kernel': ["linear", "rbf"],
'C': [1, 3, 5, 8, 10],
'epsilon': [0.0, 0.025, 0.05, 0.075, 0.1],
'gamma': [0., 1., 2., 3., 4.]
}

LinSVRreg = SVR() # Wichtig für notizen Jan erklären es gibt mehrer kernals bzw alle werte erklären wichtig
CV_svrmodel = GridSearchCV(estimator=LinSVRreg, param_grid=param_grid, cv=4,n_jobs=-1) # n_jobs verwendet nun alle CPU Kerner erhöht die lesitung cv nicht auf 10 ändern dann dauert das noch länger
CV_svrmodel.fit(X_train_svr, Y_train_svr)

Y_train_pred = CV_svrmodel.predict(X_train_svr)
Y_train_dev = sum((Y_train_svr - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_svr - Y_train_svr.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

Y_test_pred = CV_svrmodel.predict(X_test_svr)
Y_test_dev = sum((Y_test_svr - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_svr - Y_test_svr.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev

DataPrep.report.loc[len(DataPrep.report)] = ["SVRLE ", r2, pseudor2,"", CV_svrmodel.cv_results_['mean_test_score'][CV_svrmodel.best_index_], CV_svrmodel.cv_results_['std_test_score'][CV_svrmodel.best_index_]]
print(CV_svrmodel.best_params_)

#####################################
# Support Vector Regression One Hot #
#####################################

(X_train_svr, X_test_svr, Y_train_svr, Y_test_svr) = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

LinSVRreg = SVR() # Wichtig für notizen Jan erklären es gibt mehrer kernals bzw alle werte erklären wichtig
CV_svrmodel = GridSearchCV(estimator=LinSVRreg, param_grid=param_grid, cv=4,n_jobs=-1) # n_jobs verwendet nun alle CPU Kerner erhöht die lesitung cv nicht auf 10 ändern dann dauert das noch länger
CV_svrmodel.fit(X_train_svr, Y_train_svr)

Y_train_pred = CV_svrmodel.predict(X_train_svr)
Y_train_dev = sum((Y_train_svr - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_svr - Y_train_svr.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

Y_test_pred = CV_svrmodel.predict(X_test_svr)
Y_test_dev = sum((Y_test_svr - Y_test_pred)**2)
Y_train_meandev = sum((Y_test_svr - Y_test_svr.mean())**2)  # [aus PDF]
pseudor2 = 1 - Y_test_dev / Y_train_meandev

DataPrep.report.loc[len(DataPrep.report)] = ["SVROH ", r2, pseudor2,"", CV_svrmodel.cv_results_['mean_test_score'][CV_svrmodel.best_index_], CV_svrmodel.cv_results_['std_test_score'][CV_svrmodel.best_index_]]
print(DataPrep.report.head())
print(CV_svrmodel.best_params_)

