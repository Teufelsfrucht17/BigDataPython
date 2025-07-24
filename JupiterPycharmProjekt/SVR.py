from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from JupiterPycharmProjekt import DataPrep

(X_train_svr, X_test_svr, Y_train_svr, Y_test_svr) = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)


LinSVRreg = SVR(kernel='linear', C=1.0, epsilon=0.1)

LinSVRreg.fit(X_train_svr, Y_train_svr)
Y_train_pred = LinSVRreg.predict(X_train_svr)
Y_train_dev = sum((Y_train_svr - Y_train_pred)**2)
Y_train_meandev = sum((Y_train_svr - Y_train_svr.mean())**2)  # [aus PDF]
r2 = 1 - Y_train_dev / Y_train_meandev  # [aus PDF]

print(r2)