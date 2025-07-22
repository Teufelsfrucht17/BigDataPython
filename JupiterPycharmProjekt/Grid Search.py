import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# read training and test data - prepaired in 'DataPreperation'
data_rig_train = pd.read_csv("prepared_used_car_data_train.csv")
data_rig_test = pd.read_csv("prepared_used_car_data_test.csv")

# Split train and test data into x and y for regression
y_train_rig = data_rig_train['selling_price']  # [nicht in PDF]

y_train_rig = data_rig_train['selling_price']  # [nicht in PDF]

X_train_rig = data_rig_train.drop(columns=["selling_price"])  # [nicht in PDF]
y_test_rig = data_rig_test['selling_price']  # [nicht in PDF]
X_test_rig = data_rig_test.drop(columns=["selling_price"])  # [nicht in PDF]

# only keep numerical columnes
#X_train_rig = X_train_rig.select_dtypes(include=['number']) # [nicht in PDF]
#X_test_rig = X_test_rig.select_dtypes(include=['number']) # [nicht in PDF]

# [nicht in PDF] Konvertiere alle Spalten auf float zur Vermeidung von bool-Arithmetikfehlern
X_train_rig = X_train_rig.astype(float)
X_test_rig = X_test_rig.astype(float)


# normalisierung auf 0/1
X_train_rig = (X_train_rig - X_train_rig.min()) / (X_train_rig.max() - X_train_rig.min())
X_test_rig = (X_test_rig - X_test_rig.min()) / (X_test_rig.max() - X_test_rig.min())

# [nicht in PDF] Fehlende Werte durch 0 ersetzen (nach Skalierung)
X_train_rig = X_train_rig.fillna(0)
X_test_rig = X_test_rig.fillna(0)

print (X_train_rig)



ridgeCV = Ridge()
# parameter für Grid festlegen

param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1,0.5, 1, 2, 3, 5, 10, 50, 100, 1000]}

CV_rrmodel = GridSearchCV(estimator=ridgeCV,param_grid=param_grid, cv=10)

CV_rrmodel.fit(X_train_rig, y_train_rig)

print("Best parameters set values:", CV_rrmodel.best_params_)