import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
np.seterr(all='ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import DataPrep


# LE
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
CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=2,n_jobs=-1)
with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore', all='ignore'):
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

#######################################################
# Visualizing Neural Network Hyperparameter Tuning LE #
#######################################################

# Convert CV results to DataFrame
cv_results_df = pd.DataFrame(CV_nnmodel.cv_results_)
best_params = CV_nnmodel.best_params_

# Filter für besten Hidden Layer & Activation
filtered_alpha = cv_results_df[
    (cv_results_df['param_hidden_layer_sizes'] == best_params['hidden_layer_sizes']) &
    (cv_results_df['param_activation'] == best_params['activation']) &
    (cv_results_df['param_solver'] == best_params['solver']) &
    (cv_results_df['param_learning_rate'] == best_params['learning_rate']) &
    (cv_results_df['param_max_iter'] == best_params['max_iter']) &
    (cv_results_df['param_random_state'] == best_params['random_state'])
].copy()

# Konvertiere Alpha von string zu float
filtered_alpha['param_alpha'] = filtered_alpha['param_alpha'].astype(float)
filtered_alpha = filtered_alpha.sort_values(by='param_alpha')

# Plot mit Unsicherheitsbereich
plt.figure(figsize=(12, 6))
plt.plot(filtered_alpha['param_alpha'], filtered_alpha['mean_test_score'], marker='o', label='Mean CV Score')
plt.fill_between(
    filtered_alpha['param_alpha'],
    filtered_alpha['mean_test_score'] - filtered_alpha['std_test_score'],
    filtered_alpha['mean_test_score'] + filtered_alpha['std_test_score'],
    alpha=0.2, color='skyblue', label='±1 Std. Dev'
)
plt.title("CV Score vs Alpha (Neural Network LE)", fontsize=14)
plt.xlabel("Alpha", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Aktivierungsfunktionen mit Unsicherheitsbereich plotten
mask = pd.Series(True, index=cv_results_df.index)
for param, value in best_params.items():
    if f'param_{param}' in cv_results_df.columns and param != 'activation':
        mask &= (cv_results_df[f'param_{param}'] == value)

filtered_act = cv_results_df[mask].copy().sort_values(by='param_activation')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_act['param_activation'], filtered_act['mean_test_score'], marker='o')
plt.fill_between(
    filtered_act['param_activation'],
    filtered_act['mean_test_score'] - filtered_act['std_test_score'],
    filtered_act['mean_test_score'] + filtered_act['std_test_score'],
    alpha=0.2, color='skyblue'
)
plt.title("CV Score vs Activation Function (Neural Network LE)", fontsize=14)
plt.xlabel("Activation Function", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# OH
(X_train_nn, X_test_nn, Y_train_nn, Y_test_nn) = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)

CV_nnmodel = GridSearchCV(estimator=NNetRregCV, param_grid=param_grid, cv=2,n_jobs=-1)
with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore', all='ignore'):
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


#######################################################
# Visualizing Neural Network Hyperparameter Tuning OH #
#######################################################

# Convert CV results to DataFrame
cv_results_df = pd.DataFrame(CV_nnmodel.cv_results_)
best_params = CV_nnmodel.best_params_

# Filter für besten Hidden Layer & Activation
filtered_alpha = cv_results_df[
    (cv_results_df['param_hidden_layer_sizes'] == best_params['hidden_layer_sizes']) &
    (cv_results_df['param_activation'] == best_params['activation']) &
    (cv_results_df['param_solver'] == best_params['solver']) &
    (cv_results_df['param_learning_rate'] == best_params['learning_rate']) &
    (cv_results_df['param_max_iter'] == best_params['max_iter']) &
    (cv_results_df['param_random_state'] == best_params['random_state'])
].copy()

# Konvertiere Alpha von string zu float
filtered_alpha['param_alpha'] = filtered_alpha['param_alpha'].astype(float)
filtered_alpha = filtered_alpha.sort_values(by='param_alpha')

# Plot mit Unsicherheitsbereich
plt.figure(figsize=(12, 6))
plt.plot(filtered_alpha['param_alpha'], filtered_alpha['mean_test_score'], marker='o', label='Mean CV Score')
plt.fill_between(
    filtered_alpha['param_alpha'],
    filtered_alpha['mean_test_score'] - filtered_alpha['std_test_score'],
    filtered_alpha['mean_test_score'] + filtered_alpha['std_test_score'],
    alpha=0.2, color='skyblue', label='±1 Std. Dev'
)
plt.title("CV Score vs Alpha (Neural Network OH)", fontsize=14)
plt.xlabel("Alpha", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Aktivierungsfunktionen mit Unsicherheitsbereich plotten
mask = pd.Series(True, index=cv_results_df.index)
for param, value in best_params.items():
    if f'param_{param}' in cv_results_df.columns and param != 'activation':
        mask &= (cv_results_df[f'param_{param}'] == value)

filtered_act = cv_results_df[mask].copy().sort_values(by='param_activation')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_act['param_activation'], filtered_act['mean_test_score'], marker='o')
plt.fill_between(
    filtered_act['param_activation'],
    filtered_act['mean_test_score'] - filtered_act['std_test_score'],
    filtered_act['mean_test_score'] + filtered_act['std_test_score'],
    alpha=0.2, color='skyblue'
)
plt.title("CV Score vs Activation Function (Neural Network OH)", fontsize=14)
plt.xlabel("Activation Function", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()