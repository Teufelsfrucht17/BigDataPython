import pandas as pd
from matplotlib import pyplot as plt
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


###########################################################
# Visualisation SVR OH according to Bostan Housing PDF LE #
###########################################################

# Visualisation CV Score vs Gamma

# Extract full CV results as DataFrame
cv_results_df = pd.DataFrame(CV_svrmodel.cv_results_)
# Filter: only rows with best C, epsilon, and kernel
best_params = CV_svrmodel.best_params_
filtered = cv_results_df[
(cv_results_df['param_C'] == best_params['C']) &
(cv_results_df['param_epsilon'] == best_params['epsilon']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
]
# Sort by gamma for plotting
filtered = filtered.sort_values(by='param_gamma')
# Plot mean test score vs gamma
plt.figure(figsize=(12, 6))
plt.plot(filtered['param_gamma'], filtered['mean_test_score'], marker='o')
plt.fill_between(filtered['param_gamma'],
filtered['mean_test_score'] - filtered['std_test_score'],
filtered['mean_test_score'] + filtered['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs Gamma (RBF Kernel) LE", fontsize=14)
plt.xlabel("Gamma", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered['param_gamma'])
plt.tight_layout()
plt.show()

# Visualisation CV Score vs C

# Filter for best gamma, epsilon, and kernel
filtered_C = cv_results_df[
(cv_results_df['param_gamma'] == best_params['gamma']) &
(cv_results_df['param_epsilon'] == best_params['epsilon']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
].sort_values(by='param_C')
# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_C['param_C'], filtered_C['mean_test_score'], marker='o')
plt.fill_between(filtered_C['param_C'],
filtered_C['mean_test_score'] - filtered_C['std_test_score'],
filtered_C['mean_test_score'] + filtered_C['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs C (RBF Kernel) LE", fontsize=14)
plt.xlabel("C", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered_C['param_C'])
plt.tight_layout()
plt.show()

# Visualisation CV Score vs Epsilon

# Filter for best gamma, C, and kernel
filtered_eps = cv_results_df[
(cv_results_df['param_gamma'] == best_params['gamma']) &
(cv_results_df['param_C'] == best_params['C']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
].sort_values(by='param_epsilon')
# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_eps['param_epsilon'], filtered_eps['mean_test_score'], marker='o')
plt.fill_between(filtered_eps['param_epsilon'],
filtered_eps['mean_test_score'] - filtered_eps['std_test_score'],
filtered_eps['mean_test_score'] + filtered_eps['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs Epsilon (RBF Kernel) LE", fontsize=14)
plt.xlabel("Epsilon", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered_eps['param_epsilon'])
plt.tight_layout()
plt.show()


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

###########################################################
# Visualisation SVR OH according to Bostan Housing PDF OH #
###########################################################

# Visualisation CV Score vs Gamma

# Extract full CV results as DataFrame
cv_results_df = pd.DataFrame(CV_svrmodel.cv_results_)
# Filter: only rows with best C, epsilon, and kernel
best_params = CV_svrmodel.best_params_
filtered = cv_results_df[
(cv_results_df['param_C'] == best_params['C']) &
(cv_results_df['param_epsilon'] == best_params['epsilon']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
]
# Sort by gamma for plotting
filtered = filtered.sort_values(by='param_gamma')
# Plot mean test score vs gamma
plt.figure(figsize=(12, 6))
plt.plot(filtered['param_gamma'], filtered['mean_test_score'], marker='o')
plt.fill_between(filtered['param_gamma'],
filtered['mean_test_score'] - filtered['std_test_score'],
filtered['mean_test_score'] + filtered['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs Gamma (RBF Kernel) OH", fontsize=14)
plt.xlabel("Gamma", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered['param_gamma'])
plt.tight_layout()
plt.show()

# Visualisation CV Score vs C

# Filter for best gamma, epsilon, and kernel
filtered_C = cv_results_df[
(cv_results_df['param_gamma'] == best_params['gamma']) &
(cv_results_df['param_epsilon'] == best_params['epsilon']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
].sort_values(by='param_C')
# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_C['param_C'], filtered_C['mean_test_score'], marker='o')
plt.fill_between(filtered_C['param_C'],
filtered_C['mean_test_score'] - filtered_C['std_test_score'],
filtered_C['mean_test_score'] + filtered_C['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs C (RBF Kernel) OH", fontsize=14)
plt.xlabel("C", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered_C['param_C'])
plt.tight_layout()
plt.show()

# Visualisation CV Score vs Epsilon

# Filter for best gamma, C, and kernel
filtered_eps = cv_results_df[
(cv_results_df['param_gamma'] == best_params['gamma']) &
(cv_results_df['param_C'] == best_params['C']) &
(cv_results_df['param_kernel'] == best_params['kernel'])
].sort_values(by='param_epsilon')
# Plot
plt.figure(figsize=(12, 6))
plt.plot(filtered_eps['param_epsilon'], filtered_eps['mean_test_score'], marker='o')
plt.fill_between(filtered_eps['param_epsilon'],
filtered_eps['mean_test_score'] - filtered_eps['std_test_score'],
filtered_eps['mean_test_score'] + filtered_eps['std_test_score'],
alpha=0.2)
plt.title("SVR Cross-Validation Score vs Epsilon (RBF Kernel) OH", fontsize=14)
plt.xlabel("Epsilon", fontsize=12)
plt.ylabel("Mean CV Score", fontsize=12)
plt.grid(True)
plt.xticks(filtered_eps['param_epsilon'])
plt.tight_layout()
plt.show()