

import DataPrep

# Run all models
import JupiterPycharmProjekt.OLS_Regression
import JupiterPycharmProjekt.Ridge_Regression
import JupiterPycharmProjekt.SVR
import JupiterPycharmProjekt.NeuralNetworks
import JupiterPycharmProjekt.RandomForest
import JupiterPycharmProjekt.KNN


# Optional: include visualisation step if needed
# import JupiterPycharmProjekt.Visualisation

# Print final report

print(DataPrep.report)
DataPrep.report.to_csv("Modell_Report.csv", index=False)