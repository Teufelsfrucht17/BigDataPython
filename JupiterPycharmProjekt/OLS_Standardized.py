from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import DataPrep


#################
# Lable Encoded #
#################

X_train_LE, X_test_LE, Y_train_LE, Y_test_LE = train_test_split(DataPrep.X_LE, DataPrep.Y_LE, test_size=0.2, random_state=42)
print(Y_train_LE.shape)
print(Y_test_LE.shape)

lm = LinearRegression()
lm.fit(X_train_LE, Y_train_LE)

Y_train_pred_LE = lm.predict(X_train_LE)

R2_LE = r2_score(Y_train_LE, Y_train_pred_LE)
# Y_train_dev = sum((Y_train - Y_train_pred) ** 2)
# R2 = 1 - (Y_train_dev / Y_train_meandev)
print("R2 LE: ")
print(R2_LE)

Y_test_pred_LE = lm.predict(X_test_LE)
PseudoR2_LE = r2_score(Y_test_LE, Y_test_pred_LE)

# Y_test_dev = sum((Y_test - Y_test_pred) ** 2)
# PseudoR2 = 1 - (Y_test_dev / Y_train_meandev)
print("PseudoR2 LE: ")
print(PseudoR2_LE)

###########
# One-Hot #
###########

X_train_OH, X_test_OH, Y_train_OH, Y_test_OH = train_test_split(DataPrep.X_OH, DataPrep.Y_OH, test_size=0.2, random_state=42)
print(Y_train_OH.shape)
print(Y_test_OH.shape)

lm = LinearRegression()
lm.fit(X_train_OH, Y_train_OH)

Y_train_pred_OH = lm.predict(X_train_OH)

R2_OH = r2_score(Y_train_OH, Y_train_pred_OH)
# Y_train_dev = sum((Y_train - Y_train_pred) ** 2)
# R2 = 1 - (Y_train_dev / Y_train_meandev)
print("R2 OH: ")
print(R2_OH)

Y_test_pred_OH = lm.predict(X_test_OH)
PseudoR2_OH = r2_score(Y_test_OH, Y_test_pred_OH)

# Y_test_dev = sum((Y_test - Y_test_pred) ** 2)
# PseudoR2 = 1 - (Y_test_dev / Y_train_meandev)
print("PseudoR2 OH: ")
print(PseudoR2_OH)

