import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Read original Data from CSV 
data = pd.read_csv('UsedCarSellingPrices.csv')


####################################################
# Label-Encoding for Visualisation before cleaning #
####################################################

# Define columns that need to be lable-encoded
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']

# Copy data into var label_encoded_data; create empty array lable_encoders
label_encoded_data = data.copy()
label_encoders = {}

# Run lable-encoder for every previosly defined columne (function imported from sklearn)
for col in categorical_columns:
    le = LabelEncoder()
    label_encoded_data[col] = le.fit_transform(label_encoded_data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\nLabel-Encoded Data for Visualisation:")
print(label_encoded_data.head())

#
numeric_data_LableEncoded = all_data_LableEncoded = pd.concat([label_encoded_data], axis=1)
# Nicht-numerische Spalten entfernen (z. B. 'name'), um Skalierung zu ermöglichen
all_data_LableEncoded = all_data_LableEncoded.select_dtypes(include=['number'])
sscaler = preprocessing.StandardScaler()
all_data_LableEncoded = sscaler.fit_transform(all_data_LableEncoded)
nscaler = preprocessing.MinMaxScaler()
all_data_LableEncoded = nscaler.fit_transform(all_data_LableEncoded)


#######################################
# Visualisation before cleaining Data #
#######################################

# Zusätzliche Visualisierungen nach Skalierung
sns.boxplot(data=all_data_LableEncoded, orient='v', palette='Set2')
plt.show()

# Skalierte Daten wieder in DataFrame mit Spaltennamen überführen
scaled_df = pd.DataFrame(all_data_LableEncoded, columns=label_encoded_data.select_dtypes(include='number').columns)

# 1. Boxplot mit lesbaren Achsenbeschriftungen
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df, orient='v', palette='Set2')
plt.xticks(rotation=45)
plt.title("Boxplot der skalierten numerischen Features")
plt.tight_layout()
plt.show()

# 2. Boxplot nur für ausgewählte numerische Spalten
selected_cols = ['selling_price', 'km_driven', 'year']
plt.figure(figsize=(8, 5))
sns.boxplot(data=scaled_df[selected_cols], orient='v', palette='Set3')
plt.title("Boxplot ausgewählter Merkmale")
plt.tight_layout()
plt.show()

# 3. Pairplot für Verteilungen und Korrelationen
sns.pairplot(scaled_df[selected_cols])
plt.suptitle("Paarweise Verteilungen ausgewählter Merkmale", y=1.02)
plt.show()
#print("Trainingsdaten zusätzlich als 'prepared_used_car_data_train.parquet' gespeichert.")
#print("Testdaten zusätzlich als 'prepared_used_car_data_test.parquet' gespeichert.")
#print("Gesamtdaten zusätzlich als 'prepared_used_car_data.parquet' gespeichert.")


######################################
# Clean Data & Create variable Brand #
######################################

# Fehlende Werte entfernen
data = data.dropna()
print("Daten nach Entfernen fehlender Werte:")
print(data.isnull().sum())
print(f"Verbleibende Zeilen: {len(data)}")

# Ausreißerentfernung mit IQR-Methode
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Auf die wichtigsten Spalten anwenden
data = remove_outliers_iqr(data, 'selling_price')
data = remove_outliers_iqr(data, 'km_driven')

print("\nDaten nach IQR-basierter Ausreißerbereinigung:")
print(f"Max. Verkaufspreis: {data['selling_price'].max()}")
print(f"Max. Kilometerstand: {data['km_driven'].max()}")
print(f"Verbleibende Zeilen nach IQR-Filter: {len(data)}")

# Create 'Brand' as new columne
data['brand'] = data['name'].str.split().str[0]
print(data)


###################################################
# Label-Encoding for Visualisation after cleaning #
###################################################

categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
label_encoded_data = data.copy()
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    label_encoded_data[col] = le.fit_transform(label_encoded_data[col])
    label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print("\nLabel-Encoded Data (nur zur Referenz):")
print(label_encoded_data.head())

numeric_data_LableEncoded = all_data_LableEncoded = pd.concat([label_encoded_data], axis=1)
# Nicht-numerische Spalten entfernen (z. B. 'name'), um Skalierung zu ermöglichen
all_data_LableEncoded = all_data_LableEncoded.select_dtypes(include=['number'])
sscaler = preprocessing.StandardScaler()
all_data_LableEncoded = sscaler.fit_transform(all_data_LableEncoded)
nscaler = preprocessing.MinMaxScaler()
all_data_LableEncoded = nscaler.fit_transform(all_data_LableEncoded)


#######################################
# Visualisation after cleaining Data #
#######################################

# Zusätzliche Visualisierungen nach Skalierung
sns.boxplot(data=all_data_LableEncoded, orient='v', palette='Set2')
plt.show()

# Skalierte Daten wieder in DataFrame mit Spaltennamen überführen
scaled_df = pd.DataFrame(all_data_LableEncoded, columns=label_encoded_data.select_dtypes(include='number').columns)

# 1. Boxplot mit lesbaren Achsenbeschriftungen
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df, orient='v', palette='Set2')
plt.xticks(rotation=45)
plt.title("Boxplot der skalierten numerischen Features")
plt.tight_layout()
plt.show()

# 2. Boxplot nur für ausgewählte numerische Spalten
selected_cols = ['selling_price', 'km_driven', 'year']
plt.figure(figsize=(8, 5))
sns.boxplot(data=scaled_df[selected_cols], orient='v', palette='Set3')
plt.title("Boxplot ausgewählter Merkmale")
plt.tight_layout()
plt.show()

# 3. Pairplot für Verteilungen und Korrelationen
sns.pairplot(scaled_df[selected_cols])
plt.suptitle("Paarweise Verteilungen ausgewählter Merkmale", y=1.02)
plt.show()
#print("Trainingsdaten zusätzlich als 'prepared_used_car_data_train.parquet' gespeichert.")
#print("Testdaten zusätzlich als 'prepared_used_car_data_test.parquet' gespeichert.")
#print("Gesamtdaten zusätzlich als 'prepared_used_car_data.parquet' gespeichert.")


############################################################
# One-Hot-Encoding for Regression Model Training & Testing #
############################################################

# One-Hot-Encoding für kategoriale Variablen
encoded_data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

print("\nOne-Hot-Encoded Data:")
print(encoded_data.head())

print(encoded_data)

# Daten sortieren nach Jahr und Kilometerstand
encoded_data_sorted = encoded_data.sort_values(by=['year', 'km_driven'], ascending=[False, True])

# Regressionsdaten vorbereiten
features = encoded_data_sorted.columns.drop(['name', 'selling_price'])
target = 'selling_price'

X = encoded_data_sorted[features]
y = encoded_data_sorted[target]

# Testdaten speichern
all_data = pd.concat([X, y], axis=1)
all_data.to_csv('prepared_used_car_data_all.csv', index=False)

# Trainings- und Testdaten erstellen (optional, zur Kontrolle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trainingsdaten speichern
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('prepared_used_car_data_train.csv', index=False)

# Testdaten speichern
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('prepared_used_car_data_test.csv', index=False)

print("\nTrainingsdaten gespeichert als 'prepared_used_car_data_train.csv'")
print("Testdaten gespeichert als 'prepared_used_car_data_test.csv'")