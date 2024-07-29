import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df=pd.read_excel(r"C:\Users\PC\Desktop\master thesis(ΑυτόματηΑνάκτησηfinished).xlsx")
df
from sklearn.preprocessing import LabelEncoder
le_Location = LabelEncoder()
df['Location'] = le_Location.fit_transform(df['Location'])
df["Location"].unique()

le_Construction_material = LabelEncoder()
df['Construction_material'] = le_Construction_material.fit_transform(df['Construction_material'])
df["Construction_material"].unique()

le_Close_to_the_sea = LabelEncoder()
df['Close_to_the_sea_(<500m)'] = le_Close_to_the_sea.fit_transform(df['Close_to_the_sea_(<500m)'])
df["Close_to_the_sea_(<500m)"].unique()

le_Close_to_the_center = LabelEncoder()
df['Close_to_the_center'] = le_Close_to_the_center.fit_transform(df['Close_to_the_center'])
df["Close_to_the_center"].unique()

le_Heat = LabelEncoder()
df['Heat'] = le_Heat.fit_transform(df['Heat'])
df["Heat"].unique()

le_Renovated = LabelEncoder()
df['Renovated'] = le_Renovated.fit_transform(df['Renovated'])
df["Renovated"].unique()

le_Garden = LabelEncoder()
df['Garden'] = le_Garden.fit_transform(df['Garden'])
df["Garden"].unique()

le_Parking = LabelEncoder()
df['Parking'] = le_Parking.fit_transform(df['Parking'])
df["Parking"].unique()

df['View'].fillna('Unknown', inplace=True)

le_View = LabelEncoder()
df['View'] = le_View.fit_transform(df['View'])
df["View"].unique

X=df.drop(columns=['Price'])
y=df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)  # 20% validation, 10% test

reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)

y_test_pred = reg.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluation on test data
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")

import joblib
data = {'model': reg, 'le_Location':le_Location, 'le_Construction_material':le_Construction_material, 'le_Close_to_the_sea':le_Close_to_the_sea, 'le_Close_to_the_center':le_Close_to_the_center, 'le_Heat':le_Heat, 'le_Renovated':le_Renovated, 'le_Garden':le_Garden, 'le_Parking':le_Parking, 'le_View':le_View   }
with open('model.joblib2', 'wb') as file:
   joblib.dump(data, file, compress=True, protocol=-1)

