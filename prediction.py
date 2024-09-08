import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df=pd.read_excel(r"C:\Users\PC\Desktop\master thesis(ΑυτόματηΑνάκτηση3).xlsx")
df
from sklearn.preprocessing import LabelEncoder
le_Location = LabelEncoder()
df['Location'] = le_Location.fit_transform(df['Location'])
df["Location"].unique()

le_Close_to_the_sea = LabelEncoder()
df['Close_to_the_sea'] = le_Close_to_the_sea.fit_transform(df['Close_to_the_sea'])
df["Close_to_the_sea"].unique()

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

X=df.drop(columns=['Price','Construction_material'])
y=df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)  # 20% validation, 10% test

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
model = RandomForestRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],           # Number of trees in the forest
    'max_depth': [10, 20, 30, None],           # Maximum depth of the tree (None for no limit)
    'min_samples_split': [2, 5, 10],           # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples required to be at a leaf node
    'max_features': ['sqrt', 'log2'],  # Number of features to consider for the best split
    'criterion': ['squared_error'], 
}

# Define cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X, y)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validated MSE: {best_score}")

# Optionally, fit the best model and evaluate on the test set
best_model = grid_search.best_estimator_



val_preds = best_model.predict(X_val)
test_preds = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Evaluating the model
val_mse = mean_squared_error(y_val, val_preds)
val_mae = mean_absolute_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

test_mse = mean_squared_error(y_test, test_preds)
test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)


print(f'Validation MSE: {val_mse}')
print(f'Validation MAE: {val_mae}')
print(f'Validation R²: {val_r2}')

print(f'Test MSE: {test_mse}')
print(f'Test MAE: {test_mae}')
print(f'Test R²: {test_r2}')
train_preds = best_model.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds)
train_mae = mean_absolute_error(y_train, train_preds)
train_r2 = r2_score(y_train, train_preds)

print(f"Train MSE: {train_mse}")
print(f"Train MAE: {train_mae}")
print(f"Train R²: {train_r2}")

import joblib
data = {'model': model, 'le_Location':le_Location,'le_Close_to_the_sea':le_Close_to_the_sea, 'le_Close_to_the_center':le_Close_to_the_center, 'le_Heat':le_Heat, 'le_Renovated':le_Renovated, 'le_Garden':le_Garden, 'le_Parking':le_Parking, 'le_View':le_View}
with open('model.joblib2', 'wb') as file:
   joblib.dump(data, file, compress=True, protocol=-1)

