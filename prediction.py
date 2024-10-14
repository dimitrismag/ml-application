import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

X=df.drop(columns=['Price'])
y=df['Price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)  # 20% validation, 10% test

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

best_model = xgb.XGBRegressor(
    n_estimators=500,           # Best number of trees
    learning_rate=0.1,          # Best learning rate
    max_depth=3,                # Best tree depth
    subsample=0.8,              # Best subsample ratio of training instances
    colsample_bytree=1.0,       # Best subsample ratio of columns for each tree
    gamma=0,                    # Best minimum loss reduction to make a partition
    min_child_weight=1,         # Best minimum sum of weights of all observations needed in a child
    reg_alpha=1.0,              # Best L1 regularization term on weights
    reg_lambda=1.5              # Best L2 regularization term on weights
)

# Fit the model to your data
best_model.fit(X, y)



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
data = {'model': best_model, 'le_Location':le_Location,'le_Close_to_the_sea':le_Close_to_the_sea, 'le_Close_to_the_center':le_Close_to_the_center, 'le_Heat':le_Heat, 'le_Renovated':le_Renovated, 'le_Garden':le_Garden, 'le_Parking':le_Parking, 'le_View':le_View}
with open('model.joblib1', 'wb') as file:
   joblib.dump(data, file, compress=True, protocol=-1)

