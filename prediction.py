import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df=pd.read_excel(r"C:\Users\PC\Desktop\master thesis(ΑυτόματηΑνάκτηση).xlsx")
from sklearn.preprocessing import LabelEncoder
le_Location = LabelEncoder()
df['Location'] = le_Location.fit_transform(df['Location'])
df["Location"].unique()

le_Construction_material = LabelEncoder()
df['Construction_material'] = le_Construction_material.fit_transform(df['Construction_material'])
df["Construction_material"].unique()

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

X=df.drop(columns=['Price'])
y=df['Price']

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=4)

reg=RandomForestRegressor()

reg.fit(X_train, y_train)
reg.get_params()

y_pred=reg.predict(X_train)

from sklearn import metrics
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

import joblib
data = {'model': reg, 'le_Location':le_Location, 'le_Construction_material':le_Construction_material, 'le_Close_to_the_sea':le_Close_to_the_sea, 'le_Close_to_the_center':le_Close_to_the_center, 'le_Heat':le_Heat, 'le_Renovated':le_Renovated, 'le_Garden':le_Garden, 'le_Parking':le_Parking   }
with open('model.joblib', 'wb') as file:
   joblib.dump(data, file, compress=True, protocol=-1)

