from flask import Flask,request,jsonify
import numpy as np
import joblib
from joblib import load
from sklearn.preprocessing import LabelEncoder





with open('model.joblib2', 'rb') as file:
    data = joblib.load(file)
    reg = data['model']
    le_Location = data['le_Location']
    le_Close_to_the_sea = data['le_Close_to_the_sea']
    le_Close_to_the_center = data['le_Close_to_the_center']
    le_Heat = data['le_Heat']
    le_Renovated = data['le_Renovated']
    le_Garden = data['le_Garden']
    le_Parking = data['le_Parking']
    le_View = data['le_View']


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    Location = request.form.get('Location')
    Square_feet = int(request.form.get('Square_feet'))
    Rooms = int(request.form.get('Rooms'))
    Bathrooms = int(request.form.get('Bathrooms'))
    Construction_year = int(request.form.get('Construction_year'))
    Number_of_levels = int(request.form.get('Number_of_levels'))
    Close_to_the_sea = request.form.get('Close_to_the_sea')
    Close_to_the_center = request.form.get('Close_to_the_center')
    Floor = int(request.form.get('Floor'))
    Number_of_balconies = int(request.form.get('Number_of_balconies'))
    Heat = request.form.get('Heat')
    Renovated = request.form.get('Renovated')
    Garden = request.form.get('Garden')
    Postcard = int(request.form.get('Postcard'))
    Parking = request.form.get('Parking')
    View = request.form.get('View')

    location_encoded = le_Location.transform([Location])[0]
    close_to_the_sea_encoded = le_Close_to_the_sea.transform([Close_to_the_sea])[0]
    close_to_the_center_encoded = le_Close_to_the_center.transform([Close_to_the_center])[0]
    heat_encoded = le_Heat.transform([Heat])[0]
    renovated_encoded = le_Renovated.transform([Renovated])[0]
    garden_encoded = le_Garden.transform([Garden])[0]
    parking_encoded = le_Parking.transform([Parking])[0]
    View_encoded = le_View.transform([View])[0]

    input_query = np.array([[location_encoded, Square_feet, Bedrooms, Bathrooms, Construction_year, Number_of_levels, close_to_the_sea_encoded, close_to_the_center_encoded, Floor, Number_of_balconies, heat_encoded, renovated_encoded, garden_encoded, Postcard, parking_encoded, View_encoded]])
    result = reg.predict(input_query)[0]
    formatted_price = '{:.2f} €'.format(result)
    return jsonify({'price':formatted_price})

if __name__ == '__main__':
    app.run(debug=True)
