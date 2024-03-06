from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the models
model_fertilizer = pickle.load(open('model_fertilizer.pkl', 'rb'))
model_pesticide = pickle.load(open('model_pesticide.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predicting results
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    nitrogen_level = float(request.form['nitrogen'])
    phosphorus_level = float(request.form['phosphorus'])
    potassium_level = float(request.form['potassium'])
    soil_color = int(request.form['soilColor'])
    crop = int(request.form['crop'])
    pest = int(request.form['pest'])

    # Make predictions
    predicted_fertilizer, predicted_pesticide = predict_fertilizer_and_pesticide(
        nitrogen_level, phosphorus_level, potassium_level, soil_color, crop, pest
    )

    # Render the prediction results on the frontend
    return render_template('index.html', predicted_fertilizer=predicted_fertilizer, predicted_pesticide=predicted_pesticide)

def predict_fertilizer_and_pesticide(nitrogen_level, phosphorus_level, potassium_level, soil_color, crop, pest):
    # Prepare input data for prediction
    input_data = {
        'Nitrogen_Level': [nitrogen_level],
        'Phosphorus_Level': [phosphorus_level],
        'Potassium_Level': [potassium_level],
        'Soil_Color': [soil_color],
        'Crop': [crop],
        'Pest':[pest]
    }
    input_df = pd.DataFrame(input_data)

    # Make predictions
    predicted_fertilizer = model_fertilizer.predict(input_df)
    predicted_pesticide = model_pesticide.predict(input_df)

    return predicted_fertilizer[0], predicted_pesticide[0]

if __name__ == '__main__':
    app.run()
