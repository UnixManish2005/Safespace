# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Global variables for models and data
classifier = None
regressor = None
scaler = None
df = None

def load_model():
    global classifier, regressor, scaler, df
    
    # Load the saved models and scaler
    classifier = pickle.load(open('models/classifier.pkl', 'rb'))
    regressor = pickle.load(open('models/regressor.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    
    # Load the data for reference values
    df = pd.read_csv('data/Final_Dataset.csv')
    print("Models and data loaded successfully!")

def get_recommendations(risk_category):
    """
    Provide recommendations based on risk category.
    """
    recommendations = {
        'Low Risk': [
            "The area shows relatively lower risk compared to other regions",
            "Continue supporting prevention programs and awareness campaigns",
            "Maintain vigilance and safety practices"
        ],
        'Moderate Risk': [
            "Take standard safety precautions",
            "Be aware of surroundings, especially in less populated areas",
            "Travel in groups when possible, particularly at night",
            "Support community safety initiatives"
        ],
        'High Risk': [
            "Exercise increased caution in this area",
            "Avoid traveling alone, especially at night",
            "Stay in well-lit, populated areas",
            "Consider personal safety devices",
            "Know emergency contacts and safe locations"
        ],
        'Very High Risk': [
            "Exercise maximum caution in this area",
            "Travel with trusted companions",
            "Plan routes carefully and inform others of your whereabouts",
            "Avoid isolated areas",
            "Advocate for increased security measures and support services for survivors"
        ]
    }
    
    return recommendations.get(risk_category, ["Exercise general caution"])

def predict_safety(state, year, age):
    """
    Predict safety level for a location based on state, year, and age.
    """
    # Determine population category based on age
    if 5 <= age <= 17:
        population_col = 'Number of Child Women (5-17 years)'
        victim_col = 'Number of Child Victims'
    elif 18 <= age <= 65:
        population_col = 'Number of Adult Women (18-65 years)'
        victim_col = 'Number of Adult Victims'
    else:
        return {"Error": "Age must be between 5 and 65."}
    
    # Get average population for normalization
    state_data = df[df['State'] == state]
    if state_data.empty:
        state_data = df  # Use all data if state not found
        
    year_data = state_data[state_data['Year'] == year]
    if year_data.empty:
        year_data = state_data  # Use all state data if year not found
    
    avg_population = year_data[population_col].mean()
    avg_victims = year_data[victim_col].mean()
    avg_adult_population = year_data['Number of Adult Women (18-65 years)'].mean()
    avg_adult_victims = year_data['Number of Adult Victims'].mean()
    
    # Estimate rates
    victim_rate = (avg_victims / avg_population) * 100000 if avg_population > 0 else 0
    
    # Prepare input features
    X_new = pd.DataFrame({
        'Year': [year],
        'Child_Victim_Rate': [victim_rate if age <= 17 else 0],
        'Adult_Victim_Rate': [victim_rate if age > 17 else 0],
        'Child_to_Adult_Ratio': [avg_population / avg_adult_population if avg_adult_population > 0 else 0],
        'Child_Victim_Percentage': [avg_victims / (avg_victims + avg_adult_victims) if avg_victims + avg_adult_victims > 0 else 0]
    })
    
    # Ensure no NaN values before scaling
    X_new.fillna(0, inplace=True)
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    risk_category = classifier.predict(X_new_scaled)[0]
    safety_score = regressor.predict(X_new_scaled)[0]
    
    return {
        'State': state,
        'Year': year,
        'Age': age,
        'Risk_Category': risk_category,
        'Safety_Score': round(float(safety_score), 4),
        'Recommendations': get_recommendations(risk_category)
    }

@app.route('/')
def home():
    # Get list of states from the dataset
    states = sorted(df['State'].unique())
    years = sorted(df['Year'].unique())
    return render_template('index.html', states=states, years=years)

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form.get('state')
    year = int(request.form.get('year'))
    age = int(request.form.get('age'))
    
    result = predict_safety(state, year, age)
    return jsonify(result)

# Initialize models when the app starts
@app.before_request
def before_request():
    global classifier, regressor, scaler, df
    if classifier is None or regressor is None or scaler is None or df is None:
        load_model()

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load models at startup
    load_model()
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)
    