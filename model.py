# save_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
import pickle
import os

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load the data
df = pd.read_csv('Final Dataset.csv')
# Save a copy in the data directory
df.to_csv('data/Final_Dataset.csv', index=False)

# Data preprocessing
# Calculate incident rates per 100,000 population
df['Child_Victim_Rate'] = (df['Number of Child Victims'] / df['Number of Child Women (5-17 years)']) * 100000
df['Adult_Victim_Rate'] = (df['Number of Adult Victims'] / df['Number of Adult Women (18-65 years)']) * 100000
df['Total_Victim_Rate'] = ((df['Number of Child Victims'] + df['Number of Adult Victims']) / 
                           (df['Number of Child Women (5-17 years)'] + df['Number of Adult Women (18-65 years)'])) * 100000

# Feature engineering
df['Child_to_Adult_Ratio'] = df['Number of Child Women (5-17 years)'] / df['Number of Adult Women (18-65 years)']
df['Child_Victim_Percentage'] = df['Number of Child Victims'] / (df['Number of Child Victims'] + df['Number of Adult Victims'])

# Create safety score (inverse of victim rate - higher means safer)
df['Safety_Score'] = 1 / (1 + df['Total_Victim_Rate'])

# Define risk categories based on victim rates
quartiles = df['Total_Victim_Rate'].quantile([0.25, 0.5, 0.75]).values
df['Risk_Category'] = pd.cut(
    df['Total_Victim_Rate'], 
    bins=[0, quartiles[0], quartiles[1], quartiles[2], float('inf')],
    labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Very High Risk']
)

# Select features for the model
features = ['Year', 'Child_Victim_Rate', 'Adult_Victim_Rate', 'Child_to_Adult_Ratio', 'Child_Victim_Percentage']

# For a classification model (predicting risk category)
X = df[features]
y_classification = df['Risk_Category']

# For a regression model (predicting safety score)
y_regression = df['Safety_Score']

# Split data
X_train, X_test, y_train_class, y_test_class = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a classifier for risk categories
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train_class)

# Train a regressor for safety scores
regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_scaled, y_train_reg)

# Save the models and scaler
pickle.dump(classifier, open('models/classifier.pkl', 'wb'))
pickle.dump(regressor, open('models/regressor.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))

print("Models trained and saved successfully!")
print("Classifier saved as: models/classifier.pkl")
print("Regressor saved as: models/regressor.pkl")
print("Scaler saved as: models/scaler.pkl")
print("Dataset copied to: data/Final_Dataset.csv")

# Test model loading
try:
    test_classifier = pickle.load(open('models/classifier.pkl', 'rb'))
    test_regressor = pickle.load(open('models/regressor.pkl', 'rb'))
    test_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("Model loading test successful!")
except Exception as e:
    print(f"Error testing model loading: {e}")