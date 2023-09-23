# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

# Load and preprocess the dataset (replace with your dataset)
data = pd.read_csv('forest_fire_dataset.csv')
# Perform data cleaning and feature engineering steps as needed

# Split the dataset into training and testing sets
X = data.drop('target_column', axis=1)
y = data['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model (replace with your chosen model)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Create a Flask web application for real-time predictions
app = Flask(_name_)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame(data, index=[0])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if _name_ == '_main_':
    app.run(debug=True)
