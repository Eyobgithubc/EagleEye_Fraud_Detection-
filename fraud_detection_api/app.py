from flask import Flask, jsonify, request
import pandas as pd
import joblib
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the dataset and model
df = pd.read_csv('C:/Users/teeyob/EagleEye_Fraud_Detection/data/merged_fraud_data.csv')  # Replace with your actual CSV path
model = joblib.load('C:/Users/teeyob/EagleEye_Fraud_Detection/notebooks/merged_RandomForestClassifier().pkl')    # Replace with your actual model path

# Endpoint for total transactions and fraud cases
@app.route('/api/stats', methods=['GET'])
def get_stats():
    total_transactions = df.shape[0]
    total_fraud_cases = df[df['class'] == 1].shape[0]
    fraud_percentage = (total_fraud_cases / total_transactions) * 100
    return jsonify({
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    })

# Endpoint for fraud cases over time
@app.route('/api/trends', methods=['GET'])
def get_trends():
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    monthly_fraud = df[df['class'] == 1].groupby(df['purchase_time'].dt.to_period("M")).size()
    return jsonify(monthly_fraud.to_dict())

# Endpoint for fraud by device
@app.route('/api/fraud_by_device', methods=['GET'])
def get_device_fraud():
    device_counts = df[df['class'] == 1]['device_id'].value_counts()
    return jsonify(device_counts.to_dict())

# Endpoint for fraud prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
