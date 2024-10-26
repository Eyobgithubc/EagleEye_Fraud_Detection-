from flask import Flask, request, jsonify
import logging

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_fraud(data):
    try:
        transaction_amount = data.get("transaction_amount")
        if transaction_amount is None:
            logger.error("Transaction amount is missing.")
            return {"error": "Transaction amount is required."}
        
        
        if transaction_amount > 1000:
            return {"fraud": True}
        return {"fraud": False}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return {"error": "Prediction failed"}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    logger.info(f"Received request data: {data}")
    prediction = predict_fraud(data)
    logger.info(f"Prediction result: {prediction}")
    return jsonify(prediction)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
