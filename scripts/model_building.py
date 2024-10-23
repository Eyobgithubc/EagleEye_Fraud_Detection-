# model_building.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import joblib
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM
from keras.utils import to_categorical

def data_preparation(fraud_data, credit_data):
    """
    Perform feature-target separation and train-test split for both fraud and credit datasets.
    
    Parameters:
    fraud_data (pd.DataFrame): Fraud dataset
    credit_data (pd.DataFrame): Credit card dataset
    
    Returns:
    tuple: Train and test sets for both fraud and credit datasets
    """
    # Feature and Target Separation
    X_fraud = fraud_data.drop(columns=['class'])  # Features
    y_fraud = fraud_data['class']  # Target

    X_credit = credit_data.drop(columns=['Class'])  # Features
    y_credit = credit_data['Class']  # Target

    # Train-Test Split
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

    return X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test

def train_classical_models(X_train, y_train, X_test, y_test):
    """
    Train classical machine learning models and evaluate them on the test set.
    
    Parameters:
    X_train (pd.DataFrame): Training features
    y_train (pd.Series): Training target
    X_test (pd.DataFrame): Testing features
    y_test (pd.Series): Testing target
    
    Returns:
    dict: Dictionary of trained models
    """
    # List of models to train
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'MLP': MLPClassifier()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"{name} Classification Report:\n {classification_report(y_test, y_pred)}")
        print(f"{name} Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}\n")
    
    return models

def train_cnn_model(X_train, y_train, X_test, y_test):
    """
    Train a Convolutional Neural Network (CNN) model and evaluate its performance.
    
    Parameters:
    X_train (np.array): Training features
    y_train (pd.Series): Training target
    X_test (np.array): Testing features
    y_test (pd.Series): Testing target
    
    Returns:
    keras.Model: Trained CNN model
    """
    # Reshape data for CNN
    X_train_cnn = np.expand_dims(X_train.values, axis=2)
    X_test_cnn = np.expand_dims(X_test.values, axis=2)

    # CNN Model
    cnn_model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the CNN model
    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the CNN model
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test)
    print(f"CNN Test Accuracy: {cnn_test_acc}")

    return cnn_model

def log_model_with_mlflow(model, X_train, y_train, X_test, y_test, model_name):
    """
    Log model training and evaluation with MLflow.
    
    Parameters:
    model (sklearn/keras model): Trained model
    X_train, y_train, X_test, y_test: Training and testing data
    model_name (str): Name of the model
    
    Returns:
    None
    """
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        print(f"Logged {model_name} with accuracy: {accuracy}")

def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The trained model
    filename (str): The name of the file to save the model
    
    Returns:
    None
    """
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load a saved model from a file.
    
    Parameters:
    filename (str): The name of the file to load the model from
    
    Returns:
    model: The loaded model
    """
    model = joblib.load(filename)
    return model
