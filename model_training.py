# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_model():
    # Load the dataset
    data = pd.read_csv('data_balita.csv')

    # Encode categorical variables
    data['Jenis Kelamin'] = data['Jenis Kelamin'].map({'laki-laki': 0, 'perempuan': 1})
    data['Status Gizi'] = data['Status Gizi'].map({'stunted': 0, 'normal': 1, 'tinggi': 2, 'severely stunted': 3})

    # Define features and target variable
    X = data.drop(columns=['Status Gizi'])
    y = data['Status Gizi']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, 'trained_model.pkl')

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    return report

if __name__ == "__main__":
    report = train_model()
    print(report)