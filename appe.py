
%pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib # Import joblib

# Load models
with open('logistic_regression_selected.pkl', 'rb') as f:
    model_no_opt = joblib.load(f) # Use joblib.load

with open('random_forest_selected.pkl', 'rb') as f:
    model_opt = joblib.load(f) # Use joblib.load



st.title("ML Model Deployment using Streamlit")
st.write("Enter the input features below:")

# Example feature inputs (replace with actual ones)
# Create input fields for the 10 selected features
feature_inputs = []
for i in range(10):
    feature = st.number_input(f"Feature {i+1}", min_value=0.0, max_value=100.0, step=0.1)
    feature_inputs.append(feature)

# Select model version
model_choice = st.radio("Choose Model Version:", ("Without Optimization", "With Optimization"))

# Make prediction on button click
if st.button("Predict"):
    input_data = np.array([feature_inputs])  # Use all 10 feature inputs
    model = model_opt if model_choice == "With Optimization" else model_no_opt
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader(f"Predicted Class: {prediction}")
    st.write("Probability Scores:")

    # Assuming classes are [0, 1], or get from model.classes_
    for cls, prob in zip(model.classes_, probabilities):
        st.write(f"Class {cls}: {prob:.4f}")

