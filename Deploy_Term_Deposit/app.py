import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(page_title="Bank Term Deposit Prediction", page_icon="🏦", layout="wide")

# Title and description
st.title("🏦 Bank Term Deposit Prediction")
st.markdown("""
This app predicts whether a client will subscribe to a term deposit based on their profile.
Enter the client details below, and the model will predict the likelihood of subscription.
""")

# Load the trained model, label encoders, and feature order
try:
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    with open('feature_order.pkl', 'rb') as file:
        feature_order = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Ensure 'rf_model.pkl', 'label_encoders.pkl', and 'feature_order.pkl' are in the same directory.")
    st.stop()

# Define feature options
feature_options = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
    'marital': ['married', 'single', 'divorced'],
    'education': ['primary', 'secondary', 'tertiary', 'unknown'],
    'default': ['yes', 'no'],
    'housing': ['yes', 'no'],
    'loan': ['yes', 'no'],
    'contact': ['cellular', 'telephone', 'unknown'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'poutcome': ['failure', 'success', 'unknown', 'other']
}

# Create a two-column layout
col1, col2 = st.columns(2)

# Input fields for numerical features
with col1:
    st.header("Client Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    balance = st.number_input("Average Yearly Balance (€)", min_value=-10000, max_value=100000, value=0, step=100)
    day = st.number_input("Day of Last Contact (1-31)", min_value=1, max_value=31, value=1, step=1)
    duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=0, step=10)
    campaign = st.number_input("Number of Contacts (this campaign)", min_value=1, max_value=50, value=1, step=1)
    pdays = st.number_input("Days Since Last Contact (-1 if not contacted)", min_value=-1, max_value=1000, value=-1, step=1)
    previous = st.number_input("Previous Contacts (before this campaign)", min_value=0, max_value=50, value=0, step=1)

# Input fields for categorical features
with col2:
    st.header("Client Profile")
    job = st.selectbox("Job", options=feature_options['job'])
    marital = st.selectbox("Marital Status", options=feature_options['marital'])
    education = st.selectbox("Education Level", options=feature_options['education'])
    default = st.selectbox("Credit in Default?", options=feature_options['default'])
    housing = st.selectbox("Housing Loan?", options=feature_options['housing'])
    loan = st.selectbox("Personal Loan?", options=feature_options['loan'])
    contact = st.selectbox("Contact Type", options=feature_options['contact'])
    month = st.selectbox("Month of Last Contact", options=feature_options['month'])
    poutcome = st.selectbox("Previous Campaign Outcome", options=feature_options['poutcome'])

# Predict button
if st.button("Predict Subscription", type="primary"):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    # Encode categorical features
    try:
        for column in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
            if input_data[column][0] not in label_encoders[column].classes_:
                st.error(f"Invalid value '{input_data[column][0]}' for {column}. Expected one of: {list(label_encoders[column].classes_)}")
                st.stop()
            input_data[column] = label_encoders[column].transform([input_data[column][0]])[0]
    except Exception as e:
        st.error(f"Error encoding categorical features: {e}")
        st.stop()

    # Reorder columns to match training data
    try:
        input_data = input_data[feature_order]
    except KeyError as e:
        st.error(f"Feature mismatch: {e}. Expected features: {feature_order}")
        st.stop()

    # Verify feature names match model
    if not all(input_data.columns == model.feature_names_in_):
        st.error(f"Input feature names {list(input_data.columns)} do not match model feature names {list(model.feature_names_in_)}")
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of 'yes'
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.stop()

    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"The client is likely to subscribe to a term deposit! (Probability: {probability:.2%})")
    else:
        st.warning(f"The client is unlikely to subscribe to a term deposit. (Probability: {probability:.2%})")
