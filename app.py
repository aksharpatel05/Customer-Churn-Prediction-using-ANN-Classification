import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle



#load the model
model = tf.keras.models.load_model('model.h5')

#load the encoder and scaler
with open('lable_encoder_gender.pkl', 'rb') as file:
    lable_encoder_gender = pickle.load(file)

with open('onehotencoder_geography.pkl', 'rb') as file:
    onehotencoder_geography = pickle.load(file)

with open('sc.pkl', 'rb') as file:
    sc = pickle.load(file)

#streamlit app
# Header Section
st.title("🔮 Customer Churn Prediction")
st.subheader("Predict the likelihood of a customer leaving the service.")
st.write("Provide the customer details below, and we'll calculate the probability of churn.")

# Input Form Section
st.markdown("### ✍️ Input Customer Details")

with st.form("churn_form"):
    st.markdown("#### 📋 General Information")
    credit_score = st.slider("Credit Score (300-900)", min_value=300, max_value=900, value=600, step=1, help="A measure of the customer's creditworthiness.")
    geography = st.radio("Geography", options=["France", "Germany", "Spain"], help="The customer's country of residence.")
    gender = st.radio("Gender", options=["Male", "Female"], help="The customer's gender.")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="The customer's age in years.")
    
    st.markdown("#### 🏦 Financial Details")
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f", help="The customer's current bank account balance.")
    estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f", help="The customer's estimated yearly salary.")
    
    st.markdown("#### 💼 Service Details")
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5, step=1, help="The number of years the customer has been with the service.")
    num_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], help="The number of products the customer is using.")
    has_cr_card = st.radio("Has Credit Card?", options=["Yes", "No"], help="Does the customer have a credit card?")
    is_active_member = st.radio("Is Active Member?", options=["Yes", "No"], help="Is the customer actively using the service?")

    # Submit button
    submitted = st.form_submit_button("🔍 Predict")

# Prediction Section
if submitted:

    # data
    data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Preprocess gender
    gender_encoded = lable_encoder_gender.transform([gender])[0]  # Label encoding
    gender_encoded_df = pd.DataFrame([[gender_encoded]], columns = ['Gender'])

    # Preprocess geography (ensure columns match training data)
    geography_encoded = onehotencoder_geography.transform([[geography]])  # One-hot encoding
    geography_encoded_df = pd.DataFrame(
        geography_encoded, 
        columns = ['France', 'Germany', 'Spain']
    )

    # Create a DataFrame for binary features
    binary_features = {
        "HasCrCard": [1 if has_cr_card == "Yes" else 0],
        "IsActiveMember": [1 if is_active_member == "Yes" else 0]
    }
    binary_features_df = pd.DataFrame(binary_features)

    # Combine all features into a single DataFrame (ensure the order matches training data)
    input_features = pd.concat(
        [
            data[['CreditScore']],
            gender_encoded_df,
            data[['Age', 'Tenure', 'Balance', 'NumOfProducts']],
            binary_features_df,
            data[['EstimatedSalary']],
            geography_encoded_df  
        ],
        axis=1
    )
    

    # Scale the input features
    input_features_scaled = sc.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)
    churn_probability = prediction[0][0]  # Assuming binary classification output

    # Display result
    st.markdown("### 🧾 Prediction Result")
    if churn_probability > 0.5:
        st.error(f"⚠️ High Risk of Churn: {churn_probability:.2%}")
        st.write("This customer is likely to leave the service. Consider offering incentives or engaging them more actively.")
    else:
        st.success(f"✅ Low Risk of Churn: {churn_probability:.2%}")
        st.write("This customer is likely to stay with the service. Keep up the good work!")
