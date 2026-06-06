import pandas as pd
import joblib
import streamlit as st
model = joblib.load("model.pkl")
bank_columns = joblib.load("features.pkl")
st.title("Bank Customer Retention Dashboard")
st.write("Enter customer details below to calculate their risk of leaving the bank")
age = st.slider("customer age", min_value=18, max_value=100, value=25)
balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.0)
credit_score = st.slider("credit score", min_value=300, max_value=850, value=600)
activate_number = st.selectbox("Is this an Active Member?", options=["Yes", "No"])
estimated_salary = st.number_input("Estimated salary ($)", min_value=0.0, value=5000.0)
tenure = st.slider("Years with bank", min_value=0, max_value=10, value=5)
gender = st.selectbox("Gender", options=["Male", "Female"])
country = st.selectbox("Country", options=["Portugal", "France", "Spain"])
products_number = st.slider("Number of Products", min_value=1, max_value=4, value=1)

active_val = 1 if activate_number == "Yes" else 0
gender_val = 1 if gender == "Male" else 0

if st.button("Analyze Churn Risk"):

    payload = {
        "age": age,
        "credit_score": credit_score,
        "balance": balance,
        "active_member": active_val,
        "estimated_salary": estimated_salary,
        "tenure": tenure,
        "gender": gender,
        "country": country,
        "products_number": products_number,
        "limite_credito": 10000,
        "credit_card": 1,
        "cidade": "Lisbon"
    }

    df_input = pd.DataFrame([payload])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=bank_columns, fill_value=0)
    prediction = model.predict(df_input)[0]
    if prediction == 1:
        st.error("HIGH RISK: This customer is highly likely to close their account!")
    else:
        st.success("LOW RISK: This customer is stable and likely to stay.")