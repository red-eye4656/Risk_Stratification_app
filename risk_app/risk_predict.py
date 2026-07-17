
import streamlit as st
import pandas as pd 
import joblib
model = joblib.load("risk_model.pkl")
st.title("Risk Stratification App")
age = st.number_input("Age", min_value=0)
length_of_stay = st.number_input("Length of Stay (days)", min_value=0)
treatment_cost = st.number_input("Treatment Cost", min_value=0.0)
if st.button('Predict Risk'):
    input_data = pd.DataFrame(
    [[age, length_of_stay, treatment_cost]],
    columns=['Age', 'LengthofStay', 'TreatmentCost']
)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Risk Prediction: {'High Risk' if prediction ==1 else 'Low Risk'}")
    st.write(f"Risk Probability: {round(probability, 2)}")