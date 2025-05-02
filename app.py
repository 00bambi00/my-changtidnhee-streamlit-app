import streamlit as st
import pandas as pd
import joblib

# โหลด model และ encoders
rf_loaded = joblib.load('credit_scoring_model.pkl')
le_home_loaded = joblib.load('le_home.pkl')
le_default_loaded = joblib.load('le_default.pkl')

st.title('Credit Scoring Prediction App')

# สร้าง input form
person_age = st.number_input('Age', min_value=18, max_value=100, value=30)
person_income = st.number_input('Income', min_value=0, value=50000)
person_home_ownership = st.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE'])
person_emp_length = st.number_input('Employment Length', min_value=0.0, value=5.0)
loan_percent_income = st.number_input('Loan % Income', min_value=0.0, max_value=1.0, value=0.2)
cb_person_cred_hist_length = st.number_input('Credit History Length', min_value=0, value=3)
cb_person_default_on_file = st.selectbox('Default on File', ['Y', 'N'])

if st.button('Predict'):
    # เตรียม input
    new_data = pd.DataFrame([{
        'person_age': person_age,
        'person_income': person_income,
        'person_home_ownership': person_home_ownership,
        'person_emp_length': person_emp_length,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'cb_person_default_on_file': cb_person_default_on_file
    }])

    # แปลง categorical
    new_data['person_home_ownership'] = le_home_loaded.transform(new_data['person_home_ownership'])
    new_data['cb_person_default_on_file'] = le_default_loaded.transform(new_data['cb_person_default_on_file'])

    # predict
    prediction = rf_loaded.predict(new_data)
    probability = rf_loaded.predict_proba(new_data)

    prediction_text = 'Loan Rejected' if prediction[0] == 1 else 'Loan Approved'

    st.write(f'**Predicted loan_status:** {prediction_text}')
    st.write(f'**Probability [class 0, class 1]:** {probability[0]}')
