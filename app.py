import streamlit as st
import pandas as pd
import joblib
# import numpy as np

# โหลด model และ encoders
model = joblib.load("loan_model_extended_muticlass_randomforest_credit_score.pkl")
encoder = joblib.load("encoders_extended_muticlass_randomforest_credit_score.pkl")

# Manual mapping แทน encoder ที่หายไป
education_map = {'Vocational': 0, 'Secondary': 1, 'Primary': 2, 'None': 3}
loan_purpose_map = {'business': 0, 'personal': 1}
home_ownership_map = {'own': 0, 'rent': 1}
certificate_map = {'Yes': 0, 'No': 1}
gender_map = {'Male': 0, 'Female': 1}
marital_status_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
region_map = {'North': 0, 'Central': 1, 'South': 2, 'East': 3, 'West': 4}
occupation_map = {'Private': 0, 'Government': 1, 'Freelancer': 2, 'Unemployed': 3}

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("🎯 Project: AI-Powered Credit rating service (3-Class)")

with st.form("loan_form"):
    st.subheader("📋 Applicant Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        worker_id = st.text_input("Worker ID")
        Gender = st.selectbox("Gender", list(gender_map.keys()))
        Age = st.slider("Age", 18, 70, 30)
        Occupation = st.selectbox("Occupation", list(occupation_map.keys()))
        Education = st.selectbox("Education", list(education_map.keys()))
        Marital_Status = st.selectbox("Marital Status", list(marital_status_map.keys()))
        Work_Experience = st.slider("Work Experience (Years)", 0, 40, 5)

    with col2:
        Certificate = st.selectbox("Certificate", list(certificate_map.keys()))
        Region = st.selectbox("Region", list(region_map.keys()))
        Monthly_Income = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
        Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
        loan_purpose = st.selectbox("Loan Purpose", list(loan_purpose_map.keys()))
        home_ownership = st.selectbox("Home Ownership", list(home_ownership_map.keys()))
        dependents = st.slider("Dependents", 0, 10, 1)

    with col3:
        job_completion_rate = st.slider("Job Completion Rate (%)", 0.0, 100.0, 85.0)
        on_time_rate = st.slider("On Time Rate (%)", 0.0, 100.0, 90.0)
        avg_response_time_mins = st.number_input("Avg. Response Time (mins)", value=10.0)
        customer_rating_avg = st.slider("Customer Rating Avg.", 0.0, 5.0, 4.2)
        job_acceptance_rate = st.slider("Job Acceptance Rate (%)", 0.0, 100.0, 80.0)
        job_cancellation_count = st.number_input("Job Cancellation Count", 0, 100, 2)
        weekly_active_days = st.slider("Weekly Active Days", 0, 7, 5)
        membership_duration_months = st.number_input("Membership Duration (months)", 0, 240, 24)
        simulated_credit_score = st.slider("Simulated Credit Score",400, 900, 600)
        work_consistency_index = st.slider("Work Consistency Index", 0.0, 1.0, 0.75)
        inactive_days_last_30 = st.number_input("Inactive Days (last 30)", 0, 30, 3)
        rejected_jobs_last_30 = st.number_input("Rejected Jobs (last 30)", 0, 30, 1)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame([{
            "Gender": gender_map[Gender],
            "Age": Age,
            "Occupation": occupation_map[Occupation],
            "Education": education_map[Education],
            "Marital_Status": marital_status_map[Marital_Status],
            "Work_Experience": Work_Experience,
            "Certificate": certificate_map[Certificate],
            "Region": region_map[Region],
            "Monthly_Income": Monthly_Income,
            "Loan_Amount": Loan_Amount,
            "loan_purpose": loan_purpose_map[loan_purpose],
            "home_ownership": home_ownership_map[home_ownership],
            "dependents": dependents,
            "job_completion_rate": job_completion_rate,
            "on_time_rate": on_time_rate,
            "avg_response_time_mins": avg_response_time_mins,
            "customer_rating_avg": customer_rating_avg,
            "job_acceptance_rate": job_acceptance_rate,
            "job_cancellation_count": job_cancellation_count,
            "weekly_active_days": weekly_active_days,
            "membership_duration_months": membership_duration_months,
            "simulated_credit_score": simulated_credit_score,
            "work_consistency_index": work_consistency_index,
            "inactive_days_last_30": inactive_days_last_30,
            "rejected_jobs_last_30": rejected_jobs_last_30
        }])

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        st.success(f"📊 Loan Status Prediction: **{prediction}**")
        st.info(f"Prediction Probabilities: {prediction_proba}")