import streamlit as st
import joblib
import re
from scipy.sparse import hstack, csr_matrix

# Load model and vectorizer
model = joblib.load('models/ensemble_model.pkl')
vectorizer = joblib.load('models/ensemble_vectorizer.pkl')

# UI
st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write("Fill out the job posting fields below to predict whether the job is **real or fake**.")

# Input fields
job_title = st.text_input("Job Title")
job_requirements = st.text_area("Requirements", height=150)
job_description = st.text_area("Job Description", height=200)

if st.button("Detect"):
    if not job_description.strip() and not job_requirements.strip() and not job_title.strip():
        st.warning("Please fill in at least one job field.")
    else:
        # Combine fields into one text blob
        combined_text = " ".join([job_title.strip(), job_requirements.strip(), job_description.strip()])

        # Detect suspicious keywords
        pattern = re.compile(
    r"(money|earn|transfer|urgent|wire|\$\d+|start immediately|quick cash|"
    r"work from home|no skills|required|instant payments|flexible hours|click here|"
    r"guarantee|easy|limited spots|sign up now|send you a check|salary transfers|"
    r"direct deposit|onboarding fee|payment in advance)", re.IGNORECASE)

        has_suspicious = int(bool(pattern.search(combined_text)))

        # Vectorize + combine
        text_vector = vectorizer.transform([combined_text])
        suspicious_flag = csr_matrix([[has_suspicious]])
        combined_input = hstack([text_vector, suspicious_flag])

        # Predict
        prediction = model.predict(combined_input)[0]
        probability = model.predict_proba(combined_input)[0][1]

        # Override logic
        if has_suspicious and probability < 0.90:
            prediction = 1  # force fake

        # Show result
        if prediction == 1:
            st.error(f"🚨 This job is likely **FAKE** with {probability * 100:.2f}% probability.")
        else:
            st.success(f"✅ This job seems **REAL** with {(1 - probability) * 100:.2f}% confidence.")
   