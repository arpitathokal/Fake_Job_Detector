🕵️‍♀️ Fake Job Detector

Detect fake job postings using Machine Learning and Natural Language Processing (NLP).

Overview

This project identifies fake job advertisements by analyzing text features like job descriptions and requirements, along with structured features such as telecommuting, logo presence, etc. It includes a trained ensemble model and an interactive Streamlit web app.

---

📁 Project Structure
fake-job-detector/
├── app/ # Streamlit UI app
├── models/ # Saved models (.pkl)
├── notebooks/ # EDA and modeling
├── requirements.txt # Dependencies
├── fake_job_posting.csv
├── README.md # Project guide

⚙️ Features

- EDA with visualizations
- Data cleaning & preprocessing pipeline
- Supervised models: Random Forest, Logistic Regression, Naive Bayes
- Anomaly detectors: One-Class SVM & Isolation Forest (for comparison)
- Voting ensemble model with keyword flag
- Streamlit UI for real-time prediction

---

📦 Dataset

- Kaggle: [Fake Job Postings Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

---

🚀 How to Run the App

Install requirements:

bash
pip install -r requirements.txt 

🧠 Model Training
Train and evaluate using the notebooks in this order:

01_EDA.ipynb
02_Modeling_RF.ipynb
03_Modeling_OCSVM.ipynb
04_Modeling_IF.ipynb
05_Ensemble_Eval_with_Keywords.ipynb

💡 How It Works
Text data is vectorized using TF-IDF
A soft-voting ensemble predicts final results
UI displays real/fake label with confidence score

👩‍💻 Author
Arpita Thokal
