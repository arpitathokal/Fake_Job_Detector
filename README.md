🕵️‍♀️ Fake Job Detector

Detect fake job postings using Machine Learning and Natural Language Processing (NLP).
---

### Overview

This project identifies fraudulent job advertisements by combining text features (title, description, requirements) with structured signals (e.g., employment type, country). It includes:

-A trained soft-voting ensemble (LR, RF, NB, DT)
-A Streamlit web app for real-time predictions
-Reproducible notebooks for EDA and model training
---

### 📁 Project Structure

```text
fake-job-detector/
├── app/                   # Streamlit UI app
│   └── fake_job_streamlit_app.py
├── images                 # fake and real UI image
├── models/                # Saved models (.pkl)
│  ├─ ensemble_model.pkl
│  ├─ preprocessor.pkl
│  └─ (other .pkl/.csv as needed)
├── notebooks/             # EDA and model training notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Supervised_ensemble.ipynb
├──  fake_job_postings.csv  #dataset
├── requirements.txt       # Project dependencies
├── README.md              # Project guide
```

---
### ⚙️ Features

-EDA with visualizations (class imbalance, metadata distribution, etc.)
-Preprocessing pipeline: text cleaning, lemmatization, one-hot encoding
-Feature extraction with TF-IDF (text) + metadata
-Supervised models: Logistic Regression, Decision Tree, Random Forest, Naïve Bayes
-Soft-voting ensemble for balanced performance
-Streamlit app for interactive job posting classification

---

### 📦 Dataset

- Kaggle: [Fake Job Postings Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

---

### 🚀 How to Run the App

Install requirements:

bash
pip install -r requirements.txt 

---

### 🧠 Model Training
Train and evaluate using the notebooks in this order:

01_EDA.ipynb
02_fake_job_streamlit_app.py

---

### 💡 How It Works

Text preprocessing: lowercasing, stopword removal, lemmatization
Feature extraction: TF-IDF for text + one-hot encoding for metadata
Balancing: SMOTE applied to handle class imbalance
Modeling: classical ML models trained and compared
Ensemble: weighted soft-voting ensemble selected as best performer
Deployment: Streamlit app for real-time predictions

---
RESULTS

| Model                    | Accuracy  | Precision | Recall    | F1        | ROC-AUC   |
| ------------------------ | --------- | --------- | --------- | --------- | --------- |
| Naïve Bayes              | 0.904     | 0.321     | 0.873     | 0.469     | 0.961     |
| Logistic Regression      | 0.977     | 0.725     | 0.855     | 0.785     | 0.981     |
| Decision Tree            | 0.958     | 0.547     | 0.746     | 0.631     | 0.857     |
| Random Forest            | 0.984     | 0.983     | 0.682     | 0.805     | 0.991     |
| **Soft Voting Ensemble** | **0.985** | **0.852** | **0.832** | **0.842** | **0.988** |

---
## 👩‍💻 Author
Arpita Thokal
