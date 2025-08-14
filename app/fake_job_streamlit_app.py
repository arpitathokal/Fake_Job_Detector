import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path("./models/artifacts_supervised_grid_ensemble")

# ---------- utils ----------
def simple_clean(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_meta():
    meta_path = ARTIFACT_DIR / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    # sensible fallbacks
    return {
        "combined_text_col": "combined_text",
        "numeric_cols_used": [],
        "models": []
    }

def find_preprocessor():
    # prefer the "best" preprocessor, else fallback
    for name in ["preprocessor_best.pkl", "preprocessor.pkl"]:
        p = ARTIFACT_DIR / name
        if p.exists():
            return p
    return None

def list_available_models():
    # try "best" first, then individual models saved by your notebook
    order = [
        ("Best model (auto)", "model_best.pkl"),
        ("Voting (soft)", "voting_soft.pkl"),
        ("Random Forest", "rf.pkl"),
        ("Logistic Regression", "lr.pkl"),
        ("Naive Bayes", "nb.pkl"),
        ("Decision Tree", "dt.pkl"),
    ]
    available = []
    for label, fname in order:
        p = ARTIFACT_DIR / fname
        if p.exists():
            available.append((label, p))
    return available

# ---------- load artifacts ----------
st.title("🕵️‍♂️ Fake Job Posting Detector — Supervised Models")
if not ARTIFACT_DIR.exists():
    st.error(f"Artifacts folder not found: {ARTIFACT_DIR.resolve()}")
    st.stop()

meta = load_meta()
text_col = meta.get("combined_text_col", "combined_text")
num_cols = meta.get("numeric_cols_used", [])  # may be empty

preproc_path = find_preprocessor()
if preproc_path is None:
    st.error("No preprocessor found. Expected preprocessor_best.pkl or preprocessor.pkl in the artifacts folder.")
    st.stop()

try:
    preprocessor = joblib.load(preproc_path)
except Exception as e:
    st.error(f"Failed to load preprocessor: {e}")
    st.stop()

model_choices = list_available_models()
if not model_choices:
    st.error("No model files found in artifacts folder.")
    st.stop()

label_default, path_default = model_choices[0]
label_to_path = {lbl: p for lbl, p in model_choices}
choice = st.selectbox("Choose model", [lbl for lbl, _ in model_choices], index=0)
model_path = label_to_path[choice]

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.caption(f"Using preprocessor: `{preproc_path.name}` • model: `{model_path.name}`")

# ---------- inputs ----------
col1, col2 = st.columns(2)
with col1:
    job_title = st.text_input("Job Title")
with col2:
    benefits = st.text_input("Benefits (optional)")

requirements = st.text_area("Requirements", height=120)
description = st.text_area("Job Description", height=180)

# ---------- predict ----------
if st.button("Detect"):
    if not (job_title.strip() or requirements.strip() or description.strip() or benefits.strip()):
        st.warning("Please enter at least one field.")
        st.stop()

    combined_text = " ".join([
        simple_clean(job_title),
        simple_clean(requirements),
        simple_clean(description),
        simple_clean(benefits),
    ]).strip()

    # Build EXACT input schema preprocessor expects
    row = {text_col: combined_text}
    # Supply numeric columns if your pipeline used *_word_count (set 0 if you don't compute them here)
    for c in num_cols:
        row[c] = 0

    X_df = pd.DataFrame([row])

    try:
        X_trans = preprocessor.transform(X_df)
    except Exception as e:
        st.error(f"Preprocessor.transform failed. Check meta.json columns.\nError: {e}")
        st.stop()

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_trans)[:, 1][0])
        else:
            proba = None
        pred = int(model.predict(X_trans)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    # Display result
    if pred == 1:
        if proba is not None:
            st.error(f"🚨 Likely **FAKE** ({proba*100:.2f}% probability)")
        else:
            st.error("🚨 Likely **FAKE**")
    else:
        if proba is not None:
            st.success(f"✅ Likely **REAL** ({(1-proba)*100:.2f}% confidence)")
        else:
            st.success("✅ Likely **REAL**")

    # Debug info
    with st.expander("Debug details"):
        st.write("Input row:", row)
        st.write("Text column name:", text_col)
        st.write("Numeric columns expected:", num_cols)
        st.write("Feature shape:", X_trans.shape)
