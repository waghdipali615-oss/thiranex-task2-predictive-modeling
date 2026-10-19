from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT/"models"/"logistic_regression_model.joblib"
st.set_page_config(page_title="Thiranex Task 2 | Predictive Modeling", page_icon="📊", layout="wide")
st.title("Predictive Modeling Using Machine Learning")
st.caption("Thiranex Data Science Internship — Task 2")
st.warning("Educational demonstration only. This model must not be used for medical diagnosis or decisions.")
if not MODEL_PATH.exists():
    st.error("Model not found. Open terminal in this folder and run: python train.py")
    st.stop()
model = joblib.load(MODEL_PATH)
df = load_breast_cancer(as_frame=True).frame.copy()
X, y = df.drop(columns=["target"]), df["target"]
st.subheader("Model Overview")
a,b,c = st.columns(3)
a.metric("Algorithm","Logistic Regression"); b.metric("Input features",X.shape[1]); c.metric("Dataset rows",len(df))
st.write("The model uses an 80/20 stratified train-test split and standardizes input features.")
st.subheader("Try a sample prediction")
idx = st.selectbox("Choose a sample row", list(range(30)))
sample = X.iloc[[idx]]
prediction = int(model.predict(sample)[0])
proba = model.predict_proba(sample)[0]
st.write("**Predicted class:**", "Benign" if prediction == 1 else "Malignant")
st.write(f"**Model probability for benign class:** {proba[1]:.1%}")
st.dataframe(sample, use_container_width=True)
st.subheader("Dataset class distribution")
st.bar_chart(y.map({0:"malignant",1:"benign"}).value_counts())
for filename,title in [("confusion_matrix.png","Confusion Matrix"),("roc_curve.png","ROC Curve"),("feature_importance.png","Random Forest Feature Importance")]:
    path=ROOT/"outputs"/filename
    if path.exists():
        st.subheader(title); st.image(str(path),use_container_width=True)
st.info("If your internship provided an official dataset, replace this educational benchmark dataset before submitting.")
