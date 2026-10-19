# Thiranex Internship — Task 2: Predictive Modeling Using Machine Learning

## Objective
Build a supervised ML model, train/test it, evaluate its performance, and visualize results.

## Dataset
This project uses the Wisconsin Diagnostic Breast Cancer benchmark dataset included with scikit-learn. A CSV copy is in `data/breast_cancer_data.csv`. This is an educational benchmark, not company-provided data or medical advice.

**Important:** If Thiranex supplied a dataset or specified a prediction problem, use that official dataset before submitting. Do not describe this example dataset as provided by Thiranex.

## Algorithms and evaluation
- Logistic Regression classification model with standardized features
- Random Forest for feature-importance visualization
- Stratified 80/20 train/test split
- Accuracy, classification report, confusion matrix, ROC curve and ROC-AUC

## Files
- `train.py` — trains and evaluates models; saves charts and model file
- `app.py` — interactive Streamlit demo
- `data/breast_cancer_data.csv` — dataset copy
- `outputs/` — evaluation report and charts
- `models/` — trained model (included; can be regenerated)

## Run in VS Code
```bash
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Ethical note
This is a learning project only and is not validated for clinical use. Never use it for diagnosis or medical decisions.
