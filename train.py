from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
DATA_DIR, OUT_DIR, MODEL_DIR = ROOT/"data", ROOT/"outputs", ROOT/"models"
for folder in (DATA_DIR, OUT_DIR, MODEL_DIR): folder.mkdir(exist_ok=True)

df = load_breast_cancer(as_frame=True).frame.copy()
df.to_csv(DATA_DIR/"breast_cancer_data.csv", index=False)
X, y = df.drop(columns=["target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))])
model.fit(X_train, y_train)
pred, prob = model.predict(X_test), model.predict_proba(X_test)[:, 1]
accuracy, auc = accuracy_score(y_test, pred), roc_auc_score(y_test, prob)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred, target_names=["malignant", "benign"], zero_division=0)
joblib.dump(model, MODEL_DIR/"logistic_regression_model.joblib")

fig, ax = plt.subplots(figsize=(6,5)); ax.imshow(cm)
ax.set_title("Confusion Matrix — Logistic Regression"); ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
ax.set_xticks([0,1], labels=["malignant","benign"]); ax.set_yticks([0,1], labels=["malignant","benign"])
for i in range(2):
    for j in range(2): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
fig.tight_layout(); fig.savefig(OUT_DIR/"confusion_matrix.png",dpi=160); plt.close(fig)

fpr,tpr,_ = roc_curve(y_test,prob)
fig, ax = plt.subplots(figsize=(6,5)); ax.plot(fpr,tpr,label=f"ROC-AUC = {auc:.3f}")
ax.plot([0,1],[0,1],linestyle="--",label="Random baseline")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curve — Logistic Regression"); ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR/"roc_curve.png",dpi=160); plt.close(fig)

rf = RandomForestClassifier(n_estimators=250,random_state=42); rf.fit(X_train,y_train)
importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False).head(10)
fig,ax=plt.subplots(figsize=(8,5)); importance.sort_values().plot(kind="barh",ax=ax)
ax.set_title("Top 10 Feature Importances — Random Forest"); ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig(OUT_DIR/"feature_importance.png",dpi=160); plt.close(fig)

results={"dataset":"Wisconsin Diagnostic Breast Cancer benchmark (scikit-learn)","task":"binary classification",
"train_rows":int(len(X_train)),"test_rows":int(len(X_test)),"accuracy":float(accuracy),"roc_auc":float(auc),
"confusion_matrix":cm.tolist(),"classification_report":report}
(OUT_DIR/"model_evaluation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
(OUT_DIR/"model_evaluation.txt").write_text(
f"Thiranex Internship Task 2 — Predictive Modeling Using Machine Learning\nDataset: {results['dataset']}\n"
f"Train rows: {len(X_train)}\nTest rows: {len(X_test)}\nAccuracy: {accuracy:.4f}\nROC-AUC: {auc:.4f}\n\n"
f"Classification report:\n{report}\nConfusion matrix (rows=true labels; columns=predicted labels):\n{cm}\n"
"Educational benchmark only; not medical advice.\n",encoding="utf-8")
print(f"Training complete. Accuracy={accuracy:.4f}, ROC-AUC={auc:.4f}")
