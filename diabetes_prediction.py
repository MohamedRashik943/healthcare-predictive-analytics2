# =========================================
# Healthcare Predictive Analytics Project
# Disease Detection - Diabetes Prediction
# =========================================

# 1. Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

import joblib

# 2. Load Dataset
# Download dataset from UCI Machine Learning Repository:
# https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes

df = pd.read_csv("diabetes.csv")

# 3. Basic Information
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nData Description:\n", df.describe())

# 4. Replace invalid zero values with NaN (realistic medical cleaning)
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# 5. Data Visualization
plt.figure(figsize=(5,4))
sns.countplot(x="Outcome", data=df)
plt.title("Diabetes Outcome Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# 6. Split Features and Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 7. Feature Scaling (Normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9. Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

results = {}

# 10. Training and Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n==============================")
    print("Model:", name)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 11. Best Model Selection
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# 12. ROC-AUC Score (for best model if possible)
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test)[:, 1]
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# 13. Feature Importance (Random Forest only)
rf = models["Random Forest"]

importances = rf.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# 14. Save Model
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved successfully!")

# 15. Sample Prediction
sample = np.array([[2, 120, 70, 20, 85, 25.5, 0.5, 30]])
sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)

print("\nSample Prediction Result:")
if prediction[0] == 1:
    print("High Risk of Diabetes")
else:
    print("Low Risk of Diabetes")
