# ============================================================
# Nutritional Deficiency Risk Classification
# ============================================================
# Project:
# Nutritional-Deficiency-Risk-Classification-in-Low-Income-Populations
#
# Author: Your Name
# ============================================================

# =========================
# Import Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("nutritional_deficiency_risk_dataset.csv")

print("=" * 60)
print("Dataset Shape")
print(df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

# ============================================================
# Exploratory Data Analysis
# ============================================================

print("\nTarget Distribution")
print(df["nutritional_deficiency_risk"].value_counts())

# Risk Distribution Plot

plt.figure(figsize=(8, 5))
sns.countplot(
    x="nutritional_deficiency_risk",
    data=df
)
plt.title("Nutritional Deficiency Risk Distribution")
plt.tight_layout()
plt.show()

# ============================================================
# Income vs Risk
# ============================================================

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="nutritional_deficiency_risk",
    y="monthly_household_income_usd",
    data=df
)
plt.title("Income vs Nutritional Deficiency Risk")
plt.tight_layout()
plt.show()

# ============================================================
# BMI Distribution
# ============================================================

plt.figure(figsize=(10, 6))
sns.histplot(
    df["bmi"],
    bins=30,
    kde=True
)
plt.title("BMI Distribution")
plt.tight_layout()
plt.show()

# ============================================================
# Correlation Heatmap
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# Data Preprocessing
# ============================================================

data = df.copy()

# Encode categorical columns

categorical_cols = [
    "gender",
    "education_level",
    "location_type"
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode Target

target_encoder = LabelEncoder()

data["nutritional_deficiency_risk"] = target_encoder.fit_transform(
    data["nutritional_deficiency_risk"]
)

# ============================================================
# Features and Target
# ============================================================

X = data.drop(
    columns=[
        "person_id",
        "nutritional_deficiency_risk"
    ]
)

y = data["nutritional_deficiency_risk"]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ============================================================
# Random Forest Model
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)

print("\nConfusion Matrix")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# ============================================================
# Confusion Matrix Visualization
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ============================================================
# Feature Importance
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Features")
print(importance.head(15))

# ============================================================
# Feature Importance Plot
# ============================================================

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.show()

# ============================================================
# High Risk Population Analysis
# ============================================================

high_risk = df[
    df["nutritional_deficiency_risk"] == "High"
]

print("\nHigh Risk Population Summary")

print(
    high_risk[
        [
            "monthly_household_income_usd",
            "daily_calorie_intake",
            "daily_protein_intake_g",
            "bmi"
        ]
    ].describe()
)

# ============================================================
# Risk by Location
# ============================================================

risk_location = pd.crosstab(
    df["location_type"],
    df["nutritional_deficiency_risk"]
)

risk_location.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Risk Level by Location")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ============================================================
# Save Model Outputs
# ============================================================

importance.to_csv(
    "feature_importance.csv",
    index=False
)

predictions = pd.DataFrame({
    "Actual": target_encoder.inverse_transform(y_test),
    "Predicted": target_encoder.inverse_transform(y_pred)
})

predictions.to_csv(
    "predictions.csv",
    index=False
)

print("\nFiles Saved:")
print("feature_importance.csv")
print("predictions.csv")

# ============================================================
# Risk Prediction Function
# ============================================================

def predict_risk(sample_data):
    """
    Predict nutritional deficiency risk
    """

    prediction = model.predict(sample_data)

    return target_encoder.inverse_transform(prediction)

print("\nProject Completed Successfully")
