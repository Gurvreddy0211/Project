import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "dataset/processed_queue_dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset Loaded")
print("Dataset shape:", df.shape)


features = [

    
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "booking_lead_hours",
    "arrival_delay",

    "queue_length_at_arrival",
    "staff_on_duty_at_arrival",
    "queue_pressure",
    "queue_per_staff",
    "estimated_backlog_minutes",

    "previous_appointments",
    "previous_no_shows",
    "no_show_rate",

    "service_id",
    "location_id",
    "booking_channel",
    "age_band",
    "user_type",
    "distance_km",
    "reported_urgency",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

target = "no_show"


X = df[features].copy()
y = df[target]


if "reported_urgency" in X.columns:
    X["reported_urgency"] = X["reported_urgency"].replace("Unknown", np.nan)
    X["reported_urgency"] = X["reported_urgency"].fillna(0)


X = X.apply(pd.to_numeric, errors="coerce")


X.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nColumns with missing values:")
print(X.isnull().sum()[X.isnull().sum() > 0])

X = X.fillna(X.median())

if X.isnull().sum().sum() > 0:
    raise ValueError("Dataset still contains NaN values!")

print("Feature matrix:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nTraining Logistic Regression Model...")

log_model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]


print("\n Logistic Regression Results ")

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_log))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))


print("\nTraining Random Forest Model...")

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=18,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]


print("\n Random Forest Results ")

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


feature_importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save feature importance
feature_importance.to_csv("models/feature_importance.csv", index=False)


os.makedirs("models", exist_ok=True)

joblib.dump(log_model, "models/logistic_no_show_model.pkl")
joblib.dump(rf_model, "models/random_forest_no_show_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels saved successfully!")

print("\nTraining completed.")