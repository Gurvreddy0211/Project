import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import RandomForestRegressor


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
    "estimated_backlog_minutes",
    "queue_service_capacity",
    
    "service_id",
    "location_id",
    "avg_duration_min",
    "duration_std_min",
    "service_complexity",

    "distance_km",
    "reported_urgency",

    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

target = "actual_wait_time"


X = df[features].copy()
y = df[target]


X = X.apply(pd.to_numeric, errors="coerce")


X.replace([np.inf, -np.inf], np.nan, inplace=True)

X = X.fillna(X.median())

print("Feature matrix:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)



print("\nTraining Waiting Time Model...")

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nWaiting Time Prediction Results")

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop Important Features:")
print(importance.head(10))


os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/wait_time_model.pkl")

importance.to_csv("models/wait_time_feature_importance.csv", index=False)

print("\nWaiting Time Model Saved Successfully!")