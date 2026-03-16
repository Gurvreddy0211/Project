import joblib
import pandas as pd
import numpy as np


no_show_model = joblib.load("models/random_forest_no_show_model.pkl")
wait_model = joblib.load("models/wait_time_model.pkl")

print("Models loaded successfully")

df = pd.read_csv(
    "dataset/processed_queue_dataset.csv",
    low_memory=False
)

print("Dataset loaded")


no_show_features = [

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


wait_features = [

    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "booking_lead_hours",
    "arrival_delay",

    "queue_length_at_arrival",
    "staff_on_duty_at_arrival",
    "queue_service_capacity",
    "estimated_backlog_minutes",

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

sample = df.sample(1)

print("\nSimulated Appointment Data:")
print(sample[no_show_features].T)

sample["reported_urgency"] = sample["reported_urgency"].replace({
    "low":0,
    "medium":1,
    "high":2
})

sample["reported_urgency"] = sample["reported_urgency"].replace({
    "low": 0,
    "medium": 1,
    "high": 2
})

sample = sample.apply(pd.to_numeric, errors="coerce")

sample = sample.fillna(sample.median())


no_show_prob = no_show_model.predict_proba(
    sample[no_show_features]
)[:, 1][0]


sample_wait = sample[wait_features].copy()


sample_wait = sample_wait.apply(pd.to_numeric, errors="coerce")


sample_wait = sample_wait.fillna(sample_wait.median())

if hasattr(wait_model, "feature_names_in_"):
    sample_wait = sample_wait[wait_model.feature_names_in_]

wait_time = wait_model.predict(sample_wait)[0]


print("\n Smart Queue Prediction ")

print("No-show probability:", round(no_show_prob, 3))
print("Estimated waiting time:", round(wait_time, 2), "minutes")