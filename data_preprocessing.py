import pandas as pd
import numpy as np

df1 = pd.read_csv("dataset/combined_flat_dataset 1.csv")
df2 = pd.read_csv("dataset/synthetic_30k_dataset.csv")

print("Dataset 1:", df1.shape)
print("Dataset 2:", df2.shape)

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("dataset/combined_dataset.csv", index=False)
print("Combined dataset saved")

print("Combined dataset:", df.shape)

df = df.drop_duplicates(subset=["appointment_id"])

print("After removing duplicates:", df.shape)

datetime_cols = [
    'booking_created_ts',
    'scheduled_start_ts',
    'scheduled_end_ts',
    'checkin_ts',
    'cancel_ts',
    'service_start_ts',
    'service_end_ts',
    'arrival_ts',
    'created_at'
]

for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df = df.sort_values("scheduled_start_ts")

df["no_show"] = (df["status"] == "no_show").astype(int)

df["actual_wait_time"] = (
    df["service_start_ts"] -
    df["arrival_ts"].fillna(df["scheduled_start_ts"])
).dt.total_seconds() / 60

df["actual_service_duration"] = (
    df["service_end_ts"] - df["service_start_ts"]
).dt.total_seconds() / 60

df["hour"] = df["arrival_ts"].dt.hour
df["day_of_week"] = df["arrival_ts"].dt.dayofweek
df["month"] = df["arrival_ts"].dt.month

df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

df["booking_lead_hours"] = (
    df["scheduled_start_ts"] - df["booking_created_ts"]
).dt.total_seconds() / 3600

df["arrival_delay"] = (
    df["arrival_ts"].fillna(df["scheduled_start_ts"]) -
    df["scheduled_start_ts"]
).dt.total_seconds() / 60

df["arrival_delay"] = df["arrival_delay"].fillna(0)

df["is_peak_hour"] = df["hour"].isin([9,10,11,16,17]).astype(int)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

if "queue_length_at_arrival" in df.columns and "staff_on_duty_at_arrival" in df.columns:

    
    staff = df["staff_on_duty_at_arrival"].replace(0,1)

    df["queue_pressure"] = df["queue_length_at_arrival"] / staff
    df["queue_per_staff"] = df["queue_length_at_arrival"] / staff


if "avg_duration_min" in df.columns:

    df["estimated_backlog_minutes"] = (
        df["queue_length_at_arrival"] *
        df["avg_duration_min"]
    )


if "queue_length_at_arrival" in df.columns and "avg_duration_min" in df.columns:

    staff = df["staff_on_duty_at_arrival"].replace(0, 1)

    df["queue_service_capacity"] = (
        df["queue_length_at_arrival"] *
        df["avg_duration_min"]
    ) / staff

if "queue_length_at_arrival" in df.columns and "avg_duration_min" in df.columns:

    staff = df["staff_on_duty_at_arrival"].replace(0, 1)

    df["dynamic_queue_load"] = (
        df["queue_length_at_arrival"] /
        (staff * df["avg_duration_min"])
    )
    
if "reported_urgency" in df.columns:

    df["urgency_queue"] = (
        df["reported_urgency"] *
        df["queue_length_at_arrival"]
    )

if "avg_duration_min" in df.columns:

    df["service_complexity"] = (
        df["avg_duration_min"] /
        df["avg_duration_min"].max()
    )

if "user_id" in df.columns:

    df["previous_appointments"] = df.groupby("user_id").cumcount()

    df["previous_no_shows"] = (
        df.groupby("user_id")["no_show"]
        .cumsum()
        .shift()
        .fillna(0)
    )

    df["no_show_rate"] = (
        df["previous_no_shows"] /
        df["previous_appointments"].replace(0,1)
    )



df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = [
    "service_name",
    "location_name",
    "preferred_contact",
    "visit_type"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

df = df[df["actual_wait_time"] >= 0]
df = df[df["actual_service_duration"] >= 0]

df = df[df["actual_wait_time"] < 600]
df = df[df["actual_service_duration"] < 600]

categorical_features = [
    "service_id",
    "location_id",
    "booking_channel",
    "age_band",
    "user_type"
]

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

print("\nMissing values summary:")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)

df.to_csv("dataset/processed_queue_dataset.csv", index=False)

print("\nPreprocessing completed successfully!")