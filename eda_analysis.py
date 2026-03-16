import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

DATA_PATH = "dataset/processed_queue_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded")
print("Shape:", df.shape)


os.makedirs("eda_plots", exist_ok=True)

print("\n Dataset Info ")
print(df.info())

print("\n Dataset Description ")
print(df.describe())

missing = df.isnull().sum()
missing = missing[missing > 0]

print("\nMissing values:")
print(missing)

if len(missing) > 0:
    plt.figure(figsize=(10,5))
    missing.sort_values().plot(kind="barh")
    plt.title("Missing Values")
    plt.savefig("eda_plots/missing_values.png")
    plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x="no_show", data=df)
plt.title("No Show Distribution")
plt.savefig("eda_plots/no_show_distribution.png")
plt.close()

print("\nNo-show percentage:")
print(df["no_show"].value_counts(normalize=True)*100)

plt.figure(figsize=(8,5))
sns.histplot(df["actual_wait_time"], bins=40, kde=True)
plt.title("Waiting Time Distribution")
plt.xlabel("Minutes")
plt.savefig("eda_plots/wait_time_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.histplot(df["queue_length_at_arrival"], bins=30)
plt.title("Queue Length Distribution")
plt.savefig("eda_plots/queue_length_distribution.png")
plt.close()


plt.figure(figsize=(8,6))
sns.scatterplot(
    x="queue_length_at_arrival",
    y="actual_wait_time",
    data=df,
    alpha=0.4
)

plt.title("Queue Length vs Waiting Time")
plt.savefig("eda_plots/queue_vs_wait.png")
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(
    x="staff_on_duty_at_arrival",
    y="actual_wait_time",
    data=df
)

plt.title("Staff vs Waiting Time")
plt.savefig("eda_plots/staff_vs_wait.png")
plt.close()

hourly = df.groupby("hour").size()

plt.figure(figsize=(8,5))
hourly.plot(kind="bar")
plt.title("Appointments by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Count")
plt.savefig("eda_plots/hourly_demand.png")
plt.close()

service_wait = df.groupby("service_id")["actual_wait_time"].mean()

plt.figure(figsize=(8,5))
service_wait.plot(kind="bar")
plt.title("Average Waiting Time by Service")
plt.savefig("eda_plots/service_wait_time.png")
plt.close()

numeric_df = df.select_dtypes(include=["float64","int64"])

corr = numeric_df.corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

corr_wait = corr["actual_wait_time"].sort_values(ascending=False)

print("\nTop correlated features with waiting time:")
print(corr_wait.head(10))

plt.figure(figsize=(8,5))
sns.barplot(
    x="hour",
    y="no_show",
    data=df
)

plt.title("No-show Rate by Hour")
plt.savefig("eda_plots/no_show_by_hour.png")
plt.close()

plt.figure(figsize=(8,6))
sns.boxplot(
    x="no_show",
    y="distance_km",
    data=df
)

plt.title("Distance vs No-show")
plt.savefig("eda_plots/distance_vs_noshow.png")
plt.close()

print("\nEDA Completed. Plots saved in eda_plots/")