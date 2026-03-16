import pandas as pd
import numpy as np

df = pd.read_csv("combined_flat_dataset 1.csv")

print("Original dataset shape:", df.shape)

df["appointment_id_num"] = df["appointment_id"].str.replace("A", "").astype(int)
last_id = df["appointment_id_num"].max()

print("Last appointment id:", last_id)

num_new_records = 30000

new_ids = [
    f"A{str(i).zfill(6)}"
    for i in range(last_id + 1, last_id + num_new_records + 1)
]

new_rows = []

for i in range(num_new_records):

    
    row = df.sample(1).iloc[0].copy()

    row["appointment_id"] = new_ids[i]

    if "distance_km" in row:
        row["distance_km"] = max(
            0, row["distance_km"] + np.random.normal(0, 1)
        )

    if "queue_length_at_arrival" in row:
        row["queue_length_at_arrival"] = max(
            0,
            row["queue_length_at_arrival"]
            + np.random.randint(-2, 3),
        )

    if "staff_on_duty_at_arrival" in row:
        row["staff_on_duty_at_arrival"] = max(
            1,
            row["staff_on_duty_at_arrival"]
            + np.random.randint(-1, 2),
        )

    new_rows.append(row)

new_df = pd.DataFrame(new_rows)

if "appointment_id_num" in new_df.columns:
    new_df = new_df.drop(columns=["appointment_id_num"])

duplicates = new_df["appointment_id"].duplicated().sum()

print("Duplicate appointment IDs:", duplicates)

new_df.to_csv("synthetic_30k_dataset.csv", index=False)

print("New dataset saved successfully!")
print("New dataset shape:", new_df.shape)