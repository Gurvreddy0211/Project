import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

DATA_PATH = "dataset/processed_queue_dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Dataset Loaded")
print("Shape:", df.shape)

os.makedirs("validation_plots", exist_ok=True)


features = [

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

X = df[features]
y = df[target]


X = X.apply(pd.to_numeric, errors="coerce")

X.replace([np.inf, -np.inf], np.nan, inplace=True)

X = X.fillna(X.median())


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("\nRunning 5-Fold Cross Validation...")

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print("\nCross Validation Scores:", scores)
print("Mean ROC-AUC:", scores.mean())
print("Std Dev:", scores.std())


plt.figure(figsize=(6,4))
plt.plot(scores, marker='o')
plt.title("Cross Validation ROC-AUC Scores")
plt.xlabel("Fold")
plt.ylabel("ROC-AUC")
plt.savefig("validation_plots/cross_validation_scores.png")
plt.close()


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)


fpr, tpr, _ = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - No Show Prediction")
plt.legend()
plt.savefig("validation_plots/roc_curve.png")
plt.close()


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("validation_plots/confusion_matrix.png")
plt.close()



importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(
    x="importance",
    y="feature",
    data=importance.head(10)
)

plt.title("Top 10 Important Features")
plt.savefig("validation_plots/feature_importance.png")
plt.close()


print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nValidation graphs saved in folder: validation_plots/")