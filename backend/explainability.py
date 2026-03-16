import shap
import joblib
import numpy as np

no_show_model = joblib.load("models/random_forest_no_show_model.pkl")
wait_model = joblib.load("models/wait_time_model.pkl")

no_show_explainer = shap.TreeExplainer(no_show_model)
wait_explainer = shap.TreeExplainer(wait_model)


def extract_shap_values(shap_values):

    values = np.array(shap_values)

    if values.ndim == 3:
        values = values[0]

    if values.ndim == 2:
        values = values[:, 0]

    return values.flatten()

def explain_no_show(input_df):

    shap_values = no_show_explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    values = extract_shap_values(shap_values)

    explanation = {}

    for feature, val in zip(input_df.columns, values):
        explanation[feature] = float(val)

    
    sorted_features = sorted(
        explanation.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    
    top_10 = dict(sorted_features[:10])

    return top_10

def explain_wait_time(input_df):

    shap_values = wait_explainer.shap_values(input_df)

    values = extract_shap_values(shap_values)

    explanation = {}

    for feature, val in zip(input_df.columns, values):
        explanation[feature] = float(val)

    
    sorted_features = sorted(
        explanation.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_10 = dict(sorted_features[:10])

    return top_10

def generate_human_explanation(wait_time, shap_values):

    top_features = sorted(
        shap_values.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    reasons = []

    for feature, value in top_features:

        if feature == "queue_length_at_arrival":
            reasons.append("queue length is high")

        elif feature == "staff_on_duty_at_arrival":
            reasons.append("staff availability affects service speed")

        elif feature == "queue_pressure":
            reasons.append("queue pressure is increasing waiting time")

        elif feature == "distance_km":
            reasons.append("patient distance affects no-show probability")

        elif feature == "booking_lead_hours":
            reasons.append("long booking lead time influences attendance")

    text = "AI Insight:\n"

    if wait_time > 15:
        text += "The predicted wait time is relatively high because "

    else:
        text += "The predicted wait time is moderate because "

    text += ", ".join(reasons) + "."

    return text