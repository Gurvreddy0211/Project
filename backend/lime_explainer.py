import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

no_show_model = joblib.load("models/random_forest_no_show_model.pkl")

explainer = None


def explain_lime(input_df):

    global explainer

    feature_names = list(input_df.columns)

    if explainer is None:

        explainer = LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))),
            feature_names=feature_names,
            mode="classification"
        )

    explanation = explainer.explain_instance(
        input_df.values[0],
        no_show_model.predict_proba
    )

    return explanation.as_list()