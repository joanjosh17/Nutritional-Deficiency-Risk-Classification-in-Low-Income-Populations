import joblib
import pandas as pd
from config import MODEL_PATH
from feature_engineering import add_nutrition_features

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def predict_risk(input_data, model=None):
    """
    Predict nutritional risk from a dictionary or DataFrame.
    Returns predicted class and class probabilities.
    """
    if model is None:
        model = load_model()

    if isinstance(input_data, dict):
        X = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        X = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")

    X = add_nutrition_features(X)
    predictions = model.predict(X)

    result = pd.DataFrame({"Predicted_Risk": predictions})

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for i, class_name in enumerate(model.classes_):
            safe_name = class_name.replace(" ", "_")
            result[f"Probability_{safe_name}"] = probabilities[:, i]
        result["Maximum_Risk_Probability"] = probabilities.max(axis=1)

    return result
