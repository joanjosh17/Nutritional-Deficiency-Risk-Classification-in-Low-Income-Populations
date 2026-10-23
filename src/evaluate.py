import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
from config import MODEL_PATH, OUTPUTS_DIR, TARGET_COLUMN

def main():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(OUTPUTS_DIR / "test_set_for_evaluation.csv")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_test, probabilities,
            multi_class="ovr", average="macro",
            labels=model.classes_
        )

    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "final_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame({
        "Actual_Risk": y_test,
        "Predicted_Risk": predictions
    }).to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    main()
