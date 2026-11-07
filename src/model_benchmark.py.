import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from config import CV_FOLDS, DATA_PATH, N_JOBS, RANDOM_STATE, OUTPUTS_DIR
from data_loader import load_dataset, split_features_target
from feature_engineering import add_nutrition_features
from preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline

def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    }

def make_pipeline(model, X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", model)
    ])

def main():
    df = load_dataset(DATA_PATH)
    X, y = split_features_target(df)
    X = add_nutrition_features(X)

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr": "roc_auc_ovr"
    }

    results = []
    for name, model in build_models().items():
        print(f"Benchmarking: {name}")
        start = time.time()
        scores = cross_validate(
            make_pipeline(model, X), X, y, cv=cv,
            scoring=scoring, n_jobs=N_JOBS
        )
        results.append({
            "Model": name,
            "CV_Accuracy": scores["test_accuracy"].mean(),
            "CV_Balanced_Accuracy": scores["test_balanced_accuracy"].mean(),
            "CV_Precision_Macro": scores["test_precision_macro"].mean(),
            "CV_Recall_Macro": scores["test_recall_macro"].mean(),
            "CV_F1_Macro": scores["test_f1_macro"].mean(),
            "CV_F1_Weighted": scores["test_f1_weighted"].mean(),
            "CV_ROC_AUC_OVR": scores["test_roc_auc_ovr"].mean(),
            "Training_Time_Seconds": time.time() - start
        })

    results_df = pd.DataFrame(results).sort_values("CV_F1_Macro", ascending=False)
    metrics_dir = OUTPUTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = metrics_dir / "model_comparison.csv"
    results_df.to_csv(output, index=False)
    print(results_df.to_string(index=False))
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
