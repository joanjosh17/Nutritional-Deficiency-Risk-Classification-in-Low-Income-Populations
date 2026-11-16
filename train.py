import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH, MODEL_PATH, MODELS_DIR, OUTPUTS_DIR,
    BEST_PARAMS_PATH, RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_JOBS
)
from data_loader import load_dataset, split_features_target
from feature_engineering import add_nutrition_features
from preprocessing import build_preprocessor

def main():
    df = load_dataset(DATA_PATH)
    X, y = split_features_target(df)
    X = add_nutrition_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model)
    ])

    param_distributions = {
        "model__n_estimators": [250, 350, 500, 650],
        "model__max_depth": [None, 8, 12, 16, 20, 25],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None]
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    joblib.dump(search.best_estimator_, MODEL_PATH)

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=4)

    test_output = X_test.copy()
    test_output["Nutritional_Risk_Level"] = y_test.values
    test_output.to_csv(OUTPUTS_DIR / "test_set_for_evaluation.csv", index=False)

    print(f"Best CV Macro F1: {search.best_score_:.4f}")
    print("Best parameters:")
    print(json.dumps(search.best_params_, indent=4))
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
