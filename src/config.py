from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_PATH = DATA_DIR / "nutritional_deficiency_risk_dataset_v2.csv"
MODEL_PATH = MODELS_DIR / "nutritional_deficiency_risk_model.joblib"
METRICS_PATH = OUTPUTS_DIR / "metrics" / "model_comparison.csv"
BEST_PARAMS_PATH = OUTPUTS_DIR / "metrics" / "best_model_params.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1

TARGET_COLUMN = "Nutritional_Risk_Level"
ID_COLUMNS = ["Respondent_ID"]
EXCLUDE_FEATURES = ID_COLUMNS + ["Survey_Year", TARGET_COLUMN]
RISK_CLASSES = ["Low Risk", "Moderate Risk", "High Risk"]
