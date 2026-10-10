import pandas as pd
from config import DATA_PATH, TARGET_COLUMN, EXCLUDE_FEATURES

def load_dataset(path=DATA_PATH):
    """Load dataset and perform structural validation."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("The dataset is empty.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    return df

def split_features_target(df):
    """Separate predictors from the target."""
    feature_columns = [c for c in df.columns if c not in EXCLUDE_FEATURES]
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy()

def dataset_summary(df):
    """Return a compact data-quality summary."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True)
    })
