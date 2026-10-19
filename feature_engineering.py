import numpy as np
import pandas as pd

def add_nutrition_features(X):
    """Create interpretable nutrition, socioeconomic and WASH features."""
    X = X.copy()

    if {"Daily_Protein_Intake_g", "Daily_Calorie_Intake_kcal"}.issubset(X.columns):
        X["Protein_Calorie_Ratio"] = (
            X["Daily_Protein_Intake_g"] /
            X["Daily_Calorie_Intake_kcal"].replace(0, np.nan)
        )

    food_cols = [
        "Fruit_Frequency_Weekly",
        "Vegetable_Frequency_Weekly",
        "Legume_Frequency_Weekly",
        "Animal_Source_Food_Frequency_Weekly",
        "Fortified_Food_Frequency_Weekly"
    ]
    if set(food_cols).issubset(X.columns):
        X["Healthy_Food_Group_Frequency"] = X[food_cols].sum(axis=1)

    if {"Food_Insecurity_Score", "Dietary_Diversity_Score"}.issubset(X.columns):
        X["Food_Insecurity_Dietary_Gap"] = (
            X["Food_Insecurity_Score"] *
            (12 - X["Dietary_Diversity_Score"])
        )

    if {"Clean_Water_Access", "Handwashing_Facility"}.issubset(X.columns):
        X["WASH_Access_Gap"] = (
            (X["Clean_Water_Access"] == "No").astype(int) +
            (X["Handwashing_Facility"] == "No").astype(int)
        )

    if {"Household_Size", "Monthly_Household_Income_NGN"}.issubset(X.columns):
        X["Income_Per_Household_Member"] = (
            X["Monthly_Household_Income_NGN"] /
            X["Household_Size"].replace(0, np.nan)
        )

    if {"Household_Size", "Daily_Protein_Intake_g"}.issubset(X.columns):
        X["Protein_Per_Household_Member"] = (
            X["Daily_Protein_Intake_g"] /
            X["Household_Size"].replace(0, np.nan)
        )

    return X
