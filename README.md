# Nutritional Deficiency Risk Classification in Low-Income Populations

## Overview

Nutritional deficiencies remain a major public health challenge in low-income populations, contributing to poor health outcomes, impaired cognitive development, reduced productivity, and increased disease susceptibility. Early identification of at-risk individuals can support targeted interventions and improve resource allocation.

This project leverages machine learning techniques to classify individuals into nutritional deficiency risk categories using socioeconomic, dietary, health, and environmental indicators. The solution demonstrates how data-driven approaches can support public health decision-making and nutrition surveillance programs.

---

## Objectives

* Identify individuals at risk of nutritional deficiencies.
* Analyze the relationship between income, diet, health, and nutritional status.
* Build a machine learning classification model for risk prediction.
* Generate actionable insights for policymakers and public health practitioners.
* Visualize key drivers of nutritional deficiency risk.

---

## Problem Statement

Low-income households often face multiple barriers to achieving adequate nutrition, including food insecurity, poor dietary diversity, limited healthcare access, and inadequate sanitation.

Traditional nutritional assessments can be expensive and time-consuming. This project explores the use of machine learning to provide an efficient and scalable method for identifying vulnerable populations and prioritizing interventions.

---

## Dataset

This project uses a realistic synthetic dataset designed to simulate nutritional and socioeconomic conditions commonly observed in vulnerable populations.

### Dataset Features

| Feature                      | Description                       |
| ---------------------------- | --------------------------------- |
| person_id                    | Unique individual identifier      |
| age                          | Age in years                      |
| gender                       | Male/Female                       |
| household_size               | Number of household members       |
| education_level              | Highest education attained        |
| monthly_household_income_usd | Monthly household income          |
| meals_per_day                | Number of daily meals             |
| dietary_diversity_score      | Diversity of food groups consumed |
| daily_calorie_intake         | Average calorie intake            |
| daily_protein_intake_g       | Daily protein consumption         |
| bmi                          | Body Mass Index                   |
| recent_illness               | Recent illness indicator          |
| access_to_clean_water        | Access to safe drinking water     |
| sanitation_score             | Household sanitation score        |
| location_type                | Urban or Rural                    |
| food_insecurity_index        | Household food insecurity measure |
| hemoglobin_g_dl              | Hemoglobin concentration          |
| nutritional_deficiency_risk  | Target variable                   |

### Target Classes

* Low Risk
* Moderate Risk
* High Risk

---

## Project Workflow

### 1. Data Collection

* Synthetic public health dataset generation
* Demographic, nutritional, and environmental indicators

### 2. Data Cleaning

* Missing value assessment
* Data validation
* Feature consistency checks

### 3. Exploratory Data Analysis (EDA)

* Risk distribution analysis
* Income and nutrition assessment
* BMI trends
* Food insecurity patterns
* Correlation analysis

### 4. Feature Engineering

* Categorical encoding
* Feature selection
* Data transformation

### 5. Machine Learning Modeling

Implemented using:

* Random Forest Classifier
* Scikit-learn Pipeline

### 6. Model Evaluation

Performance measured using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### 7. Explainability

* Feature importance analysis
* Risk factor interpretation

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Project Structure

```text
Nutritional-Deficiency-Risk-Classification-in-Low-Income-Populations/

├── data/
│   └── nutritional_deficiency_risk_dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── outputs/
│   ├── feature_importance.csv
│   ├── predictions.csv
│   └── figures/
│
├── src/
│   └── nutritional_deficiency_classifier.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Nutritional-Deficiency-Risk-Classification-in-Low-Income-Populations.git
```

Navigate to the project directory:

```bash
cd Nutritional-Deficiency-Risk-Classification-in-Low-Income-Populations
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the classification pipeline:

```bash
python src/nutritional_deficiency_classifier.py
```

---

## Example Outputs

### Visualizations

* Nutritional Risk Distribution
* Income vs Risk Analysis
* BMI Distribution
* Correlation Heatmap
* Feature Importance Rankings
* Risk by Location Analysis

### Model Outputs

* Classification Report
* Confusion Matrix
* Risk Predictions
* Feature Importance Scores

---

## Key Insights

The project helps identify major predictors of nutritional deficiency risk, including:

* Household income
* Food insecurity
* Dietary diversity
* Protein intake
* Calorie intake
* Hemoglobin levels
* Access to clean water
* BMI

These insights can support targeted nutrition programs and evidence-based public health interventions.

---

## Public Health Applications

Potential applications include:

* Community nutrition screening
* Malnutrition prevention programs
* NGO intervention targeting
* Nutrition surveillance systems
* Resource allocation planning
* Rural health assessment
* Maternal and child health programs

---

## Future Improvements

* Incorporate geospatial analysis and mapping.
* Integrate real-world survey datasets.
* Develop a web-based risk assessment dashboard.
* Implement advanced ensemble learning models.
* Deploy the model using Streamlit or FastAPI.
* Add explainable AI techniques such as SHAP and LIME.

---

## Results

The developed model successfully classifies individuals into nutritional deficiency risk categories using socioeconomic, dietary, and health indicators, demonstrating the potential of machine learning for supporting nutrition-focused public health initiatives.

---

## Disclaimer

This project uses synthetic data generated for educational, research, and portfolio purposes. The dataset does not contain real patient information and should not be used for clinical decision-making.

---

## Author

**Joan Joshua**

Data Scientist | Machine Learning Engineer | Public Health Analytics

---

## License

This project is licensed under the MIT License.
