# Student Performance Evaluation

PISA 2022 Multi-Output Student Performance Prediction System using Machine Learning, Regression Analysis, Cross-Validation, and Hyperparameter Optimization.

---

# Table of Contents

- Project Overview
- Objectives
- Technologies Used
- Dataset Information
- Feature Engineering
- Machine Learning Architecture
- Regression Models
- Cross Validation
- Hyperparameter Tuning
- Feature Importance Analysis
- Model Evaluation Metrics
- Final Experimental Results
- Visualization and Generated Graphs
- FastAPI Prediction Service
- Example API Requests
- Risk Classification
- Project Structure
- Installation
- Training Pipeline
- Research Findings
- Future Improvements
- Author

---

# Project Overview

This project predicts student mathematics performance using the OECD PISA 2022 dataset.

The system analyzes how educational, psychological, social, and economic factors influence mathematics achievement among students.

The project combines:

- Educational Data Mining
- Machine Learning
- Regression Analysis
- Ensemble Learning
- Hyperparameter Optimization
- API-based Intelligent Prediction Systems

The project was developed as an intelligent educational performance prediction and comparative regression analysis system.

---

# Objectives

The main objectives of this project are:

- Predict student mathematics performance
- Analyze educational and socio-economic factors
- Compare regression algorithms
- Evaluate machine learning performance
- Perform cross-validation
- Apply hyperparameter tuning
- Build an intelligent prediction API
- Analyze important educational variables

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Main programming language |
| Pandas | Data analysis and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning framework |
| XGBoost | Gradient boosting regression |
| FastAPI | REST API framework |
| Uvicorn | FastAPI server |
| Joblib | Model serialization |
| Pyreadstat | Reading SPSS `.sav` files |
| Matplotlib | Graph generation and visualization |

---

# Dataset Information

## Dataset Source

The project uses the:

- OECD PISA 2022 Dataset
- PISA 2022 Student Questionnaire Dataset

## Dataset File

```text
CY08MSP_STU_QQQ.sav
```

## Dataset Description

PISA (Programme for International Student Assessment) evaluates the educational performance of students worldwide.

The dataset contains:

- Student socio-economic information
- Family support indicators
- Educational background
- ICT access
- Mathematics motivation
- Learning attitudes
- School belonging
- Bullying experiences
- Teacher support
- Mathematics anxiety

---

# Feature Engineering

## Numerical Features

```text
ESCS
HOMEPOS
ICTRES
ICTHOME
BELONG
BULLIED
FAMSUP
TEACHSUP
MATHEFF
FAMCON
ANXMAT
```

### Feature Descriptions

| Feature | Description |
|---|---|
| ESCS | Economic, social and cultural status |
| HOMEPOS | Home possessions index |
| ICTRES | ICT resources |
| ICTHOME | ICT access at home |
| BELONG | Sense of school belonging |
| BULLIED | Bullying exposure |
| FAMSUP | Family support |
| TEACHSUP | Teacher support |
| MATHEFF | Mathematics self-efficacy |
| FAMCON | Family connectedness |
| ANXMAT | Mathematics anxiety |

---

## Categorical Features

```text
ST004D01T
ST005Q01JA
ST006Q01JA
ST006Q02JA
ST006Q03JA
ST007Q01JA
WORKPAY
WORKHOME
MATHPREF
MATHEASE
MATHMOT
MATHPERS
```

These features include:

- Gender
- Educational level
- Work conditions
- Mathematics attitudes
- Learning preferences
- Motivation indicators

---

# Target Variables

## Multi-Output Prediction Targets

The main intelligent prediction system predicts:

```text
PV1MATH
PV2MATH
PV3MATH
PV4MATH
PV5MATH
```

These represent plausible mathematics values from the PISA dataset.

---

## Regression Comparison Target

A new regression target was created:

```text
AVG_MATH
```

Formula:

```text
AVG_MATH = (PV1MATH + PV2MATH + PV3MATH + PV4MATH + PV5MATH) / 5
```

This target was used for regression comparison experiments.

---

# Machine Learning Architecture

## Main Prediction System

The main prediction system uses:

- MultiOutputRegressor
- XGBoost Regressor
- FastAPI deployment architecture

The system predicts all five mathematics plausible values simultaneously.

---

# Regression Models

The project compares multiple regression approaches.

## Implemented Models

### 1. Linear Regression

Classical statistical regression baseline model.

Characteristics:

- Simple linear relationships
- High interpretability
- Baseline comparison model

---

### 2. Random Forest Regressor

Ensemble learning model using multiple decision trees.

Characteristics:

- Nonlinear learning capability
- Strong generalization
- Reduced overfitting

---

### 3. Gradient Boosting Regressor

Boosting-based ensemble regression model.

Characteristics:

- Sequential error correction
- Improved prediction accuracy
- Better nonlinear learning

---

### 4. XGBoost Regressor

Advanced optimized gradient boosting framework.

Characteristics:

- High performance
- Regularization support
- Parallel processing
- Superior generalization

---

# Cross Validation

The project uses:

```text
5-Fold Cross Validation
```

Cross-validation was applied to:

- Evaluate model stability
- Measure generalization capability
- Reduce overfitting risk
- Validate experimental consistency

---

# Hyperparameter Tuning

Hyperparameter optimization was performed using:

```text
RandomizedSearchCV
```

## Tuned Models

- Random Forest Regressor
- XGBoost Regressor

## Optimized Parameters

### Random Forest Parameters

- Number of estimators
- Maximum depth
- Minimum samples split
- Minimum samples leaf

### XGBoost Parameters

- Number of estimators
- Maximum depth
- Learning rate
- Subsample ratio
- Column sample ratio

---

# Feature Importance Analysis

Feature importance analysis was performed using the tuned XGBoost model.

## Most Important Features

| Feature | Importance |
|---|---|
| MATHEFF | Mathematics self-efficacy |
| HOMEPOS | Home possessions |
| ESCS | Economic and social status |
| FAMCON | Family connectedness |
| ANXMAT | Mathematics anxiety |

The analysis demonstrates that psychological and socio-economic variables strongly influence mathematics achievement.

---

# Model Evaluation Metrics

The following metrics were used:

## R² Score

Measures prediction quality and explained variance.

## MAE (Mean Absolute Error)

Measures average absolute prediction error.

## RMSE (Root Mean Squared Error)

Measures prediction deviation magnitude.

---

# Final Experimental Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.3804 | 61.30 | 77.66 |
| Random Forest | 0.5384 | 53.06 | 67.03 |
| Gradient Boosting | 0.5618 | 51.67 | 65.31 |
| XGBoost | 0.5754 | 50.83 | 64.29 |
| Random Forest Tuned | 0.5479 | 52.53 | 66.34 |
| XGBoost Tuned | 0.5837 | 50.29 | 63.65 |

## Best Performing Model

```text
Tuned XGBoost Regressor
```

The tuned XGBoost model achieved the highest prediction accuracy and best generalization performance.

---

# Visualization and Generated Graphs

The project automatically generates:

| File | Description |
|---|---|
| final_model_comparison.csv | Final model comparison |
| feature_importance.csv | Feature importance values |
| model_r2_comparison.png | R² comparison graph |
| model_rmse_comparison.png | RMSE comparison graph |
| feature_importance.png | Feature importance visualization |
| hyperparameter_tuning_results.csv | Tuning results |

## Generated Graphs

### 1. Model R² Comparison

Compares prediction quality between regression models.

### 2. Model RMSE Comparison

Compares prediction errors.

### 3. Feature Importance Graph

Visualizes most influential educational factors.

---

# FastAPI Prediction Service

The project includes a FastAPI-based prediction API.

## Run API

```bash
uvicorn app:app --reload
```

## Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# Example API Request

```json
{
  "ESCS": 0.3,
  "HOMEPOS": 0.5,
  "ICTRES": 0.2,
  "ICTHOME": 0.4,
  "ST004D01T": "Female",
  "WORKPAY": "No work for pay",
  "BELONG": 0.5,
  "BULLIED": -0.3,
  "FAMSUP": 0.7,
  "TEACHSUP": 0.6,
  "MATHEFF": 0.4,
  "FAMCON": 0.5,
  "ANXMAT": -0.2
}
```

---

# Example API Response

```json
{
  "PV1MATH": 470.12,
  "PV2MATH": 465.88,
  "PV3MATH": 472.01,
  "PV4MATH": 468.33,
  "PV5MATH": 471.92,
  "average_score": 469.65,
  "risk_level": "MEDIUM_RISK"
}
```

---

# Risk Classification

| Average Score | Risk Level |
|---|---|
| < 400 | HIGH_RISK |
| 400 - 499 | MEDIUM_RISK |
| >= 500 | LOW_RISK |

---

# Project Structure

```text
.
├── app.py
├── train_multi_output.py
├── train_regression_comparison.py
├── hyperparameter_tuning.py
├── feature_importance_analysis.py
├── generate_graphs.py
├── merge_results.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── pisa_multi_output_model.pkl
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── xgboost_model.pkl
│   ├── random_forest_tuned_model.pkl
│   └── xgboost_tuned_model.pkl
│
├── results/
│   ├── final_model_comparison.csv
│   ├── feature_importance.csv
│   ├── model_r2_comparison.png
│   ├── model_rmse_comparison.png
│   ├── feature_importance.png
│   └── hyperparameter_tuning_results.csv
│
└── data/
    └── CY08MSP_STU_QQQ.sav
```

---

# Installation

## Create Virtual Environment

```bash
python3 -m venv venv
```

## Activate Environment

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Training Pipeline

## Train Multi-Output Prediction Model

```bash
python train_multi_output.py
```

## Train Regression Comparison Models

```bash
python train_regression_comparison.py
```

## Perform Hyperparameter Tuning

```bash
python hyperparameter_tuning.py
```

## Generate Feature Importance Analysis

```bash
python feature_importance_analysis.py
```

## Generate Visualization Graphs

```bash
python generate_graphs.py
```

---

# Research Findings

The project demonstrates that:

- Ensemble boosting models outperform classical linear regression
- Educational performance contains nonlinear relationships
- Psychological and socio-economic factors strongly affect achievement
- Hyperparameter tuning improves generalization performance
- XGBoost achieved the best predictive capability

---

# Future Improvements

Possible future improvements include:

- Deep learning architectures
- Explainable AI methods
- Student clustering analysis
- Country-specific educational analysis
- Real-time educational dashboards
- SHAP-based interpretability

---

# Author

Developed by Gurban Suleymanov as a machine learning and educational data mining research project using the OECD PISA 2022 dataset.
