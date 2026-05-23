Student Performance Evaluation

PISA 2022 Multi-Output Student Performance Prediction System using Machine Learning and Regression Analysis.

This project predicts student mathematics performance using the OECD PISA 2022 dataset and compares multiple regression models using cross-validation and hyperparameter tuning techniques.

Project Overview

The system analyzes how socio-economic, demographic, psychological, and educational factors influence student mathematics achievement.

The project includes:

Multi-output mathematics prediction system
Regression model comparison
Cross-validation analysis
Hyperparameter tuning
Feature importance analysis
REST API using FastAPI
Visualization and evaluation graphs

The project predicts student mathematics performance using both:

Advanced machine learning models
Classical regression approaches
Technologies Used
Python 3
Pandas
NumPy
Scikit-learn
XGBoost
FastAPI
Uvicorn
Joblib
Pyreadstat
Matplotlib
Dataset

Dataset source:

OECD PISA 2022 Database
PISA 2022 Student Questionnaire Dataset

Dataset file:

CY08MSP_STU_QQQ.sav

The project uses student-related variables such as:

Economic and social status
Home possessions
ICT resources
Mathematics anxiety
Family support
Teacher support
Bullying experience
School belonging
Mathematics motivation
Learning preferences
Machine Learning Architecture
Numerical Features
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
Categorical Features
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
Target Variables
Multi-Output Prediction Targets
PV1MATH
PV2MATH
PV3MATH
PV4MATH
PV5MATH
Regression Comparison Target
AVG_MATH

Average mathematics score:

AVG_MATH=
5
PV1MATH+PV2MATH+PV3MATH+PV4MATH+PV5MATH
	​


AVG_MATH=
5
PV1MATH+PV2MATH+PV3MATH+PV4MATH+PV5MATH
	​


Implemented Models
Main Prediction System
MultiOutputRegressor
XGBoost Regressor
Regression Comparison Models
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Tuned Models
Tuned Random Forest
Tuned XGBoost
Cross Validation

The project uses:

5-Fold Cross Validation

Cross-validation was applied to evaluate model generalization performance and stability.

Hyperparameter Tuning

Hyperparameter optimization was performed using:

RandomizedSearchCV

Optimized parameters include:

Number of estimators
Maximum tree depth
Learning rate
Subsample ratio
Minimum sample split
Minimum sample leaf
Final Model Performance
Model	R²	MAE	RMSE
Linear Regression	0.3804	61.30	77.66
Random Forest	0.5384	53.06	67.03
Gradient Boosting	0.5618	51.67	65.31
XGBoost	0.5754	50.83	64.29
Random Forest Tuned	0.5479	52.53	66.34
XGBoost Tuned	0.5837	50.29	63.65

Best performing model:

Tuned XGBoost Regressor
Feature Importance Analysis

Top influential features include:

Mathematics self-efficacy (MATHEFF)
Home possessions (HOMEPOS)
Economic and social status (ESCS)
Family connectedness (FAMCON)
Mathematics anxiety (ANXMAT)

Feature importance analysis was performed using the tuned XGBoost model.

FastAPI Prediction Service

The project includes a FastAPI-based prediction API for real-time student performance prediction.

Run API
uvicorn app:app --reload
Swagger Documentation
http://127.0.0.1:8000/docs
Example Request
{
  "ESCS": 0.3,
  "HOMEPOS": 0.5,
  "ICTRES": 0.2,
  "ICTHOME": 0.4,

  "ST004D01T": "Female",
  "ST005Q01JA": "<ISCED level 3.4>",
  "ST006Q01JA": "Yes",
  "ST006Q02JA": "No",
  "ST006Q03JA": "Yes",
  "ST007Q01JA": "Yes",

  "WORKPAY": "No work for pay",
  "WORKHOME": "Sometimes",
  "MATHPREF": "No preference for mathematics over other subjects",
  "MATHEASE": "Agree",
  "MATHMOT": "Agree",
  "MATHPERS": "Agree",

  "BELONG": 0.5,
  "BULLIED": -0.3,
  "FAMSUP": 0.7,
  "TEACHSUP": 0.6,
  "MATHEFF": 0.4,
  "FAMCON": 0.5,
  "ANXMAT": -0.2
}
Example Response
{
  "PV1MATH": 470.12,
  "PV2MATH": 465.88,
  "PV3MATH": 472.01,
  "PV4MATH": 468.33,
  "PV5MATH": 471.92,
  "average_score": 469.65,
  "risk_level": "MEDIUM_RISK"
}
Risk Classification
Average Score	Risk Level
< 400	HIGH_RISK
400 - 499	MEDIUM_RISK
>= 500	LOW_RISK
Project Structure
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
Installation
Create Virtual Environment
python3 -m venv venv
Activate Environment
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Training Models
Train Multi-Output Prediction System
python train_multi_output.py
Train Regression Comparison Models
python train_regression_comparison.py
Perform Hyperparameter Tuning
python hyperparameter_tuning.py
Generate Feature Importance
python feature_importance_analysis.py
Generate Graphs
python generate_graphs.py
Research Contribution

This project demonstrates that ensemble boosting regression models significantly outperform classical linear regression approaches in predicting student mathematics achievement using PISA 2022 educational data.

The results show that:

nonlinear models better capture educational relationships
socio-economic and psychological variables strongly influence achievement
hyperparameter tuning improves model generalization performance
Author

Developed by Gurban Suleymanov as a machine learning and educational performance prediction research project using the OECD PISA 2022 dataset.
