# Student Performance Evaluation

## PISA 2022 Multi-Output Student Performance Prediction System

Machine Learning project for predicting student mathematics performance using the OECD PISA 2022 dataset.

The system predicts all five mathematics plausible values (`PV1MATH` - `PV5MATH`) using socio-economic, demographic, and learning-related student factors.

---

# Project Overview

This project analyzes how student background, motivation, learning environment, family support, bullying, and mathematics attitudes affect academic performance.

A Multi-Output Machine Learning model was developed using XGBoost and FastAPI to create an intelligent student performance prediction system.

The system provides:

- Multi-output mathematics score prediction
- Average mathematics score estimation
- Student risk level classification
- REST API for real-time predictions
- Swagger API documentation

---

# Technologies

- Python 3
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Joblib
- Pyreadstat

---

# Dataset

Dataset sources:

- OECD PISA 2022 Database
- PISA 2022 Student Questionnaire Dataset

The project uses selected student-related variables including:

- Economic and social status
- Home possessions
- ICT resources
- Family support
- Teacher support
- Mathematics motivation
- Mathematics anxiety
- Bullying experience
- School belonging
- Learning preferences

---

# Machine Learning Architecture

## Input Features

### Numerical Features

- ESCS
- HOMEPOS
- ICTRES
- ICTHOME
- BELONG
- BULLIED
- FAMSUP
- TEACHSUP
- MATHEFF
- FAMCON
- ANXMAT

### Categorical Features

- ST004D01T
- ST005Q01JA
- ST006Q01JA
- ST006Q02JA
- ST006Q03JA
- ST007Q01JA
- WORKPAY
- WORKHOME
- MATHPREF
- MATHEASE
- MATHMOT
- MATHPERS

---

# Target Variables

The system predicts five mathematics plausible values:

- PV1MATH
- PV2MATH
- PV3MATH
- PV4MATH
- PV5MATH

---

# Machine Learning Pipeline

## Preprocessing

- Missing value handling
- Feature selection
- Numerical feature passthrough
- Categorical encoding using `OneHotEncoder`
- Multi-output target preparation

## Model

- MultiOutputRegressor
- XGBoost Regressor
- Pipeline architecture using Scikit-learn

---

# Model Performance

| Metric | Result |
|---|---|
| R² | ~0.56 |
| MAE | ~54.92 |
| RMSE | ~69.12 |

---

# API

The project includes a FastAPI-based prediction service.

## Run API

```bash
uvicorn app:app --reload
```

---

# Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# Example Request

```json
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
```

---

# Example Response

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
├── test.py
├── train_multi_output.py
├── requirements.txt
├── README.md
├── models/
│   └── pisa_multi_output_model.pkl
└── data/
```

---

# Installation

## Create Virtual Environment

```bash
python3 -m venv venv
```

## Activate Virtual Environment

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib pyreadstat
```

---

# Generate Sample Dataset

```bash
python test.py
```

---

# Train Model

```bash
python train_multi_output.py
```

---

# Run FastAPI Server

```bash
uvicorn app:app --reload
```

---

# Author

Developed as an educational data analysis and intelligent decision support system project using the PISA 2022 dataset.
