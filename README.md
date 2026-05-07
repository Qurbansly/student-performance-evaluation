# student-performance-evaluation

# PISA 2022 Student Performance Prediction

Machine Learning project for predicting student academic performance using the PISA 2022 dataset.

## Project Overview

This project analyzes socio-economic, demographic, and behavioral factors affecting student performance and builds an intelligent decision support system using Machine Learning.

The system predicts student mathematics performance and determines risk levels based on predicted scores.

## Technologies

* Python 3
* Pandas
* Scikit-learn
* XGBoost
* FastAPI
* Joblib

## Dataset

Dataset source:

* PISA 2022 Student Questionnaire Dataset
* OECD PISA 2022 Database

The project uses selected student-related variables such as:

* ESCS (Economic, Social and Cultural Status)
* HOMEPOS (Home Possessions)
* ICTRES (ICT Resources)
* Gender
* Learning-related factors

## Machine Learning Pipeline

### Preprocessing

* Missing value handling
* Feature selection
* Categorical encoding using OneHotEncoder
* Sample generation for efficient training

### Model

* XGBoost Regressor
* Weighted training using PISA student weights (`W_FSTUWT`)
* Multiple plausible values (`PV1MATH - PV5MATH`)

## Evaluation Metrics

| Metric | Result |
| ------ | ------ |
| R²     | ~0.36  |
| MAE    | ~60    |
| RMSE   | ~76    |

## API

The project includes a FastAPI-based prediction service.

### Run API

```bash
uvicorn app:app --reload
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

### Example Request

```json
{
  "ESCS": 0.3,
  "HOMEPOS": 0.5,
  "ICTRES": 0.2,
  "ST004D01T": "Female",
  "ST005Q01JA": "<ISCED level 3.4>",
  "ST006Q01JA": "Yes",
  "ST006Q02JA": "No",
  "ST006Q03JA": "Yes",
  "ST007Q01JA": "Yes"
}
```

### Example Response

```json
{
  "predicted_math_score": 456.14,
  "risk_level": "MEDIUM_RISK"
}
```

## Project Structure

```text
.
├── app.py
├── train.py
├── test.py
├── classify.py
├── requirements.txt
├── models/
│   └── pisa_xgboost_model.pkl
└── data/
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python test.py
python train.py
```

## Author

Developed as an educational data analysis and intelligent decision support system project using PISA 2022 data.
