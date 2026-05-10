import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

df = pd.read_csv("data/sample.csv")
df = df.dropna()

features = [
    "ESCS",
    "HOMEPOS",
    "ICTRES",
    "ICTHOME",
    "ST004D01T",
    "ST005Q01JA",
    "ST006Q01JA",
    "ST006Q02JA",
    "ST006Q03JA",
    "ST007Q01JA",
    "WORKPAY",
    "WORKHOME",
    "MATHPREF",
    "MATHEASE",
    "MATHMOT",
    "BELONG",
    "BULLIED",
    "FAMSUP",
    "TEACHSUP",
    "MATHEFF",
    "FAMCON",
    "ANXMAT",
    "MATHPERS"
]

targets = [
    "PV1MATH",
    "PV2MATH",
    "PV3MATH",
    "PV4MATH",
    "PV5MATH"
]

X = df[features]
y = df[targets]

numeric_features = [
    "ESCS",
    "HOMEPOS",
    "ICTRES",
    "ICTHOME",
    "BELONG",
    "BULLIED",
    "FAMSUP",
    "TEACHSUP",
    "MATHEFF",
    "FAMCON",
    "ANXMAT"
]

categorical_features = [
    "ST004D01T",
    "ST005Q01JA",
    "ST006Q01JA",
    "ST006Q02JA",
    "ST006Q03JA",
    "ST007Q01JA",
    "WORKPAY",
    "WORKHOME",
    "MATHPREF",
    "MATHEASE",
    "MATHMOT",
    "MATHPERS"
]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

xgb = XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=2.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", MultiOutputRegressor(xgb))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/pisa_multi_output_model.pkl")

print("Model saved: models/pisa_multi_output_model.pkl")
