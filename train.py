import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

df = pd.read_csv("data/sample.csv")
df = df.dropna()

features = [
    "ESCS",
    "HOMEPOS",
    "ICTRES",
    "ST004D01T",
    "ST005Q01JA",
    "ST006Q01JA",
    "ST006Q02JA",
    "ST006Q03JA",
    "ST007Q01JA"
]

targets = [
    "PV1MATH",
    "PV2MATH",
    "PV3MATH",
    "PV4MATH",
    "PV5MATH"
]

X = df[features]
weights = df["W_FSTUWT"]

numeric_features = [
    "ESCS",
    "HOMEPOS",
    "ICTRES"
]

categorical_features = [
    "ST004D01T",
    "ST005Q01JA",
    "ST006Q01JA",
    "ST006Q02JA",
    "ST006Q03JA",
    "ST007Q01JA"
]

preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
])

predictions = []
real_values = []

for target in targets:
    y = df[target]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(
        X_train,
        y_train,
        regressor__sample_weight=w_train
    )

    y_pred = model.predict(X_test)

    predictions.append(y_pred)
    real_values.append(y_test.to_numpy())

final_pred = np.mean(predictions, axis=0)
final_real = np.mean(real_values, axis=0)

print("R2:", r2_score(final_real, final_pred))
print("MAE:", mean_absolute_error(final_real, final_pred))
print("RMSE:", np.sqrt(mean_squared_error(final_real, final_pred)))
joblib.dump(model, "models/pisa_xgboost_model.pkl")
print("Model saved: models/pisa_xgboost_model.pkl")
