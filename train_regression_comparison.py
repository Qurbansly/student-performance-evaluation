import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


NUMERICAL_FEATURES = [
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

CATEGORICAL_FEATURES = [
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

TARGETS = [
    "PV1MATH",
    "PV2MATH",
    "PV3MATH",
    "PV4MATH",
    "PV5MATH"
]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    return {
        "model": name,
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4)
    }


def main():
    df = pd.read_spss("data/CY08MSP_STU_QQQ.sav")
    
    df["AVG_MATH"] = df[TARGETS].mean(axis=1)

    selected_columns = (
        NUMERICAL_FEATURES +
        CATEGORICAL_FEATURES +
        ["AVG_MATH"]
    )

    df = df[selected_columns].dropna()

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df["AVG_MATH"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES
            )
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
           random_state=42,
           n_estimators=30,
           max_depth=10,
           n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

"XGBoost": XGBRegressor(
    random_state=42,
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    n_jobs=-1
)    
}

    results = []

    for name, regressor in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        result = evaluate_model(
            name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

        print(result)
        
        model_filename = name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(pipeline, f"models/{model_filename}")
        print(f"{name} model saved to models/{model_filename}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/regression_model_comparison_results.csv",
        index=False
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
